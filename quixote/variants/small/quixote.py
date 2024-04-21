"""Based on: https://github.com/karpathy/nanoGPT/blob/master/train.py"""

import os
import time
import math
import random
import inspect
import logging
from contextlib import nullcontext
import torch

from config import Hyperparameters
from transformer import GPT
from tokenization.bpe import RegexTokenizer


os.makedirs(Hyperparameters.output_path, exist_ok=True)
handler = logging.FileHandler(Hyperparameters.output_path + "/info.txt")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

torch.manual_seed(2409)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Quixote(torch.nn.Module):
    """The Quixote model wrapper."""
    def __init__(self, config=Hyperparameters()):
        """
        Initialize the Translator class.

        Parameters
        ----------
        config : Hyperparameters, optional
            Configuration parameters for the Translator.
        """
        super().__init__()

        self.config = config

        ptdtype = {
            'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16
        }[config.dtype]
        self.ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(
            device_type=config.device, dtype=ptdtype
        )

        self._setup_metrics()

        self.tokenizer = self._tokenizer()
        assert self.tokenizer.vocab_size == config.vocab_size

        self.transformer = GPT(config)
        self.transformer.to(config.device)
        self.optimizer = self._optimizer()

        self._data()

    def _setup_metrics(self):
        """Create the metric file(s) for the model – to be filled during training."""
        with open(os.path.join(self.config.output_path, "loss.csv"), "w") as loss:
            loss.write("epoch,train_loss,val_loss\n")

    def _optimizer(self):
        """
        Configure the optimizer for the model.

        Returns
        -------
        optimizer : torch.optim.AdamW
            The AdamW optimizer.
        """
        param_dict = {pn: p for pn, p in self.transformer.named_parameters() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # I.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        LOGGER.info("Num decayed parameter tensors: %s, with %s parameters",
                    len(decay_params), num_decay_params)
        LOGGER.info("Num non-decayed parameter tensors: %s, with %s parameters",
                    len(nodecay_params), num_nodecay_params)

        # Create AdamW optimizer and use the fused version if it is available.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.config.device == 'cuda'
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, **self.config.optimizer, **extra_args)

        LOGGER.info("Using fused AdamW: %s", use_fused)

        return optimizer

    def _tokenizer(self):
        """
        Create a tokenizer for the model.

        Returns
        -------
        tokenization.bpe.RegexTokenizer or transformers.AutoTokenizer
        """
        if self.config.tokenizer["tokenizer"]:
            return self.config.tokenizer["tokenizer"]

        data = self._data(tokenizer=True)

        k, _k = self.config.tokenizer["k"], None
        if isinstance(k, int):
            k = min(len(data), k)
            _k = random.randint(0, len(data) - k - 1)

        tokenizer = RegexTokenizer()
        if self.config.tokenizer["bpe_path"]:
            tokenizer.load(self.config.tokenizer["bpe_path"])
            return tokenizer

        LOGGER.info("Training a custom tokenizer based on %s sentences.",
                    f"{k} sample" if k else "all available")

        tokenizer.add_special_tokens(self.config.tokenizer["special_symbols"])
        tokenizer.train(
            text=data[_k:_k + k] if k else data,
            vocab_size=self.config.tokenizer["vocab_size"]
        )
        LOGGER.info("> Saving the tokenizer to '%s/tokenizer.*'.", self.config.output_path)
        tokenizer.save(os.path.join(self.config.output_path, "tokenizer"))
        LOGGER.info("> Success.\n")
        return tokenizer

    def _data(self, tokenizer=False):
        """
        Load the data from huggingface for the model.

        Parameters
        ----------
        tokenizer : bool, optional
            Whether to load the data for the tokenizer.

        Returns
        -------
        str or None
            The full string of the data. Only if `tokenizer` is True.

        Note
        ----
        The suffixes are defined in the `Hyperparameters` class, as `from_lang` and `to_lang`.
        """
        LOGGER.info("Loading data from %s.", self.config.data_path)

        with open(self.config.data_path, "r") as file:
            text = file.readline()

        if tokenizer:
            return text

        self.data = self.tokenizer.encode(text, allowed_special="none")
        LOGGER.info("> Success.\n")
        return None

    @torch.no_grad()
    def forward(self, text, generate=10, temperature=1.0, top_k=None):
        """
        Translate the input text.

        Parameters
        ----------
        text : str
            The text to be translated.
        generate : int, optional
            Number of tokens to generate.
        temperature : float, optional
            The temperature for the softmax.
        top_k : int, optional
            The number of top-k tokens to sample from.

        Returns
        -------
        str
            The translated text.
        """
        self.transformer.eval()

        prompt = torch.tensor([self.tokenizer.encode(text)[:-1]])

        generated = self.transformer.generate(
            prompt, generate, temperature, top_k,
        )

        return self.tokenizer.decode(generated.flatten().tolist(), skip_special_tokens=True)

    def get_batch(self):
        """
        Get a batch of data for training.

        Returns
        -------
        torch.Tensor, torch.Tensor
        """
        ix = torch.randint(len(self.data) - self.config.block_size, (self.config.batch_size,))

        x = torch.stack([torch.tensor(self.data[i:i + self.config.block_size]) for i in ix])
        y = torch.stack([torch.tensor(self.data[i + 1:i + 1 + self.config.block_size]) for i in ix])

        if self.config.device == 'cuda':
            x, y = (x.pin_memory().to(self.config.device, non_blocking=True),
                    y.pin_memory().to(self.config.device, non_blocking=True))
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)

        return x, y

    def learn(self, checkpoints=True, sentence=None):
        """
        Train the model for a specified number of epochs.

        Parameters
        ----------
        checkpoints : bool, optional
            Whether to save the model every `epochs // 10`.
        sentence : str, optional
            A sentence to translate during training to visualise progress.
        """
        LOGGER.info("\nTraining the model for %s epochs.\n", self.config.epochs)

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.dtype == 'float16')

        checkpoint = self.config.epochs // 10 if checkpoints else self.config.epochs

        x, y = self.get_batch()
        for epoch in range(1, self.config.epochs + 1):
            if self.config.scheduler["decay_lr"]:
                lr = self.get_lr(epoch)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            start_time = time.time()

            for _ in range(self.config.micro_steps):
                with self.ctx:
                    _, loss = self.transformer(x, y)
                    loss = loss / self.config.micro_steps
                x, y = self.get_batch()
                scaler.scale(loss).backward()

            if self.config.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), self.config.grad_clip)

            scaler.step(self.optimizer)
            scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            end_time = time.time()
            loss = loss.item() * self.config.micro_steps

            LOGGER.info("\nEpoch: %s, Train loss: %s, Epoch time = %ss",
                        epoch, loss, round(end_time - start_time, 3))

            with open(os.path.join(self.config.output_path, "loss.csv"), "a") as metrics:
                metrics.writelines(f"{epoch},{loss},{self.evaluate()}\n")

            if epoch % checkpoint == 0:
                LOGGER.info("> Saving checkpoint as '%smodel-%s.pth'.",
                            self.config.output_path, epoch)
                torch.save(self, self.config.output_path + f"model-{epoch}.pth")

            if sentence and epoch % checkpoint == 0:
                LOGGER.info("> Continuation of '%s' after epoch %s:\n  %s",
                            sentence, epoch, self(sentence))

    def get_lr(self, it):
        """
        Calculate the learning rate based on the iteration number.

        This method uses linear warmup for initial iterations, then switches to cosine decay.

        Parameters
        ----------
        it : int
            The current iteration number.

        Returns
        -------
        float
            The calculated learning rate for the current iteration.
        """
        # 1) linear warmup for warmup_iters steps
        if it < self.config.scheduler["warmup"]:
            lr = self.config.optimizer["lr"] * it / self.config.scheduler["warmup"]
            self.config.optimizer["lr"] = lr
            return lr

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.scheduler["max"]:
            return self.config.scheduler["min_lr"]

        # 3) in between, use cosine decay down to min learning rate
        warmup = self.config.scheduler["warmup"]
        decay_ratio = (it - warmup) / (self.config.scheduler["max"] - warmup)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        min_lr = self.config.scheduler["min_lr"]
        lr = min_lr + coeff * (self.config.optimizer["lr"] - min_lr)
        self.config.optimizer["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate the model by calculating the mean loss over a number of iterations.

        This method sets the model to evaluation mode, performs the evaluation, then sets the
        model back to training mode.

        Returns
        -------
        float
            The mean loss over the evaluation iterations.
        """
        self.transformer.eval()
        losses = torch.zeros(self.config.eval_iters)
        for k in range(self.config.eval_iters):
            x, y = self.get_batch()
            with self.ctx:
                _, loss = self.transformer(x, y)
            losses[k] = loss.item()
        losses = losses.mean().item()
        self.transformer.train()
        return losses
