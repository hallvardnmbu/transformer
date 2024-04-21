"""Translator model based on the Transformer architecture."""

import os
import time
import random
import logging
import torch
import datasets

from tokenization.bpe import RegexTokenizer

from config import Hyperparameters
from transformer import Transformer


os.makedirs(Hyperparameters.output_path, exist_ok=True)
handler = logging.FileHandler(Hyperparameters.output_path + 'info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)


class Translator(torch.nn.Module):
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
        self._setup_metrics()

        self.tokenizer = self._tokenizer()
        assert self.tokenizer.vocab_size == config.vocab_size

        self.loss_fn = config.loss_fn
        self.transformer = Transformer(config)
        self.transformer.to(config.device)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), **config.optimizer)

    def _setup_metrics(self):
        """Create the metric file(s) for the model – to be filled during training."""
        with open(os.path.join(self.config.output_path, "loss.csv"), "w") as loss:
            loss.write("epoch,train_loss,val_loss\n")
        if LOGGER.isEnabledFor(logging.DEBUG):
            with open(os.path.join(self.config.output_path, "debug.csv"), "w") as debug:
                debug.write("epoch,iteration,train_loss_batch\n")

    def _tokenizer(self):
        """
        Create a tokenizer for the model.

        Returns
        -------
        tokenization.bpe.RegexTokenizer or transformers.AutoTokenizer
        """
        if self.config.tokenizer["tokenizer"]:
            return self.config.tokenizer["tokenizer"]

        source, target = self._data()
        source, target = source["train"]["sentence"], target["train"]["sentence"]

        k = min(len(source), len(target), self.config.tokenizer["k"])
        LOGGER.info("Training a custom tokenizer based on %s sample sentences.", 2*k)

        tokenizer = RegexTokenizer()
        tokenizer.add_special_tokens(self.config.tokenizer["special_symbols"])
        tokenizer.train(
            text=random.sample(source, k=k) + random.sample(target, k=k),
            vocab_size=self.config.tokenizer["vocab_size"]
        )
        LOGGER.info("> Success.\n")
        return tokenizer

    def _data(self):
        """
        Load the data from huggingface for the model.

        Returns
        -------
        datasets.dataset_dict.DatasetDict, datasets.dataset_dict.DatasetDict
            The data for the two languages.
        """
        LOGGER.info("Loading data from %s for %s.",
                    self.config.data_path, self.config.data_lang)

        data = datasets.load_dataset(self.config.data_path, self.config.data_lang)
        lang1, lang2 = self.config.data_lang.split("-")

        source = datasets.DatasetDict({
            split: datasets.Dataset.from_dict({
                "sentence": [row[lang1] for row in contents['translation']],
                "tokenized": [self.tokenizer.encode(row[lang1]) for row in contents['translation']]
            })
            for split, contents in data.items()
        })
        target = datasets.DatasetDict({
            split: datasets.Dataset.from_dict({
                "sentence": [row[lang2] for row in contents['translation']],
                "tokenized": [self.tokenizer.encode(row[lang2]) for row in contents['translation']]
            })
            for split, contents in data.items()
        })

        LOGGER.info("> Success.\n")

        return source, target

    @torch.no_grad()
    def __call__(self, text, margin=10):
        """
        Translate the input text.

        Parameters
        ----------
        text : str
            The text to be translated.
        margin : int, optional
            An added number of tokens that may be generated (length of the input text + `margin`).

        Returns
        -------
        str
            The translated text.
        """
        self.transformer.eval()
        device = self.config.device

        src = torch.tensor(self.tokenizer.encode(text)).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = self.transformer.encode(src, src_mask)
        out = torch.ones(1, 1).fill_(
            self.config.tokenizer["special_symbols"]["[CLS]"]
        ).type(torch.long).to(device)

        for i in range(num_tokens + margin):
            memory = memory.to(device)
            tgt_mask = (self.square_mask(out.size(0)).type(torch.bool)).to(device)

            _out = self.transformer.decode(out, memory, tgt_mask)
            _out = _out.transpose(0, 1)
            prob = self.transformer.generator(_out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            out = torch.cat([out, torch.ones(1, 1).type_as(src.data).fill_(next_word)],
                            dim=0)
            if next_word in (self.config.tokenizer["special_symbols"].get("[EOS]", None),
                             self.config.tokenizer["special_symbols"]["[SEP]"]):
                break

        return self.tokenizer.decode(out.flatten().tolist(), skip_special_tokens=True)

    def epoch(self, source, target):
        """
        Train the model for one epoch.

        Parameters
        ----------
        source : datasets.arrow_dataset.Dataset
            The source data.
        target : datasets.arrow_dataset.Dataset
            The target data.

        Returns
        -------
        float
            The loss for the epoch.
        """
        self.transformer.train()
        losses = 0

        for i, (src, tgt) in enumerate(zip(source.iter(self.config.batch_size),
                                           target.iter(self.config.batch_size))):
            src = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in src["tokenized"]],
                padding_value=self.config.tokenizer["special_symbols"]["[PAD]"]
            )
            tgt = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in tgt["tokenized"]],
                padding_value=self.config.tokenizer["special_symbols"]["[PAD]"]
            )

            src = src.to(self.config.device)
            tgt = tgt.to(self.config.device)

            tgt_input = tgt[:-1, :]

            src_m, tgt_m, src_pad_m, tgt_pad_m = self.masking(src, tgt_input)

            logits = self.transformer(src, tgt_input, src_m, tgt_m, src_pad_m, tgt_pad_m, src_pad_m)

            self.optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            losses += loss.item() / src.shape[0]

            if LOGGER.isEnabledFor(logging.DEBUG):
                with open(os.path.join(self.config.output_path, "debug.csv"), "a") as debug:
                    debug.writelines(f",{i},{loss}\n")

        return losses

    @torch.no_grad()
    def evaluate(self, source, target):
        """
        Evaluate the model.

        Parameters
        ----------
        source : datasets.arrow_dataset.Dataset
            The source data.
        target : datasets.arrow_dataset.Dataset
            The target data.

        Returns
        -------
        float
            The loss for the evaluation.
        """
        LOGGER.debug("> Evaluating the model.")

        self.transformer.eval()
        losses = 0

        for i, (src, tgt) in enumerate(zip(source.iter(self.config.batch_size),
                                           target.iter(self.config.batch_size))):
            src = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in src["tokenized"]],
                padding_value=self.config.tokenizer["special_symbols"]["[PAD]"]
            )
            tgt = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in tgt["tokenized"]],
                padding_value=self.config.tokenizer["special_symbols"]["[PAD]"]
            )

            src = src.to(self.config.device)
            tgt = tgt.to(self.config.device)

            tgt_input = tgt[:-1, :]

            src_m, tgt_m, src_pad_m, tgt_pad_m = self.masking(src, tgt_input)

            logits = self.transformer(src, tgt_input, src_m, tgt_m, src_pad_m, tgt_pad_m, src_pad_m)

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            losses += loss.item() / src.shape[0]

        return losses

    def learn(self, checkpoints=True, sentence=None):
        """
        Train the model for a specified number of epochs.

        Parameters
        ----------
        checkpoints : bool, optional
            Whether to save the model after each epoch.
        sentence : str, optional
            A sentence to translate during training to visualise progress.
        """
        LOGGER.info("\nTraining the model for %s epochs.\n", self.config.epochs)

        checkpoint = self.config.epochs // 10 if checkpoints else self.config.epochs
        source, target = self._data()

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            train_loss = self.epoch(source["train"], target["train"])
            end_time = time.time()
            val_loss = self.evaluate(source["validation"], target["validation"])

            LOGGER.info("\nEpoch: %s, Train loss: %s, Val loss: %s, Epoch time = %ss",
                        epoch, train_loss, val_loss, round(end_time - start_time, 3))

            with open(os.path.join(self.config.output_path, "loss.csv"), "a") as loss:
                loss.writelines(f"{epoch},{train_loss},{val_loss}\n")
            if LOGGER.isEnabledFor(logging.DEBUG):
                with open(os.path.join(self.config.output_path, "debug.csv"), "a") as debug:
                    debug.writelines(f"{epoch},,\n")

            if epoch % checkpoint == 0:
                LOGGER.info(f"> Saving checkpoint as '{self.config.output_path}model-{epoch}.pth'.")
                torch.save(self, os.path.join(self.config.output_path, f"model-{epoch}.pth"))

            if sentence and epoch % checkpoint == 0:
                LOGGER.info("> Translation of '%s' after epoch %s:\n  %s",
                            sentence, epoch, self(sentence))

    def square_mask(self, dim):
        """
        Create a square mask with the given dimension.

        Parameters
        ----------
        dim : int
            The dimension for the square mask.

        Returns
        -------
        torch.Tensor
            The square mask.
        """
        mask = (torch.triu(torch.ones((dim, dim), device=self.config.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def masking(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        src_m = torch.zeros((src_seq_len, src_seq_len), device=self.config.device).type(torch.bool)
        tgt_m = self.square_mask(tgt_seq_len).to(self.config.device)

        src_pad_m = (src == self.config.tokenizer["special_symbols"]["[PAD]"]).transpose(0, 1).to(
            self.config.device)
        tgt_pad_m = (tgt == self.config.tokenizer["special_symbols"]["[PAD]"]).transpose(0, 1).to(
            self.config.device)

        return src_m, tgt_m, src_pad_m, tgt_pad_m
