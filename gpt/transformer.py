"""The GPT Language Model. From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

import math
import logging
import torch

from block import Block, LayerNorm
from config import Hyperparameters


LOGGER = logging.getLogger(__name__)


class Embedding(torch.nn.Module):
    """Embedding class for the Transformer model."""
    def __init__(self, vocab_size: int, n_embd):
        """
        Embedding class for the Transformer model.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.
        n_embd : int
            The dimension of the embeddings.
        """
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.n_embd = n_embd

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass of the Embedding class.

        Parameters
        ----------
        tokens : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.embedding(tokens.long()) * math.sqrt(self.n_embd)


class GPT(torch.nn.Module):
    """The GPT Language Model."""
    def __init__(self, config):
        """
        Initialize the GPT model.

        Parameters
        ----------
        config : dataclasses.dataclass
            Configuration object (hyperparameters) for the model.
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = torch.nn.ModuleDict({
            "wte": Embedding(config.vocab_size, config.n_embd),
            "wpe": Embedding(config.block_size, config.n_embd),
            "drop": torch.nn.Dropout(config.dropout),
            "h": torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=config.bias),
        })
        self.generator = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.generator.weight

        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        """
        Initialize the weights of a predefined model.

        Parameters
        ----------
        module : torch.nn.Module
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.

        Parameters
        ----------
        idx : torch.Tensor
            Input tensor of shape `(batch_size, sequence_length)`.
        targets : torch.Tensor, optional
            Target tensor of shape `(batch_size, sequence_length)`.

        Returns
        -------
        logits : torch.Tensor
            The output logits of the model.
        loss : torch.Tensor, optional
            The loss of the model, if targets are provided.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Calculate the loss if targets are provided.
        if targets is not None:
            logits = self.generator(x)
            loss = torch.nn.functional.cross_entropy(
                # TODO: Training Q&A instead of next-word.
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=self.config.tokenizer["special_symbols"]["[PAD]"]
            )
            return logits, loss

        # Inference-time mini-optimization: only forward the generator on the very last position.
        logits = self.generator(x[:, [-1], :])  # Note: using list [-1] to preserve the time dim.

        return logits, None

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load a pretrained GPT model from the Huggingface Transformers library.

        Parameters
        ----------
        model_type : str
            The model type to load. One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        override_args : dict, optional
            Override arguments for the model.

        Returns
        -------
        model : GPT
            The pretrained GPT model.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}

        # Only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)

        from transformers import GPT2LMHeadModel  # pylint: disable=import-outside-toplevel
        LOGGER.info("Loading weights from pretrained GPT: %s", model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': {"n_layer": 12, "n_head": 12, "n_embd": 768},   # 124M params
            'gpt2-medium': {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 350M params
            'gpt2-large': {"n_layer": 36, "n_head": 20, "n_embd": 1280},  # 774M params
            'gpt2-xl': {"n_layer": 48, "n_head": 25, "n_embd": 1600},  # 1558M params
        }[model_type]

        LOGGER.info("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257   # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024    # always 1024 for GPT model checkpoints
        config_args['bias'] = True          # always True for GPT model checkpoints

        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # Create a from-scratch initialized karpathy/nanoGPT model.
        config = Hyperparameters(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # Discard this mask / buffer

        # Fetch the huggingface/transformers model.
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all the parameters are aligned and match in names and shapes.
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight',
                      'attn.c_proj.weight',
                      'mlp.c_fc.weight',
                      'mlp.c_proj.weight']

        # OpenAI's checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them.
        assert len(sd_keys_hf) == len(sd_keys), (f"mismatched keys: "
                                                 f"{len(sd_keys_hf)} != {len(sd_keys)}")
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose.
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters.
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Parameters
        ----------
        idx : torch.Tensor
            Input tensor of shape `(batch_size, sequence_length)`.
        max_new_tokens : int
            The number of tokens to generate.
        temperature : float, optional
            The temperature for sampling.
        top_k : int, optional
            The top-k value for sampling.

        Returns
        -------
        idx : torch.Tensor
            The generated indices.

        Notes
        -----
        The temperature is a hyperparameter that controls the randomness of the sampling.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx \
                if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            # TODO: Fix this condition.
            # if idx_next.item() in (self.config.tokenizer["special_symbols"].get("[EOS]", None),
            #                        self.config.tokenizer["special_symbols"]["[SEP]"]):
            #     break

        return idx
