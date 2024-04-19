"""The GPT Language Model. From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

import math
import torch

from block import Block, LayerNorm


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

        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.generator(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=self.config.tokenizer["special_symbols"]["[PAD]"]
            )
            return logits, loss

        logits = self.generator(x[:, [-1], :])  # Note: using list [-1] to preserve the time dim.

        return logits, None

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
        idx = idx.to(self.config.device)
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size.
            idx_cond = idx \
                if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next.item() in (self.config.tokenizer["special_symbols"].get("[EOS]", None),
                                   self.config.tokenizer["special_symbols"]["[SEP]"]):
                break

        return idx
