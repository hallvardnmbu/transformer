"""
Attention mechanism for the Transformer model.

Based on the implementations from Hugging Face and Andrej Karpathy:

https://github.com/huggingface/transformers/
https://github.com/karpathy/nanoGPT/
"""

import logging
import torch

LOGGER = logging.getLogger(__name__)


class Attention(torch.nn.Module):
    """Attention mechanism for the Transformer model."""
    def __init__(self, config):
        super().__init__()

        LOGGER.info("[Attention]: Initializing with config: %s", config.__dict__)

        self.parameter = {
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "dropout": config.dropout,
            "bias": config.bias,
            "flash": hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        }

        # Input projection of `query`, `key`, `value` (therefore `3 * config.n_embd`).
        self.c_attn = torch.nn.Linear(self.parameter["n_embd"], 3 * self.parameter["n_embd"],
                                      bias=self.parameter["bias"])

        # Output projection.
        self.c_proj = torch.nn.Linear(self.parameter["n_embd"], self.parameter["n_embd"],
                                      bias=self.parameter["bias"])

        # Regularization.
        self.attn_dropout = torch.nn.Dropout(self.parameter["dropout"])
        self.resid_dropout = torch.nn.Dropout(self.parameter["dropout"])

        # Checking if Flash Attention is available.
        if not self.flash:
            LOGGER.warning("[Attention]: Flash Attention requires PyTorch >= 2.0. "
                           "Defaulting to slow attention.")

            # Only apply attention to the left side of the sequence.
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_positions, config.n_positions))
                .view(1, 1, config.n_positions, config.n_positions)
            )

        LOGGER.debug("[Attention]: Initialized. in: %s -> out: %s", self.c_attn, self.c_attn)

    def forward(self, inputs):
        """
        Forward pass of the attention mechanism.

        Parameters
        ----------
        inputs: torch.Tensor
            Input tensor of shape `(batch_size, sequence_length, embedding_dim)`.
        """
        B, T, C = inputs.size()

        LOGGER.debug("[Attention]: Forward pass. B=%s, T=%s, C=%s", B, T, C)

        # query, key, values for all heads in batch and move head forward to be the batch dim
        # (B, nh, T, hs)
        q, k, v = self.c_attn(inputs).split(self.parameter["n_embd"], dim=2)
        k = k.view(B, T, self.parameter["n_head"], C // self.parameter["n_head"]).transpose(1, 2)
        q = q.view(B, T, self.parameter["n_head"], C // self.parameter["n_head"]).transpose(1, 2)
        v = v.view(B, T, self.parameter["n_head"], C // self.parameter["n_head"]).transpose(1, 2)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels.
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.parameter["dropout"] if self.training else 0,
                is_causal=True,
                scale=None
            )
        else:
            # Manual implementation of attention.
            att = (q @ k.transpose(-2, -1)) * (1.0 / torch.tensor(torch.sqrt(k.size(-1))))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v

        # Re-assemble all head outputs side by side.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection.
        y = self.resid_dropout(self.c_proj(y))
        return y


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config.attention import AttentionConfig

    attention = Attention(AttentionConfig)
