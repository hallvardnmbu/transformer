"""From Karpathy; https://github.com/karpathy/nanoGPT/blob/master/model.py"""

import logging
import torch

from attention.attention import Attention


LOGGER = logging.getLogger(__name__)


class LayerNorm(torch.nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""
    def __init__(self, ndim, bias):
        """
        Custom LayerNorm module supporting bias.

        Parameters
        ----------
        ndim : int
        bias : bool
        """
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        """Forward pass of the LayerNorm module given an input."""
        LOGGER.debug("[LayerNorm]: Forward pass with input: %s", x)
        return torch.nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, 1e-5
        )


class MLP(torch.nn.Module):
    """MLP for the Transformer model."""
    def __init__(self, config):
        """
        Initialize the MLP module.

        Parameters
        ----------
        config : dataclasses.dataclass
            Configuration object (hyperparameters) for the model.
        """
        super().__init__()

        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        """Forward pass of the MLP module given an input."""
        LOGGER.debug("[MLP]: Forward pass with input: %s", x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(torch.nn.Module):
    """Block for the Transformer model."""
    def __init__(self, config):
        """
        Create a new block for the Transformer model.

        Parameters
        ----------
        config : dataclasses.dataclass
            Configuration object (hyperparameters) for the model.
        """
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        """Forward pass of the block given an input."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
