"""From Karpathy; https://github.com/karpathy/nanoGPT/blob/master/model.py"""

import math
import logging
import torch

LOGGER = logging.getLogger(__name__)


class LayerNorm(torch.nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        """
        Initialize the LayerNorm class.

        Parameters
        ----------
        ndim : int
            The number of dimensions for the layer normalization.
        bias : bool
            Whether to include bias in the layer normalization.
        """
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, inputs):
        """
        Forward pass of the LayerNorm class.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after the forward pass.
        """
        return torch.nn.functional.layer_norm(inputs,
                                              self.weight.shape, self.weight,
                                              self.bias, 1e-5)


class CausalSelfAttention(torch.nn.Module):
    """Self-attention layer with a causal mask."""
    def __init__(self, config):
        """
        Initialize the CausalSelfAttention class.

        Parameters
        ----------
        config
            The configuration object containing model hyperparameters.
        """
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            block = config.block_size
            self.register_buffer("bias",
                                 torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x):
        """
        Forward pass of the CausalSelfAttention class.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after the forward pass.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the
        # batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(torch.nn.Module):
    """MLP for transformer."""
    def __init__(self, config):
        """
        Initialize the MLP class.

        Parameters
        ----------
        config
            The configuration object containing model hyperparameters.
        """
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the MLP class.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after the forward pass.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(torch.nn.Module):
    """Transformer block."""
    def __init__(self, config):
        """
        Initialize the Block class.

        Parameters
        ----------
        config
            The configuration object containing model hyperparameters.
        """
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Forward pass of the Block class.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after the forward pass.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
