"""
Sample GPT configuration.

From https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
