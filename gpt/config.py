"""Sample GPT configuration. From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """
    Configuration for the GPT.

    Attributes
    ----------
    block_size: int, optional
        Size of the blocks.
    vocab_size: int, optional
        Size of the vocabulary.
    n_layer: int, optional
        Number of layers.
    n_head: int, optional
        Number of attention heads.
    n_embd: int, optional
        Dimensionality of embeddings and hidden states.
    dropout: float, optional
        Dropout rate.
    bias: bool, optional
        Whether to use bias in the linear layers.
    """
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
