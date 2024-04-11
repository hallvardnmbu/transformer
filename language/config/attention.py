"""Sample attention configuration."""

from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """
    Configuration for the Attention mechanism.

    Attributes
    ----------
    n_embd: int, optional
        Dimensionality of embeddings and hidden states.
    n_head: int, optional
        Number of attention heads.
    n_positions: int, optional
        Maximum number of positions in the input sequences.
    bias: bool, optional
        Whether to use bias in the linear layers.
    dropout: float, optional
        Dropout rate.
    """
    n_embd: int = 768
    n_head: int = 12
    n_positions: int = 1024
    bias: bool = True
    dropout: float = 0.1
