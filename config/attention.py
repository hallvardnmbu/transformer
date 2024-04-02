"""Sample attention configuration."""

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
    def __init__(
            self,
            n_embd: int = 768,
            n_head: int = 12,
            n_positions: int = 1024,
            bias: bool = True,
            dropout: float = 0.1,
            ):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.n_positions = n_positions
        self.bias = bias
        self.dropout = dropout
