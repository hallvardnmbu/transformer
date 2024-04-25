"""
Sample generative configuration.

Modified from: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from dataclasses import dataclass, field
from transformers import AutoTokenizer
import torch


@dataclass
class Hyperparameters:
    """
    Hyperparameters for the generative Don Quixote model.

    Attributes
    ----------
    vocab_size : int, optional
        The size of the vocabulary. Default is 24540.
    n_feedforward : int, optional
        The dimension of the feedforward network model. Default is 512.
    n_encoder_layer : int, optional
        The number of encoder layers. Default is 3.
    n_decoder_layer : int, optional
        The number of decoder layers. Default is 3.
    n_head : int, optional
        The number of heads in the multihead attention models. Default is 8.
    n_embd : int, optional
        The dimension of the embeddings. Default is 512.
    dropout : float, optional
        The dropout value. Default is 0.1.
    bias : bool, optional
        Whether to use bias in the model. Default is False.
    epochs : int, optional
        The number of epochs for training. Default is 50.
    batch_size : int, optional
        The batch size for training. Default is 64.
    optimizer : dict[str, int or float], optional
        The optimizer configuration. Default is a dictionary with certain keys.
    scheduler : dict[str, int], optional
        The scheduler configuration. Default is a dictionary with certain keys.
    weight_decay : float, optional
        The weight decay. Default is 1e-1.
    grad_clip : float, optional
        The gradient clipping. Default is 1.0.
    output_path : str, optional
        The path to save the output. Default is './output/'.
    data_path : str, optional
        The path to the data. Default is '../data/cervantes/quixote/quixote_pairs.txt'.
    eval_iters : int, optional
        The number of iterations for evaluation. Default is 1.
    checkpoints : bool, optional
        Whether to save checkpoints. Default is True.
    device : str, optional
        The device to use for training. Default is 'cuda' if available, else 'cpu'.
    dtype : str, optional
        The data type. Default is 'bfloat16' if 'cuda' is available and supports bf16,
        else 'float16'.
    tokenizer : dict[str, int or dict[str, int]], optional
        The tokenizer configuration. Default is a dictionary with certain keys.
    loss_fn : torch.nn.CrossEntropyLoss, optional
        The loss function to use. Default is None.
    optimizer : dict[str, int], optional
        The optimizer configuration. Default is a dictionary with certain keys.
    """
    vocab_size: int = 24540
    n_feedforward: int = 512
    n_encoder_layer: int = 3
    n_decoder_layer: int = 3
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

    epochs: int = 50
    batch_size: int = 64

    optimizer: dict[str, int or float] = field(
        default_factory=lambda: {
            "lr": 1e-3, "betas": (0.9, 0.99), "eps": 1e-9,
        }
    )
    scheduler: dict[str, int] = field(
        default_factory=lambda: {
            "decay_lr": True,
            "warmup": 100, "max": 5000,
            "min_lr": 1e-4,
        }
    )
    weight_decay: float = 1e-1
    grad_clip: float = 1.0

    output_path: str = './output/'
    data_path: str = '../data/cervantes/quixote_pairs.txt'

    eval_iters = 1
    checkpoints = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'

    tokenizer: dict[str, int or dict[str, int]] = field(
        default_factory=lambda: {
            "path": None,  # None: train new tokenizer else path to huggingface tokenizer

            # Only used if `path` is None. Set `bpe_path` to None to train a new tokenizer.
            "bpe_path": '../tokenization/quixote.model',

            # ONLY USED WHEN TRAINING A NEW TOKENIZER:
            # special_symbols: {TOKEN: ID, ...}. should include tokens [PAD], [CLS], [SEP]
            # k: int, optional. Number of characters in dataset to train tokenizer on. None for all.
            "k": None,
            "special_symbols": {
                "[PAD]": 256,
                "[CLS]": 257,
                "[SEP]": 258,
            },

            # DO NOT EDIT:
            # (is set in `__post_init__`)
            "vocab_size": None,
            "tokenizer": None,
        }
    )
    loss_fn: torch.nn.CrossEntropyLoss = None

    optimizer: dict[str, int] = field(
        default_factory=lambda: {
            "lr": 0.0001, "betas": (0.9, 0.98), "eps": 1e-9
        }
    )

    def __post_init__(self):
        """
        Post-initialization of the Hyperparameters class.

        Notes
        -----
        This method is automatically called after the instance has been initialized.
        It sets up the tokenizer and loss function based on the provided configuration.
        """
        if self.tokenizer["path"]:
            self.tokenizer["tokenizer"] = AutoTokenizer.from_pretrained(self.tokenizer["path"])
            self.tokenizer["vocab_size"] = self.vocab_size = self.tokenizer["tokenizer"].vocab_size
            self.tokenizer["special_symbols"] = self.tokenizer["tokenizer"].added_tokens_encoder
        else:
            self.tokenizer["vocab_size"] = self.vocab_size

        self.loss_fn = torch.torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer["special_symbols"]["[PAD]"]
        )
