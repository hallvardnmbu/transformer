"""Sample Kafka configuration."""

from dataclasses import dataclass, field
from transformers import AutoTokenizer
import torch


@dataclass
class Hyperparameters:
    """
    A class used to represent the hyperparameters of the Kafka model.

    Attributes
    ----------
    vocab_size : int
        the size of the vocabulary
    n_feedforward : int
        the number of feedforward layers
    n_encoder_layer : int
        the number of encoder layers
    n_decoder_layer : int
        the number of decoder layers
    n_head : int
        the number of attention heads
    n_embd : int
        the dimension of the embeddings
    dropout : float
        the dropout rate
    bias : bool
        whether to use bias in the model
    epochs : int
        the number of training epochs
    batch_size : int
        the batch size for training
    optimizer : dict
        the optimizer configuration
    scheduler : dict
        the scheduler configuration
    weight_decay : float
        the weight decay for the optimizer
    grad_clip : float
        the gradient clipping threshold
    output_path : str
        the path to save the output
    data_path : str
        the path to the data
    eval_iters : int
        the number of iterations between evaluations
    checkpoints : bool
        whether to save checkpoints during training
    device : str
        the device to use for training
    dtype : str
        the data type to use for the tensors
    tokenizer : dict
        the tokenizer configuration
    loss_fn : torch.nn.CrossEntropyLoss
        the loss function for training
    """
    vocab_size: int = 7000
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
    data_path: str = './data/kafka_pairs.txt'

    eval_iters = 2
    checkpoints = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'

    tokenizer: dict[str, int or dict[str, int]] = field(
        default_factory=lambda: {
            "path": None,  # None: train new tokenizer else path to huggingface tokenizer

            # Only used if `path` is None. Set `bpe_path` to None to train a new tokenizer.
            "bpe_path": '../tokenization/kafka/kafka.model',

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
        Initializes the tokenizer and loss function after the instance is created.

        Notes
        -----
        If a tokenizer path is provided, it loads the tokenizer from the pretrained model.
        Otherwise, it sets the vocabulary size of the tokenizer to the vocabulary size of the model.
        It also initializes the loss function with the padding token as the ignore index.
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
