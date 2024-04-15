"""Sample translation config."""

from dataclasses import dataclass, field
import torch


@dataclass
class Hyperparameters:
    vocab_size: int = 50000
    n_encoder_layer: int = 3
    n_decoder_layer: int = 3
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

    epochs: int = 10
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    from_lang: str = "nb"
    to_lang: str = "nn"
    data_path: str = "./dataset/MultiParaCrawl"

    tokenizer: dict[str, int or dict[str, int]] = field(
        default_factory=lambda: {
            "path": "ltg/norbert3-large", "vocab_size": None, "k": 500,
            "special_symbols": {
                '[BOS]': 5,
                '[EOS]': 6,
                '[UNK]': 0,
                '[SEP]': 2,
                '[PAD]': 3,
                '[CLS]': 1,
                '[MASK]': 4
            }
        }
    )
    loss_fn: torch.nn.CrossEntropyLoss = None

    optimizer: dict[str, int] = field(
        default_factory=lambda: {
            "lr": 0.0001, "betas": (0.9, 0.98), "eps": 1e-9
        }
    )

    output_path: str = "./output/"

    def __post_init__(self):
        self.tokenizer["vocab_size"] = self.vocab_size
        self.loss_fn = torch.torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer["special_symbols"]["[PAD]"]
        )
