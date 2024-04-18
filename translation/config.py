"""Sample translation config."""

from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer


@dataclass
class Hyperparameters:
    vocab_size: int = 65536
    n_encoder_layer: int = 8
    n_decoder_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

    epochs: int = 50
    batch_size: int = 96
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    data_lang: str = "en-no"
    data_path: str = "Helsinki-NLP/opus-100"

    tokenizer: dict[str, int or dict[str, int]] = field(
        default_factory=lambda: {
            "path": "ltg/nort5-base-en-no-translation",  # Set to None to train a new tokenizer.

            # ONLY USED WHEN TRAINING A NEW TOKENIZER:
            # special_symbols: {TOKEN: ID, ...}. should include tokens [PAD], [CLS], [SEP]
            "k": 500,
            "special_symbols": {},

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

    output_path: str = "./output/"

    def __post_init__(self):
        if self.tokenizer["path"]:
            self.tokenizer["tokenizer"] = AutoTokenizer.from_pretrained(self.tokenizer["path"])
            self.tokenizer["vocab_size"] = self.vocab_size = self.tokenizer["tokenizer"].vocab_size
            self.tokenizer["special_symbols"] = self.tokenizer["tokenizer"].added_tokens_encoder
        else:
            self.tokenizer["vocab_size"] = self.vocab_size

        self.loss_fn = torch.torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer["special_symbols"]["[PAD]"]
        )
