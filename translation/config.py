"""Sample translation config."""

from dataclasses import dataclass, field
import torch


@dataclass
class Hyperparameters:
    vocab_size: int = 5000
    n_encoder_layer: int = 3
    n_decoder_layer: int = 3
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    from_lang: str = "nb"
    to_lang: str = "nn"
    data_path: str = "./dataset/MultiParaCrawl"

    tokenizer: dict[str, int] = field(
        default_factory=lambda: {
            "path": "ltg/norbert3-large", "vocab_size": None, "k": 500,
            "UNK_IDX": 0, "PAD_IDX": 1, "BOS_IDX": 2, "EOS_IDX": 3,
            "special_symbols": ['<unk>', '<pad>', '<bos>', '<eos>']
        }
    )
    tokenizer["vocab_size"] = vocab_size

    loss_fn = torch.torch.nn.CrossEntropyLoss(ignore_index=tokenizer["PAD_IDX"])

    optimizer: dict[str, int] = field(
        default_factory=lambda: {
            "lr": 0.0001, "betas": (0.9, 0.98), "eps": 1e-9
        }
    )
