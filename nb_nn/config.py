"""Sample translation config."""

from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer


@dataclass
class Hyperparameters:
    """
    Hyperparameters for the translation model.

    Attributes
    ----------
    vocab_size : int, optional
        The size of the vocabulary. Default is 50000.
    n_encoder_layer : int, optional
        The number of encoder layers. Default is 8.
    n_decoder_layer : int, optional
        The number of decoder layers. Default is 12.
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
        The batch size for training. Default is 128.
    device : str, optional
        The device to use for training. Default is "cuda" if available, else "cpu".
    from_lang : str, optional
        The source language. Default is "nb".
    from_path : str, optional
        The path to the source language model. Default is "NbAiLab/norwegian-paws-x".
    to_lang : str, optional
        The target language. Default is "nn".
    to_path : str, optional
        The path to the target language model. Default is "NbAiLab/norwegian-paws-x".
    tokenizer : dict[str, int or dict[str, int]], optional
        The tokenizer configuration. Default is a dictionary with certain keys.
    loss_fn : torch.nn.CrossEntropyLoss, optional
        The loss function to use. Default is None.
    optimizer : dict[str, int], optional
        The optimizer configuration. Default is a dictionary with certain keys.
    output_path : str, optional
        The path to save the output. Default is "./output/".
    """
    vocab_size: int = 50000
    n_encoder_layer: int = 8
    n_decoder_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

    epochs: int = 50
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    from_lang: str = "nb"
    from_path: str = "NbAiLab/norwegian-paws-x"
    to_lang: str = "nn"
    to_path: str = "NbAiLab/norwegian-paws-x"

    tokenizer: dict[str, int or dict[str, int]] = field(
        default_factory=lambda: {
            "path": "bert-base-multilingual-cased",  # Set to None to train a new tokenizer.
            # "path": "ltg/norbert3-large",  # Norwegian BERT. Set vocab_size = 50000 above

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
