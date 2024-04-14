"""Translator model based on the Transformer architecture."""

import random
import torch

from transformers import AutoTokenizer
from tokenization.bpe import RegexTokenizer

from .config import Hyperparameters
from .seq2seq import Transformer


class Translator(torch.nn.Module):
    def __init__(self, config=Hyperparameters()):
        super().__init__()

        self.from_lang = config.from_lang
        self.to_lang = config.to_lang

        self.source, self.target = self._data(config.data_path)

        self.tokenizer = self._tokenizer(**config.tokenizer)

        self.transformer = Transformer(config)

    def _tokenizer(self, path="ltg/norbert3-large", vocab_size=None, k=None):
        """
        Create a tokenizer for the model.

        Parameters
        ----------
        path : False or str, optional
            Whether to use a pretrained tokenizer.
            False to train a new tokenizer on the datasets.
            String to load a pretrained tokenizer from huggingface.
        vocab_size : int, optional
            Size of the vocabulary for the custom tokenizer.
        k : int, optional
            Number of samples to use for training the custom tokenizer.

        Returns
        -------
        tokenization.bpe.RegexTokenizer or transformers.AutoTokenizer
        """
        if not path:
            k = min(len(self.source), len(self.target), k)

            tokenizer = RegexTokenizer()
            tokenizer.train(
                text=random.choices(self.source, k=k) + random.choices(self.target, k=k),
                vocab_size=vocab_size
            )

            return tokenizer

        return AutoTokenizer.from_pretrained(path)

    def _data(self, path="./dataset/MultiParaCrawl"):
        """
        Load the data for the model.

        Parameters
        ----------
        path : str
            Path to the dataset files, omitting the suffix (i.e. language abbreviation).

        Returns
        -------
        list, list
            The data for the two languages. E.g., lists of sentences.

        Note
        ----
        The suffixes are defined in the `Hyperparameters` class, as `from_lang` and `to_lang`.
        """
        from_lang = [_l1.strip() for _l1 in open(path + "." + self.from_lang, "r").readlines()]
        to_lang = [_l2.strip() for _l2 in open(path + "." + self.to_lang, "r").readlines()]

        return from_lang, to_lang
