"""Translator model based on the Transformer architecture."""

import time
import random
import logging
import torch

from transformers import AutoTokenizer
from tokenization.bpe import RegexTokenizer

from config import Hyperparameters
from seq2seq import Transformer


LOGGER = logging.getLogger(__name__)


class Translator:
    def __init__(self, config=Hyperparameters()):
        self.config = config

        self.source, self.target = None, None
        self._data(config.data_path, config.from_lang, config.to_lang)

        self.tokenizer = self._tokenizer(**config.tokenizer)
        assert self.tokenizer.vocab_size == config.vocab_size
        self.add_padding = self._padding()

        self.loss_fn = config.loss_fn
        self.transformer = Transformer(config)
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), **config.optimizer)

    def _tokenizer(self, path="ltg/norbert3-large", vocab_size=None, k=None, **other):
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
            tokenizer.add_special_tokens({**other["special_symbols"]})
            tokenizer.train(
                text=random.sample(self.source, k=k) + random.sample(self.target, k=k),
                vocab_size=vocab_size
            )
            return tokenizer
        return AutoTokenizer.from_pretrained(path)

    def _data(self, path="./dataset/MultiParaCrawl", from_lang="nb", to_lang="nn"):
        """
        Load the data for the model.

        Parameters
        ----------
        path : str
            Path to the dataset files, omitting the suffix (i.e. language abbreviation).
        from_lang : str, optional
            The source language.
        to_lang : str, optional
            The target language.

        Returns
        -------
        list, list
            The data for the two languages. E.g., lists of sentences.

        Note
        ----
        The suffixes are defined in the `Hyperparameters` class, as `from_lang` and `to_lang`.
        """
        LOGGER.info("Loading data from %s for %s -> %s.", path, from_lang, to_lang)

        self.source = [_l1.strip() for _l1 in open(path + "." + from_lang, "r").readlines()]
        self.target = [_l2.strip() for _l2 in open(path + "." + to_lang, "r").readlines()]

        LOGGER.info(" Success.")

    def __call__(self, text, margin=5):
        self.transformer.eval()
        device = self.config.device

        src = self.add_padding(text).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = self.transformer.encode(src, src_mask)
        out = torch.ones(1, 1).fill_(
            self.config.tokenizer["special_symbols"]["[BOS]"]
        ).type(torch.long).to(device)

        for i in range(num_tokens + margin):
            memory = memory.to(device)
            tgt_mask = (self.square_mask(out.size(0)).type(torch.bool)).to(device)

            _out = self.transformer.decode(out, memory, tgt_mask)
            _out = _out.transpose(0, 1)
            prob = self.transformer.generator(_out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            out = torch.cat([out, torch.ones(1, 1).type_as(src.data).fill_(next_word)],
                            dim=0)
            if next_word == self.config.tokenizer["special_symbols"]["[EOS]"]:
                break

        return self.tokenizer.decode(out.flatten()).replace("[BOS]", "").replace("[EOS]", "")

    def _padding(self):
        """
        Functionality for adding [BOS] and [EOS] (beginning & end of sentence) to an inputted text.
        Note that any transforms may be added to the pipeline (in the dictionary below).
        Also note that the (e.g.,) encoding can vary between the languages, if so, provide a mapping
        from each language to the wanted encoder (and retrive this when calling `_transform`).
        """
        def _transform(*transforms):
            def func(text):
                for transform in transforms:
                    text = transform(text)
                return text

            return func

        def _padding(tokens):
            return torch.cat((torch.tensor([self.config.tokenizer["special_symbols"]["[BOS]"]]),
                              torch.tensor(tokens),
                              torch.tensor([self.config.tokenizer["special_symbols"]["[BOS]"]])))

        # For multi-tokenizer support:
        # (Edit parts where `self.add_padding` is used to include the language mapping.)
        # return {language: _transform(self.tokenizer.encode, _padding)
        #         for language in [config.from_lang, config.to_lang]}

        return _transform(self.tokenizer.encode, _padding)

    def _split_epoch(self, batch):
        src = [random.sample(self.source, k=batch) for _ in range(len(self.source) // batch)]
        tgt = [random.sample(self.target, k=batch) for _ in range(len(self.target) // batch)]
        return src, tgt

    def train_epoch(self, batch_size=128):
        LOGGER.debug("Training the model one epoch.")

        if not self.source or not self.target:
            raise ValueError("No data loaded for the model.")

        self.transformer.train()
        losses = 0

        dataset = self._split_epoch(batch=batch_size)
        for src, tgt in zip(dataset):
            src = src.to(self.config.device)
            tgt = tgt.to(self.config.device)

            tgt_input = tgt[:-1, :]

            src_m, tgt_m, src_pad_m, tgt_pad_m = self.masking(src, tgt_input)

            logits = self.transformer(src, tgt_input, src_m, tgt_m, src_pad_m, tgt_pad_m, src_pad_m)

            self.optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            losses += loss.item() / len(" ".join([_src for _src in src]))

        return losses

    def evaluate(self, batch_size=128):
        LOGGER.debug("Evaluating the model.")

        if not self.source or not self.target:
            raise ValueError("No data loaded for the model.")

        self.transformer.eval()
        losses = 0

        dataset = self._split_epoch(batch=batch_size)
        for src, tgt in zip(dataset):
            src = src.to(self.config.device)
            tgt = tgt.to(self.config.device)

            tgt_input = tgt[:-1, :]

            src_m, tgt_m, src_pad_m, tgt_pad_m = self.masking(src, tgt_input)

            logits = self.transformer(src, tgt_input, src_m, tgt_m, src_pad_m, tgt_pad_m, src_pad_m)

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item() / len(" ".join([_src for _src in src]))

        return losses

    def train(self, batch_size=128, epochs=5):
        LOGGER.info("Training the model for %s epochs.", epochs)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(batch_size)
            end_time = time.time()
            val_loss = self.evaluate(batch_size)

            LOGGER.info("Epoch: %s, Train loss: %s, Val loss: %s, Epoch time = %ss",
                        epoch, train_loss, val_loss, round(end_time - start_time, 3))

    def square_mask(self, dim):
        mask = (torch.triu(torch.ones((dim, dim), device=self.config.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def masking(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_m = self.square_mask(tgt_seq_len)
        src_m = torch.zeros((src_seq_len, src_seq_len), device=self.config.device).type(torch.bool)

        src_pad_m = (src == self.config.tokenizer["special_symbols"]["[PAD]"]).transpose(0, 1)
        tgt_pad_m = (tgt == self.config.tokenizer["special_symbols"]["[PAD]"]).transpose(0, 1)

        return src_m, tgt_m, src_pad_m, tgt_pad_m
