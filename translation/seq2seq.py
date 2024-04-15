"""
Sequence-to-sequence model from pytorch.

https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

import math
import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, n_embd: int, dropout: float, maxlen: int = 5000):
        super().__init__()

        den = torch.exp(- torch.arange(0, n_embd, 2)* math.log(10000) / n_embd)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding = torch.zeros((maxlen, n_embd))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, n_embd):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.n_embd = n_embd

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.n_embd)


class Transformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = torch.nn.Transformer(
            d_model=config.n_embd,
            nhead=config.n_head,
            num_encoder_layers=config.n_encoder_layer,
            num_decoder_layers=config.n_decoder_layer,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout
        )

        self.src_tok_emb = Embedding(config.vocab_size, config.n_embd)
        self.tgt_tok_emb = Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = PositionalEncoding(config.n_embd, dropout=config.dropout)

        self.generator = torch.nn.Linear(config.n_embd, config.vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.to(config.device)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )
