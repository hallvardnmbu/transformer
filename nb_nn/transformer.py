"""
Sequence-to-sequence model from pytorch.

https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

import math
import torch


class PositionalEncoding(torch.nn.Module):
    """Positional Encoding class for the Transformer model."""
    def __init__(self, n_embd: int, dropout: float, maxlen: int = 5000):
        """
        Positional Encoding class for the Transformer model.

        Parameters
        ----------
        n_embd : int
            The dimension of the embeddings.
        dropout : float
            The dropout value.
        maxlen : int, optional
            The maximum length of the sequence.
        """
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
        """
        Forward pass of the PositionalEncoding class.

        Parameters
        ----------
        token_embedding : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Embedding(torch.nn.Module):
    """Embedding class for the Transformer model."""
    def __init__(self, vocab_size: int, n_embd):
        """
        Embedding class for the Transformer model.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.
        n_embd : int
            The dimension of the embeddings.
        """
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.n_embd = n_embd

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass of the Embedding class.

        Parameters
        ----------
        tokens : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.embedding(tokens.long()) * math.sqrt(self.n_embd)


class Transformer(torch.nn.Module):
    """Transformer model for sequence-to-sequence translation."""
    def __init__(self, config):
        """
        Transformer class for the sequence-to-sequence model.

        Parameters
        ----------
        config
            The configuration object containing model hyperparameters.
        """
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

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        """
        Forward pass of the Transformer model.

        Parameters
        ----------
        src : torch.Tensor
            The source sequence.
        trg : torch.Tensor
            The target sequence.
        src_mask : torch.Tensor
            The source sequence mask.
        tgt_mask : torch.Tensor
            The target sequence mask.
        src_padding_mask : torch.Tensor
            The source sequence padding mask.
        tgt_padding_mask : torch.Tensor
            The target sequence padding mask.
        memory_key_padding_mask : torch.Tensor
            The memory key padding mask.

        Returns
        -------
        torch.Tensor
            The output tensor after the forward pass.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        Encode the source sequence.

        Parameters
        ----------
        src : torch.Tensor
            The source sequence.
        src_mask : torch.Tensor
            The source sequence mask.

        Returns
        -------
        torch.Tensor
            The encoded source sequence.
        """
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decode the target sequence.

        Parameters
        ----------
        tgt : torch.Tensor
            The target sequence.
        memory : torch.Tensor
            The memory tensor.
        tgt_mask : torch.Tensor
            The target sequence mask.

        Returns
        -------
        torch.Tensor
            The decoded target sequence.
        """
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )
