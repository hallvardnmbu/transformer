"""
Helper functions, based on the pytorch implementations.

https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

import torch

from .config import Hyperparameters


CONFIG = Hyperparameters()


def masking(src, tgt, device=CONFIG.device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)) == 1).transpose(0, 1)
    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == CONFIG.tokenizer["PAD_IDX"]).transpose(0, 1)
    tgt_padding_mask = (tgt == CONFIG.tokenizer["PAD_IDX"]).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([CONFIG.tokenizer["BOS_IDX"]]),
                      torch.tensor(token_ids),
                      torch.tensor([CONFIG.tokenizer["EOS_IDX"]])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [CONFIG.from_lang, CONFIG.to_lang]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[CONFIG.from_lang](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[CONFIG.to_lang](tgt_sample.rstrip("\n")))

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=CONFIG.tokenizer["PAD_IDX"])
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=CONFIG.tokenizer["PAD_IDX"])
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(CONFIG.from_lang, CONFIG.to_lang))
    train_dataloader = DataLoader(train_iter, batch_size=CONFIG.batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(CONFIG.device)
        tgt = tgt.to(CONFIG.device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = masking(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = CONFIG.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(CONFIG.from_lang, CONFIG.to_lang))
    val_dataloader = DataLoader(val_iter, batch_size=CONFIG.batch_size, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(CONFIG.device)
        tgt = tgt.to(CONFIG.device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = CONFIG.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))
