"""
Cleaned up from:

https://github.com/ADGEfficiency/creative-writing-with-gpt2/tree/main

With specifics about T5 from:

https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
https://huggingface.co/learn/nlp-course/chapter7/4?fw=pt
"""

import csv
import datasets
from transformers import (DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer,
                          T5ForConditionalGeneration, T5Tokenizer)


def get_model(model="t5-small"):
    """
    Get the T5 tokenizer and model.

    Parameters
    ----------
    model : str, optional
        Path to local checkpoint or;
        't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'.

    Returns
    -------
    dict
    """
    _tokenizer = "t5-small" if model.startswith(".") else model
    tokenizer = T5Tokenizer.from_pretrained(_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    return {
        "tokenizer": tokenizer,
        "model": T5ForConditionalGeneration.from_pretrained(model),
    }


def evaluate(model, text, generate=50, **kwargs):
    """
    Get an output from the model.

    Parameters
    ----------
    model : dict
        Containing the tokenizer and model.
    text : str
        The prompt to send through the model.
    generate : int
        Maximum number of tokens to generate.
    kwargs : dict
        Other keyword-arguments sent to `transformers.GPT2LMHeadModel.generate()`.

    Returns
    -------
    str
        The output of the model.
    """
    data = model["tokenizer"](text, return_tensors="pt")

    attention_mask = data["input_ids"].ne(model["tokenizer"].pad_token_id).float()

    out = model["model"].generate(
        data["input_ids"], max_length=generate,
        num_beams=5, no_repeat_ngram_size=2,
        attention_mask=attention_mask,
        pad_token_id=model["tokenizer"].eos_token_id,
        **kwargs
    )
    return model["tokenizer"].decode(
        out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def get_data(tokenizer, path, prefix="Create the lyrics of"):
    """
    Get the data for finetuning.

    Parameters
    ----------
    tokenizer : transformers.T5Tokenizer
    path : str
        Path to the data.
    prefix : str, optional
        Prefix to add to the source data.

    Returns
    -------
    dict
        The tokenized data.
    """
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter='+')
        data = {src: tgt for src, tgt in list(reader)[1:] if tgt}

    src = [f"{prefix} {_src}" for _src in data.keys()]
    tgt = list(data.values())

    data = tokenizer(
        src,
        max_length=None,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )

    tgt = tokenizer(
        tgt,
        max_length=None,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )
    data["labels"] = tgt["input_ids"]

    data = datasets.Dataset.from_dict(data)  # noqa
    return data


def finetune(model, batch=2, epochs=5, data="../data/lyrics/lyrics.csv", output="./output/"):
    """
    Finetune a model.

    Parameters
    ----------
    model : dict
        Containing the tokenizer and model.
    batch : int
        Batch size.
    epochs : int
        Number of epochs to train.
    data : str
        Path to the data to train on.
    output : str
        Path to save the checkpoints.
    """
    data = get_data(model["tokenizer"], data, prefix="Create the lyrics of")

    args = Seq2SeqTrainingArguments(
        output_dir=output,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch,
        save_only_model=True,
        save_strategy="epoch",
        num_train_epochs=epochs,
        predict_with_generate=True,
        metric_for_best_model="rouge1",
        disable_tqdm=True,
        learning_rate=1e-4,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=model["tokenizer"],
        pad_to_multiple_of=1024,
    )

    trainer = Seq2SeqTrainer(
        model=model["model"], tokenizer=model["tokenizer"],
        args=args, train_dataset=data, data_collator=collator,
    )
    trainer.train()


if __name__ == "__main__":
    t5 = get_model("t5-small")
    finetune(t5, batch=2, epochs=10, data="../../data/lyrics/lyrics.csv", output="./output/")
