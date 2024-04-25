"""Cleaned up from: https://github.com/ADGEfficiency/creative-writing-with-gpt2/tree/main"""

from datasets import Dataset
from transformers import (DataCollatorForLanguageModeling,
                          GPT2LMHeadModel, GPT2Tokenizer,
                          Trainer, TrainingArguments)


def get_model(model="gpt2"):
    """
    Get the GPT-2 tokenizer and model.

    Parameters
    ----------
    model : str, optional
        Either 'gpt2' or path to local checkpoint.

    Returns
    -------
    dict
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return {
        "tokenizer": tokenizer,
        "model": GPT2LMHeadModel.from_pretrained(model),
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


def finetune(model, batch=8, epochs=5, data="./data/bible_oneline.txt", output="./output/"):
    """
    Finetune a model
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
    with open(data, "r") as bible:
        data = bible.read()
    data = Dataset.from_dict({"text": [data[i:i + 1024] for i in range(0, len(data), 1024)]})
    data = data.map(lambda _data: model["tokenizer"](_data["text"], truncation=True), batched=True)

    # https://huggingface.co/docs/transformers/
    # v4.40.1/en/main_classes/trainer#transformers.TrainingArguments
    args = TrainingArguments(
        output_dir=output,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch,
        save_only_model=True,
        save_strategy="epoch",
        num_train_epochs=epochs,
        disable_tqdm=True,
    )

    # https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
    collator = DataCollatorForLanguageModeling(
        tokenizer=model["tokenizer"],
        mlm=False,
        pad_to_multiple_of=1024,
    )

    trainer = Trainer(model=model["model"], args=args, train_dataset=data, data_collator=collator)
    trainer.train()


if __name__ == "__main__":
    gpt2 = get_model("gpt2")
    finetune(gpt2, batch=8, epochs=5, data="./bible/data/bible_oneline.txt", output="./output/")
