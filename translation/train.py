import os
import logging
import torch

from config import Hyperparameters
from translator import Translator


os.makedirs("./output", exist_ok=True)
handler = logging.FileHandler('./output/info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)

sentence = "Hvilken dag i uka er det i dag?"


if __name__ == "__main__":
    config = Hyperparameters()
    translator = Translator(config)

    LOGGER.info("Hyperparameters: \n%s", config)
    LOGGER.info("Transformer architecture: \n%s", translator.transformer.eval())
    LOGGER.info("Vocabulary size: %s", translator.tokenizer.vocab_size)

    LOGGER.info("Translation of '%s' before training: %s",
                sentence, translator(sentence))

    translator.learn(checkpoints=True, sentence=sentence)

    LOGGER.info("Translation of '%s' after training: %s",
                sentence, translator(sentence))

    torch.save(translator, "final_model.pth")
