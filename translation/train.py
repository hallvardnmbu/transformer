import os
import logging
import torch

from config import Hyperparameters
from translator import Translator


os.makedirs("./output", exist_ok=True)
handler = logging.FileHandler('./output/info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

sentence = "Jeg liker å sitte på skolen når jeg jobber."


if __name__ == "__main__":
    config = Hyperparameters()
    translator = Translator(config)

    LOGGER.info("Hyperparameters: \n%s", config)
    LOGGER.info("\nTransformer architecture: \n%s", translator.transformer.eval())
    LOGGER.info("\nVocabulary size: %s", translator.tokenizer.vocab_size)

    LOGGER.info("\n> Translation of '%s' before training:\n  %s",
                sentence, translator(sentence))

    translator.learn(checkpoints=True, sentence=sentence)

    LOGGER.info("\n> Translation of '%s' after training:\n  %s",
                sentence, translator(sentence))

    torch.save(translator, "output/final_model.pth")
