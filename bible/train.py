"""Train the Bible generative model."""

import os
import logging
import torch

from config import Hyperparameters
from bible.bible import Bible


os.makedirs("./output", exist_ok=True)
handler = logging.FileHandler('./output/info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

sentence = "Where the flower grows."


if __name__ == "__main__":
    config = Hyperparameters()
    bible = Bible(config)

    LOGGER.info("Hyperparameters: \n%s", config)
    LOGGER.info("\nTransformer architecture: \n%s", bible.transformer.eval())
    LOGGER.info("\nVocabulary size: %s", bible.tokenizer.vocab_size)

    LOGGER.info("\n> Continuation of '%s' before training:\n  %s",
                sentence, bible(sentence))

    bible.learn(checkpoints=True, sentence=sentence)

    LOGGER.info("\n> Continuation of '%s' after training:\n  %s",
                sentence, bible(sentence))

    torch.save(bible, "output/final_model.pth")
