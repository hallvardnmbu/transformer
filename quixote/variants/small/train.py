"""Train a small variant of the Quixote model."""

import os
import logging
import torch

from config import Hyperparameters
from quixote.variants.small.quixote import Quixote


os.makedirs("./output", exist_ok=True)
handler = logging.FileHandler('./output/info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

sentence = "“I hold thou art in the right of it, Sancho,” quoth Don Quixote,"


if __name__ == "__main__":
    config = Hyperparameters()
    quixote = Quixote(config)

    LOGGER.info("Hyperparameters: \n%s", config)
    LOGGER.info("\nTransformer architecture: \n%s", quixote.transformer.eval())
    LOGGER.info("\nVocabulary size: %s", quixote.tokenizer.vocab_size)

    LOGGER.info("\n> Continuation of '%s' before training:\n  %s",
                sentence, quixote(sentence))

    quixote.learn(checkpoints=True, sentence=sentence)

    LOGGER.info("\n> Continuation of '%s' after training:\n  %s",
                sentence, quixote(sentence))

    torch.save(quixote, "output/final_model.pth")
