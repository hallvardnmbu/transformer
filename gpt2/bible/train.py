"""Train the GPT-2 generative model."""

import os
import logging
import torch

from config import Hyperparameters
from gpt2.model import Model


os.makedirs("./output", exist_ok=True)
handler = logging.FileHandler('./output/info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

sentence = "Where the flower grows."


if __name__ == "__main__":
    config = Hyperparameters()
    model = Model(config)

    LOGGER.info("Hyperparameters: \n%s", config)
    LOGGER.info("\nTransformer architecture: \n%s", model.transformer.eval())
    LOGGER.info("\nVocabulary size: %s", model.tokenizer.vocab_size)

    LOGGER.info("\n> Continuation of '%s' before training:\n  %s",
                sentence, model(sentence))

    model.learn(checkpoints=True, sentence=sentence)

    LOGGER.info("\n> Continuation of '%s' after training:\n  %s",
                sentence, model(sentence))

    torch.save(model, "output/final_model.pth")
