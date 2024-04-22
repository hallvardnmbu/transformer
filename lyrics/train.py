"""Train the Lyrics generative model."""

import os
import logging
import torch

from config import Hyperparameters
from lyrics.lyrics import Lyrics


os.makedirs("./output", exist_ok=True)
handler = logging.FileHandler('./output/info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

sentence = "But I'm all right!"


if __name__ == "__main__":
    config = Hyperparameters()
    lyrics = Lyrics(config)

    LOGGER.info("Hyperparameters: \n%s", config)
    LOGGER.info("\nTransformer architecture: \n%s", lyrics.transformer.eval())
    LOGGER.info("\nVocabulary size: %s", lyrics.tokenizer.vocab_size)

    LOGGER.info("\n> Continuation of '%s' before training:\n  %s",
                sentence, lyrics(sentence))

    lyrics.learn(checkpoints=True, sentence=sentence)

    LOGGER.info("\n> Continuation of '%s' after training:\n  %s",
                sentence, lyrics(sentence))

    torch.save(lyrics, "output/final_model.pth")
