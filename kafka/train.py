"""Train the Kafka generative model."""

import os
import logging
import torch

from config import Hyperparameters
from kafka.kafka import Kafka


os.makedirs("./output", exist_ok=True)
handler = logging.FileHandler('./output/info.txt')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)

sentence = "Where the flower grows."


if __name__ == "__main__":
    config = Hyperparameters()
    kafka = Kafka(config)

    LOGGER.info("Hyperparameters: \n%s", config)
    LOGGER.info("\nTransformer architecture: \n%s", kafka.transformer.eval())
    LOGGER.info("\nVocabulary size: %s", kafka.tokenizer.vocab_size)

    LOGGER.info("\n> Continuation of '%s' before training:\n  %s",
                sentence, kafka(sentence))

    kafka.learn(checkpoints=True, sentence=sentence)

    LOGGER.info("\n> Continuation of '%s' after training:\n  %s",
                sentence, kafka(sentence))

    torch.save(kafka, "output/final_model.pth")
