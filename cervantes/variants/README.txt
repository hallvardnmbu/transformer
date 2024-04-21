The two models in the subdirectories `small` and `large` are trained only to predict the next word
in the sequences. Therefore, they are not trained to predict "stop" tokens, like the model in the
source `quixote` directory is. This means that a `generate` parameter is required, to specify how
many tokens to generate.