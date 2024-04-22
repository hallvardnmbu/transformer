"""Stack the various lyrics sources."""

import pandas as pd

dylan = pd.read_csv('./dylan.csv', header=0, usecols=[2, 3], names=["title", "lyrics"])
bsboys = pd.read_excel('./bsboys.xlsx', header=0, usecols=[5, 7], names=["title", "lyrics"])

# Chunk-wise reading (and filtering) of the genius dataset:
# --------------------------------------------------------------------------------------------------

reader = pd.read_csv('./genius.csv', header=0, usecols=[0, 6, 10], chunksize=10**5)

genius = []
for chunk in reader:
    filtered = chunk[chunk["language"] == "en"]
    genius.append(filtered[["title", "lyrics"]])
del reader

genius = pd.concat(genius)

# Combining:
# --------------------------------------------------------------------------------------------------

complete = pd.concat([dylan, bsboys, genius], axis=0)

complete.to_csv('./lyrics.csv', index=False)
