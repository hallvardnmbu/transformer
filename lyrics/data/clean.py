"""Stack and preprocess the various lyrics sources."""

import pandas as pd

dylan = pd.read_csv('./dylan.csv', header=0, usecols=[2, 3], names=["title", "lyrics"])
dylan["lyrics"] = dylan["lyrics"].str.replace('\n\n', '\n').str.replace(r'\n\n\n', '\n\n').str.replace('\n \n ', '\n').str.replace('\n \n \n \n ', '\n\n')
dylan["title"] = dylan["title"] + " by Bob Dylan"

beatles = pd.read_csv('./beatles.csv', header=0, usecols=[0, 3], names=["title", "lyrics"])
beatles["lyrics"] = beatles["lyrics"].str.replace('  ', '\n\n').str.replace('. ', '.\n').str.replace(r' ([A-Z][^\n]*?(?: {1,3}))', r'\n\1', regex=True)
beatles["title"] = beatles["title"] + " by The Beatles"

various = pd.read_csv('various.csv', header=0, usecols=[2, 3, 4, 8])
various = various[various["tag"] != "misc"]
various["lyrics"] = various["lyrics"].str.replace('\n\n\n', '\n\n').str.replace(r'^\n{1,}', '', regex=True)
various["title"] = various["title"] + " by " + various["artist"]
various = various.drop(columns=["artist", "tag"])

# Combining:
# --------------------------------------------------------------------------------------------------

complete = pd.concat([dylan, beatles, various], axis=0)
complete.to_csv('./lyrics.csv', index=False, sep='+')
