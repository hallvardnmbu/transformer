"""Clean the text of the Bible."""

import re

with open("bible_raw.txt", "r") as file:
    bible = file.readlines()

bible = bible[5:]

# --------------------------------------------------------------------------------------------------
# Regular expression to find the paragraphs of the Bible.

paragraphs = {}
pattern = re.compile(r"(.+ \d+):\d+ (.+)")
for line in bible:
    try:
        which, text = pattern.findall(line)[0]
        paragraphs[which] = paragraphs.get(which, "") + text + " "
    except IndexError:
        pass

with open("bible_paragraphs.txt", "w") as f:
    f.write("\n".join(paragraphs.values()))

# --------------------------------------------------------------------------------------------------
# Oneline version of the Bible.

with open("bible_oneline.txt", "w") as f:
    f.write(" ".join(paragraphs.values()))
