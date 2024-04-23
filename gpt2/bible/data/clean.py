"""Clean the text of the bible."""

import re

with open("bible_raw.txt", "r") as file:
    bible = file.readlines()

bible = bible[5:]

# --------------------------------------------------------------------------------------------------
# Regular expression to find the paragraphs of the bible. Q&A pairs and raw paragraphs.

paragraphs = {}
pattern = re.compile(r"(.+ \d+):\d+ (.+)")
for line in bible:
    try:
        which, text = pattern.findall(line)[0]
        paragraphs[which] = paragraphs.get(which, "") + text + " "
    except IndexError:
        pass
qa = {}
for text in paragraphs.values():
    sentences = text.split(". ")
    qa[sentences[0] + "."] = ". ".join(sentences[1:])

with open("bible_qa.txt", "w") as f:
    for question, answer in qa.items():
        f.write(f"{question}\t{answer}\n")

with open("bible_paragraphs.txt", "w") as f:
    for paragraph in paragraphs.values():
        f.write(f"{paragraph}\n")

# --------------------------------------------------------------------------------------------------
# Regular expression to find the books of the bible.

books = {}
pattern = re.compile(r"(.+) \d+:\d+ (.+)")
for line in bible:
    try:
        which, text = pattern.findall(line)[0]
        books[which] = books.get(which, "") + text + " "
    except IndexError:
        pass
assert len(books) == 66

with open("bible_books.txt", "w") as f:
    for which, text in books.items():
        f.write(f"{which}\t{text}\n")

# --------------------------------------------------------------------------------------------------
# Oneline for the bible

oneline = " ".join(books.values())
with open("bible_oneline.txt", "w") as f:
    f.write(f"{oneline}")
