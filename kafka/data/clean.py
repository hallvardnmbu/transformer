"""Clean the text of Franz Kafka."""

import re

# --------------------------------------------------------------------------------------------------
# Question -> Answer pairs

with open("kafka_raw.txt", "r") as kafka:
    book = kafka.readlines()

# Remove header and footer:
book = book[831:21105]

# Remove lines with:
# no text
# "Page NUMBER"
# "Translated by X"
page = re.compile(r"Page \d+")
book = [line.strip().replace("”", "\"").replace("“", "\"").replace("_", "").replace("ii ", "")
        for line in book
        if not page.match(line)
        and not line.startswith("WoW")
        and not line.strip() == "Wo"
        and not line.strip() == "Wot"
        and not line.startswith("II")]

# Combine stories into single lines:
# The stories begin with a capitalized word

translator = re.compile(r"Translated by [A-Z][a-z]+")
uppercase = re.compile(r"(\b[A-Z]{2,}\b(?![:(]| \().*\b[A-Z]+\b)|(\b[A-Z]+\b.*\b[A-Z]{2,}\b)")

stories = []
story = []
for i, line in enumerate(book):
    if not line:
        continue

    if uppercase.match(line):
        stories.append(" ".join(story))
        story = [line]
        continue

    if translator.match(line):
        stories.append(" ".join(story))
        story = []
        continue

    story.append(line)

stories = [story for story in stories if len(story) > 150]

# Create "question -> answer" pairs:
sentences = [sentence + "." for story in stories for sentence in story.split(". ")]
pairs = [(sentences[i], sentences[i+1]) for i in range(len(sentences) - 1)]
longer = [(sentences[i], " ".join(sentences[i+1:i+4])) for i in range(1, len(sentences) - 4, 4)]
pairs.extend(longer)

# Save the pairs to a file:
with open("kafka_pairs.txt", "w") as f:
    for pair in pairs:
        f.write(f"{pair[0]}\t{pair[1]}\n")

# --------------------------------------------------------------------------------------------------
# Oneline

with open("kafka_raw.txt", "r") as kafka:
    book = kafka.readlines()

# Remove header and footer:
book = book[831:21105]

# Remove lines with:
# no text
# "Page NUMBER"
# "Translated by X"
page = re.compile(r"Page \d+")
book = [line.strip().replace("”", "\"").replace("“", "\"").replace("_", "").replace("ii ", "")
        for line in book
        if not page.match(line)
        and not line.startswith("WoW")
        and not line.strip() == "Wo"
        and not line.strip() == "Wot"
        and not line.startswith("II")]

# Combine stories into single lines:
# The stories begin with a capitalized word

translator = re.compile(r"Translated by [A-Z][a-z]+")
uppercase = re.compile(r"(\b[A-Z]{2,}\b(?![:(]| \().*\b[A-Z]+\b)|(\b[A-Z]+\b.*\b[A-Z]{2,}\b)")

stories = []
story = []
for i, line in enumerate(book):
    if not line:
        continue

    if uppercase.match(line):
        stories.append(" ".join(story))
        story = [line]
        continue

    if translator.match(line):
        stories.append(" ".join(story))
        story = []
        continue

    story.append(line)

stories = [story for story in stories if len(story) > 150]

line = " ".join(stories)

with open("kafka_oneline.txt", "w") as f:
    f.write(f"{line}")
