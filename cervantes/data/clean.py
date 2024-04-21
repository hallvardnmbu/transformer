"""Clean the text of Don Quixote."""

# --------------------------------------------------------------------------------------------------
# Question -> Answer pairs

with open("quixote_raw.txt", "r") as quixote:
    book = quixote.readlines()

# Remove header and footer:
book = book[2672:42925]

# Remove lines with:
# no text
# "*.jpg"
# "Full Size"
# only all caps
book = [line.strip()
        for line in book
        if ".jpg" not in line
        and "Full Size" not in line
        and not line.isupper()]

# Combine paragraphs into single lines:
# The paragraphs are separated by an empty line
# Also, replacing “ and ” with ", and "_" with ""
paragraphs = []
paragraph = []
for line in book:
    if line != '':
        paragraph.append(line.replace("“", "\"").replace("”", "\"").replace("_", ""))
    else:
        if paragraph:  # if the paragraph list is not empty
            paragraphs.append(" ".join(paragraph))
            paragraph = []

# Create "question -> answer" pairs:
# A question is either the last sentence of a paragraph
# or the last paragraph if it is less than 200 characters
pairs = []
for i, part in enumerate(paragraphs):
    if i == 0:
        sentences = part.split(". ")
        pairs.append((sentences[0] + ".", ". ".join(sentences[1:])))
    elif i == len(paragraphs) - 1:
        break

    if len(part) < 200:
        pairs.append((part, paragraphs[i + 1]))
    else:
        sentences = part.split(". ")
        pairs.append((sentences[-1], paragraphs[i + 1]))

# Split large pairs into smaller ones:
for pair in pairs:
    if len(pair[1]) < 4000:
        continue

    pairs.remove(pair)

    sentences = ". ".join(pair)
    sentences = sentences.split(". ")

    middle = len(sentences) // 2

    pairs.append((sentences[0], ". ".join(sentences[1:middle])))
    pairs.append((sentences[middle], ". ".join(sentences[middle:])))

# Save the pairs to a file:
with open("quixote_pairs.txt", "w") as f:
    for pair in pairs:
        f.write(f"{pair[0]}\t{pair[1]}\n")

# --------------------------------------------------------------------------------------------------
# One line version of the book

with open("quixote_raw.txt", "r") as quixote:
    book = quixote.readlines()

# Remove header and footer:
book = book[2672:42925]

# Remove lines with:
# no text
# "*.jpg"
# "Full Size"
# only all caps
book = [line.strip()
        for line in book
        if ".jpg" not in line
        and "Full Size" not in line
        and not line.isupper()
        and line.strip()]

line = " ".join(book)

# Save the line to a file:
with open("quixote_oneline.txt", "w") as f:
    f.write(f"{line}")
