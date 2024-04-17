"""Clean the text of Don Quixote and create question-answer pairs."""

with open("./quixote.txt", "r") as quixote:
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
paragraphs = []
paragraph = []
for line in book:
    if line != '':
        paragraph.append(line)
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
        pairs.append((sentences[0], ". ".join(sentences[1:])))
    elif i == len(paragraphs) - 1:
        break

    if len(part) < 200:
        pairs.append((part, paragraphs[i + 1]))
    else:
        sentences = part.split(". ")
        pairs.append((sentences[-1], paragraphs[i + 1]))

# Save the pairs to a file:
with open("quixote_pairs.txt", "w") as f:
    for pair in pairs:
        f.write(f"{pair[0]}\t{pair[1]}\n")
