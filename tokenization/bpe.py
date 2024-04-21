"""
Byte Pair Encoding (BPE) tokenization.

From Andrej Karpathy's minbpe (and OpenAI's GPT-2):

https://github.com/karpathy/minbpe
(https://github.com/openai/gpt-2/blob/master/src/encoder.py)

Using the regex pattern of GPT-4:

https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
"""

import unicodedata
import regex


PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"  # pylint: disable=C0301 # noqa: E501


# --------------------------------------------------------------------------------------------------
# Helper functions

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs. Optionally
    allows to update an existing dictionary of counts.

    Parameters
    ----------
    ids : list of int
        List of integers
    counts : dict, optional
        Dictionary of counts of consecutive pairs

    Example
    -------
        [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences of pair with the new
    integer token idx

    Example
    -------
    ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # If not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def replace_control_characters(string):
    """
    Replace control characters in a string with their unicode escape sequences

    Parameters
    ----------
    string : str

    Returns
    -------
    str

    Notes
    -----
    We don't want to print control characters which distort the output (e.g. \n or much worse)
        https://stackoverflow.com/questions/4324790/
            removing-control-characters-from-a-string-in-python/19016117#19016117
        http://www.unicode.org/reports/tr44/#GC_Values_Table
    """
    chars = [ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}" for ch in string]
    return "".join(chars)


def render_token(t: bytes) -> str:
    """Pretty print a token, escaping control characters."""
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


# --------------------------------------------------------------------------------------------------
# Tokenizers

class RegexTokenizer:
    """Regular expression-based tokenizer."""
    def __init__(self, pattern=None):
        """
        Default: vocab size of 256 (all bytes), no merges, no patterns.

        Parameters
        ----------
        pattern : str, optional
            Override the default (GPT-4) split pattern.

        Notes
        -----
        The merges is a dictionary of pairs of integers to integers, e.g. {(1, 2): 256}.
        The special tokens is a dictionary of strings to integers, e.g. {'<|endoftext|>': 100257}.
        The vocab is a dictionary of integers to bytes, e.g. {0: b'\x00', 1: b'\x01', ...}.
        """
        self.pattern = PATTERN if pattern is None else pattern
        self._pattern = regex.compile(self.pattern)

        self.merges = {}

        self.special_tokens = {}
        self.special_mapping = {}

        self.vocab = self._build_vocab()

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model.

        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        model_file = file_prefix + ".model"
        # Writes the pattern, special tokens, and merges to the model file.
        with open(model_file, 'w') as f:
            f.write(f"{self.pattern}\n")

            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, path):
        """Reads the pattern, special tokens, and merges from the model file."""
        merges = {}
        special_tokens = {}
        idx = 256
        with open(path, 'r', encoding="utf-8") as f:
            self.pattern = f.readline().strip()
            self._pattern = regex.compile(self.pattern)

            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
                idx += 1

            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.special_mapping = {v: k for k, v in special_tokens.items()}

        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """Vocab is simply and deterministically derived from merges."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on a given text.

        Parameters
        ----------
        text : str or list[str]
            Text to train on.
        vocab_size : int
        verbose : bool, optional
        """
        vocab = self._build_vocab()
        prior_size = len(vocab)

        assert vocab_size >= prior_size
        num_merges = vocab_size - prior_size

        if isinstance(text, list):
            text = " ".join(text)

        ids = [list(chunk.encode("utf-8")) for chunk in regex.findall(self._pattern, text)]

        # Iteratively merge the most common pairs to create new tokens
        merges = {}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            # Create a new token for the most common pair, and replace all occurrences in ids
            # with the new idx.
            idx = prior_size + i
            try:
                pair = max(stats, key=stats.get)
            except ValueError:
                print("!!! No more merges available !!!")
                print("Final vocab size:", idx - 1)
                break
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            # Update the merges and vocab dictionaries.
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                  f"({vocab[idx]}) had {stats[pair]} occurrences") if verbose else None

        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    @property
    def vocab_size(self):
        """Return the size of the vocabulary."""
        return len(self.vocab)

    def add_special_tokens(self, special_tokens):
        """
        Register special tokens to be used in encode() and decode().

        Parameters
        ----------
        special_tokens : dict
            A dictionary of special tokens (str) to integer ids (int).
            E.g., {'<|endoftext|>': 100257}.
        """
        self.special_tokens = special_tokens
        self.special_mapping = {v: k for k, v in special_tokens.items()}

    def decode(self, ids, skip_special_tokens=True):
        """Given a list of integers, return the corresponding string."""
        text_bytes = b"".join([self.vocab[idx]
                               for idx in ids
                               if idx in self.vocab
                               and ((idx not in self.special_mapping)
                                    if skip_special_tokens else True)])
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        """Encode a chunk of text into a list of token ids."""
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # If there are no more merges available, the key will result in an inf for every
            # single pair, and the min will be just the first pair in the list, arbitrarily we
            # can detect this terminating case by a membership check
            if pair not in self.merges:
                break
            # otherwise let's merge the best pair (lowest merge index).
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        ids = []
        for chunk in regex.findall(self._pattern, text):
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="all"):
        """
        Unlike encode_ordinary, this function handles special tokens.

        Parameters
        ----------
        text : str
            Text to encode.
        allowed_special : str or set, optional
            Keyword relating which special tokens are allowed (or set of special tokens).
            Can be "all"|"none"|"none_raise" or a custom set of special tokens

        Notes
        -----
        If allowed_special is "none_raise", then an error is raised if any special token is
        encountered in text. This is the default tiktoken behavior.
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # Shortcut: if no special tokens, just use the ordinary encoding.
            return self.encode_ordinary(text)

        # We have to be careful with potential special tokens in the text, which is done by
        # splitting the text based on the occurrence of any exact match with any of the special
        # tokens by using regex.
        special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
        special_chunks = regex.split(special_pattern, text)

        ids = [self.special_tokens["[CLS]"]]
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        ids.append(self.special_tokens["[SEP]"])

        return ids
