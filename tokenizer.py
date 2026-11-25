"""
GPT-4-style byte-level BPE tokenizer, as in nanochat: trained with rustbpe, encoding with tiktoken.
Each tokenizer lives in BASE_DIR/tokenizers/<name>/. Train one on a character mix of downloaded sources (prepare.py),
e.g. v2, the one in use (16k, ~10% code / shell, chat and tool special tokens):
    python tokenizer.py --name=v2 --mix=climbmix:0.9,shell:0.025,python:0.02,sysadmin_qa:0.045,tldr:0.005,nl2sh:0.005
"""

import argparse
import base64
import itertools
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import rustbpe
import tiktoken

from config import BASE_DIR, fetch
from data import parse_mix, raw_files

TOKENIZERS_DIR = os.path.join(BASE_DIR, "tokenizers")

# special tokens (rendering of the chat / tool format: chat.py)
SPECIAL_TOKENS = [
    "<|bos|>",  # starts every document
    "<|system_start|>", "<|system_end|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|tool_start|>", "<|tool_end|>",  # a tool call, with the tool's name (e.g. bash, python) on the first line
    "<|output_start|>", "<|output_end|>",  # the tool's output
]
SPECIAL_TOKENS += [f"<|reserved_{i}|>" for i in range(32 - len(SPECIAL_TOKENS))]

# GPT-4's split pattern, except numbers are split into chunks of at most 2 digits (nanochat found this best)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, enc: tiktoken.Encoding):
        self.enc = enc
        self.bos_id = enc.encode_single_token("<|bos|>")
        self.vocab_size = enc.n_vocab

    @classmethod
    def train(cls, texts, vocab_size):
        bpe = rustbpe.Tokenizer()
        bpe.train_from_iterator(texts, vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
        ranks = {bytes(k): v for k, v in bpe.get_mergeable_ranks()}
        specials = {name: len(ranks) + i for i, name in enumerate(SPECIAL_TOKENS)}
        return cls(tiktoken.Encoding("bpe", pat_str=bpe.get_pattern(), mergeable_ranks=ranks, special_tokens=specials))

    @classmethod
    def load(cls, name):
        """BASE_DIR/tokenizers/<name>/tokenizer.json (the released v2 is downloaded if it is missing)."""
        path = fetch(f"tokenizers/{name}/tokenizer.json") if name == "v2" else \
            os.path.join(TOKENIZERS_DIR, name, "tokenizer.json")
        return cls.from_file(path)

    @classmethod
    def from_file(cls, path):
        """tokenizer.json: the split pattern, the special tokens {name: id} and the mergeable ranks as a list of
        base64-encoded byte strings (the token of rank i at index i), as in tiktoken's own .tiktoken files."""
        with open(path) as f:
            d = json.load(f)
        ranks = {base64.b64decode(b): i for i, b in enumerate(d["ranks"])}
        return cls(tiktoken.Encoding("bpe", pat_str=d["pattern"], mergeable_ranks=ranks,
                                     special_tokens=d["special_tokens"]))

    def save(self, name):
        os.makedirs(os.path.join(TOKENIZERS_DIR, name), exist_ok=True)
        self.save_file(os.path.join(TOKENIZERS_DIR, name, "tokenizer.json"))

    def save_file(self, path):
        ranks = self.enc._mergeable_ranks
        assert sorted(ranks.values()) == list(range(len(ranks)))
        d = dict(pattern=self.enc._pat_str, special_tokens=self.enc._special_tokens,
                 ranks=[base64.b64encode(b).decode() for b, _ in sorted(ranks.items(), key=lambda kv: kv[1])])
        with open(path, "w") as f:
            json.dump(d, f)

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def encode_docs(self, texts, num_threads=8, allow_special=False):
        """Encodes a batch of documents, each prefixed with BOS. Special token text (e.g. '<|user_start|>') becomes the
        special token only with allow_special, for documents rendered in the chat format (chat.py)."""
        if allow_special:
            batch = self.enc.encode_batch(texts, num_threads=num_threads, allowed_special="all")
        else:
            batch = self.enc.encode_ordinary_batch(texts, num_threads=num_threads)
        return [[self.bos_id] + ids for ids in batch]

    def decode(self, ids):
        return self.enc.decode(ids)

    def token_bytes(self):
        """UTF-8 byte length of every token (0 for special tokens), used to compute bits per byte."""
        special = {self.enc.encode_single_token(s) for s in self.enc.special_tokens_set}
        return np.array([0 if i in special else len(self.enc.decode_single_token_bytes(i))
                         for i in range(self.vocab_size)], dtype=np.int32)


def texts(source, split, doc_cap=None):
    for path in raw_files(source, split):
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield text[:doc_cap]


def bytes_per_token(tok, source, max_docs=10_000):
    """Compression on (the start of) a source's held-out documents."""
    docs = list(itertools.islice(texts(source, "val"), max_docs))
    return sum(len(t.encode()) for t in docs) / sum(map(len, tok.enc.encode_ordinary_batch(docs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="saved to BASE_DIR/tokenizers/<name>")
    parser.add_argument("--mix", default="climbmix:1", help="fraction of training characters from each source")
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--max_chars", type=int, default=2_000_000_000)
    parser.add_argument("--doc_cap", type=int, default=10_000, help="crop documents to this many characters")
    args = parser.parse_args()
    mix = parse_mix(args.mix)

    n_chars = dict.fromkeys(mix, 0)

    def train_texts():  # each source's share of max_chars, or all of it if it has less
        for source, weight in mix.items():
            for text in texts(source, "train", args.doc_cap):
                yield text
                n_chars[source] += len(text)
                if n_chars[source] >= weight / sum(mix.values()) * args.max_chars:
                    break

    t0 = time.time()
    tok = Tokenizer.train(train_texts(), args.vocab_size)
    print(f"trained a {tok.vocab_size}-token tokenizer in {time.time() - t0:.0f}s on " +
          ", ".join(f"{source} {n / 1e6:.0f}M" for source, n in n_chars.items()) + " characters")
    sample = "Hello world! Numbers: 123, 4567. Contractions: I'm, you're. Unicode: 你好世界 🌍"
    assert tok.decode(tok.encode(sample)) == sample
    tok.save(args.name)

    compare(mix)


def compare(sources):
    """Prints the compression of every saved tokenizer on held-out documents of each source, in bytes per token."""
    names = sorted(os.listdir(TOKENIZERS_DIR))
    print("bytes/token on val | " + " | ".join(names))
    for source in sources:
        print(f"{source} | " + " | ".join(f"{bytes_per_token(Tokenizer.load(name), source):.3f}" for name in names))


if __name__ == "__main__":
    main()
