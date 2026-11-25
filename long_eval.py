"""
Long-context evals at lengths 2048, 4096, ... up to max_len (beyond the training length means without finetuning):
- loss vs position: bits per byte per position bucket on long held-out documents (FinePDFs-Edu), each read from its
  start in one causal pass of max_len tokens (so the buckets below T are exactly what a pass of length T would give)
- passkey retrieval: a 5-digit key hidden at several depths in repeated filler text, followed by a question; correct
  only if greedy decoding reproduces the key exactly
- needles in real text: n "magic numbers" for different names at random places in a long document (n - 1 of them are
  distractors), then a question about one of them; scored like passkey
The documents are stored as raw text under BASE_DIR/evals and tokenized on the fly, so any tokenizer works.
Long-context evals of a saved run: eval.py --evals=long.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

import model
from config import BASE_DIR
from core_eval import run_batches

DOCS_PATH = os.path.join(BASE_DIR, "evals", "long_docs.jsonl")
DOCS_FILE = "datasets/HuggingFaceFW/finepdfs-edu/data/eng_Latn/train/000_00099.parquet"  # held out in data.SOURCES
TOKENS_PER_DEVICE = 2**16  # per forward pass

INTRO = ("There is an important piece of information hidden inside a lot of irrelevant text. Find it and memorize it. "
         "I will quiz you about it.")
FILLER = " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NEEDLE = " The pass key is {key}. Remember it. {key} is the pass key."
QUESTION = " What is the pass key? The pass key is"
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)  # fraction of the filler that comes before the key
MAGIC_NEEDLE = "\nThe special magic number for {name} is {key}.\n"
MAGIC_QUESTION = "\nWhat is the special magic number for {name}? The special magic number for {name} is"
NAMES = ["apple", "river", "falcon", "copper", "violet", "harbor", "maple", "comet", "tiger", "lantern", "meadow", "crystal"]
NEEDLE_COUNTS = (1, 4)


def download_docs(n_docs=200, min_chars=100_000, keep_chars=200_000):
    """Saves the first n_docs documents of at least min_chars characters (cropped to keep_chars) of the held-out file."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    print(f"downloading {n_docs} long documents to {DOCS_PATH}")
    pf, docs = pq.ParquetFile(HfFileSystem().open(DOCS_FILE)), []
    for i in range(pf.num_row_groups):
        docs += [t[:keep_chars] for t in pf.read_row_group(i, columns=["text"]).column("text").to_pylist()
                 if len(t) >= min_chars]
        if len(docs) >= n_docs:
            break
    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    with open(DOCS_PATH, "w") as f:
        f.writelines(json.dumps({"text": t}) + "\n" for t in docs[:n_docs])


def passkey_prompts(tokenizer, length, n_keys=20, seed=0):
    """Prompts of length + 1 tokens (BOS-padded at the end), with the [start, end) token span of the key to predict."""
    keys = np.random.default_rng(seed).integers(10_000, 100_000, n_keys)
    intro, filler, question = (tokenizer.encode(s) for s in (INTRO, FILLER, QUESTION))
    tokens, start, end = np.full((len(DEPTHS) * n_keys, length + 1), tokenizer.bos_id, np.int32), [], []
    for row, (depth, key) in zip(tokens, [(d, k) for d in DEPTHS for k in keys]):
        needle, answer = tokenizer.encode(NEEDLE.format(key=key)), tokenizer.encode(f" {key}")
        n_filler = (length - len(intro) - len(needle) - len(question) - len(answer)) // len(filler)
        n_before = round(depth * n_filler)
        prompt = [tokenizer.bos_id] + intro + filler * n_before + needle + filler * (n_filler - n_before) + question
        row[:len(prompt) + len(answer)] = prompt + answer
        start.append(len(prompt))
        end.append(len(prompt) + len(answer))
    return tokens, np.array(start), np.array(end)


def needle_prompts(tokenizer, docs, length, n_needles, n_prompts=100, seed=0):
    """Like passkey_prompts, with n_needles magic numbers inserted at random token positions of the documents' text."""
    rng = np.random.default_rng(seed)
    tokens, start, end = np.full((n_prompts, length + 1), tokenizer.bos_id, np.int32), [], []
    for i, row in enumerate(tokens):
        names, keys = rng.choice(NAMES, n_needles, replace=False), rng.integers(10_000, 100_000, n_needles)
        needles = [tokenizer.encode(MAGIC_NEEDLE.format(name=n, key=k)) for n, k in zip(names, keys)]
        j = rng.integers(n_needles)  # the needle to ask about
        question, answer = tokenizer.encode(MAGIC_QUESTION.format(name=names[j])), tokenizer.encode(f" {keys[j]}")
        n_text = length - sum(map(len, needles)) - len(question) - len(answer)
        text = list(docs[i % len(docs)][1:n_text + 1])  # skip the document's BOS
        for pos, needle in sorted(zip(rng.integers(0, n_text, n_needles), needles), reverse=True):
            text[pos:pos] = needle
        prompt = [tokenizer.bos_id] + text + question
        row[:len(prompt) + len(answer)] = prompt + answer
        start.append(len(prompt))
        end.append(len(prompt) + len(answer))
    return tokens, np.array(start), np.array(end)


def score(params, tokens, cfg):
    """Per-token loss in nats and whether the greedy prediction is right, both (B, T) for tokens of shape (B, T + 1)."""
    logits = model.forward(params, tokens[:, :-1], cfg)
    targets = tokens[:, 1:]
    return model.cross_entropy(logits, targets), jnp.argmax(logits, -1) == targets


class LongEval:
    """Tokenizes the documents and passkey prompts once; calling it evaluates params and returns a flat dict of results:
    'bpb@{lo}-{hi}' per position bucket, 'passkey@{T}' (accuracy over all depths), 'passkey@{T}/{depth}' and
    'needle{n}@{T}' (n needles)."""

    def __init__(self, tokenizer, cfg, max_len):
        if not os.path.exists(DOCS_PATH):
            download_docs()
        with open(DOCS_PATH) as f:
            texts = [json.loads(line)["text"] for line in f]
        self.docs = np.array([d[:max_len + 1] for d in tokenizer.encode_docs(texts) if len(d) > max_len], np.int32)
        assert len(self.docs) > 0, f"no document has {max_len + 1} tokens"
        self.token_bytes = tokenizer.token_bytes()
        self.edges = [0] + [2**k for k in range(8, max_len.bit_length()) if 2**k < max_len] + [max_len]
        self.lengths = [2048 * 2**k for k in range(8) if 2048 * 2**k <= max_len]
        self.passkey = {T: passkey_prompts(tokenizer, T) for T in self.lengths}
        self.needles = {(n, T): needle_prompts(tokenizer, self.docs, T, n) for n in NEEDLE_COUNTS for T in self.lengths}
        self.score = jax.jit(lambda params, tokens: score(params, tokens, cfg))

    def run(self, params, tokens, sharding):
        """(per-token losses, greedy hits) of rows of tokens, scored in batches of TOKENS_PER_DEVICE per device."""
        n_devices = len(sharding.device_set) if sharding else 1
        rows = n_devices * max(1, TOKENS_PER_DEVICE // tokens.shape[1])
        return run_batches(lambda batch: self.score(params, batch), [tokens], rows, sharding)

    def __call__(self, params, sharding=None):
        results = {}
        losses, _ = self.run(params, self.docs, sharding)
        n_bytes = self.token_bytes[self.docs[:, 1:]]
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            nats = np.sum(losses[:, lo:hi] * (n_bytes[:, lo:hi] > 0))
            results[f"bpb@{lo}-{hi}"] = float(nats / (np.log(2) * np.sum(n_bytes[:, lo:hi])))
        for T, prompts in self.passkey.items():
            correct = self.retrieve(params, *prompts, sharding)
            results[f"passkey@{T}"] = float(correct.mean())
            for depth, c in zip(DEPTHS, np.split(correct, len(DEPTHS))):
                results[f"passkey@{T}/{depth}"] = float(c.mean())
        for (n, T), prompts in self.needles.items():
            results[f"needle{n}@{T}"] = float(self.retrieve(params, *prompts, sharding).mean())
        return results

    def retrieve(self, params, tokens, start, end, sharding):
        """Per prompt, whether greedy decoding reproduces its answer tokens [start, end)."""
        _, hits = self.run(params, tokens, sharding)
        return np.array([np.all(h[s - 1:e - 1]) for h, s, e in zip(hits, start, end)])
