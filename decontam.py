"""
Decontamination: drops training documents that share a 13-gram of words with an eval set, while downloading
(prepare.py download --decontaminate; see filtered below).
Eval sets: every string of the CORE eval bundle (eval_bundle/eval_data, all tasks), our TOOL / bpb / long-context and
BFCL / QA / weather sets (BASE_DIR/evals, build_evals.py), and the GSM8K, ARC and MMLU test splits. Words are lowercased \\w+ runs. Eval strings
of 8-12 words are matched whole, so that short items (e.g. NL2Bash tasks) are caught too; shorter ones are ignored, and
so are eval n-grams with fewer than MIN_ALPHA real words (e.g. '1 2 3 4 5 6 7 8', which would match anything).
"""

import glob
import json
import os
import re
import zlib
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from config import BASE_DIR

N = 13
MIN_WORDS = 8
MIN_ALPHA = 6  # words of 2+ letters
WORD = re.compile(r"\w+")
P = np.uint64(1_000_003)  # base of the polynomial hash of a word sequence (mod 2^64)


def word_ids(text):
    return np.array([zlib.crc32(w.encode()) for w in WORD.findall(text.lower())], dtype=np.uint64)


def ngram_hashes(ids, n):
    """Hashes of all n-word windows of ids."""
    if len(ids) < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(ids) - n + 1, dtype=np.uint64)
    for k in range(n):
        h = h * P + ids[k:len(ids) - n + 1 + k]
    return h


def strings(value):
    """All strings in a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def hf_rows(repo, path):
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()


def eval_texts():
    """(eval set, text) for every string of every eval set."""
    for path in sorted(glob.glob(os.path.join(BASE_DIR, "eval_bundle", "eval_data", "**", "*.jsonl"), recursive=True) +
                       glob.glob(os.path.join(BASE_DIR, "evals", "**", "*.jsonl"), recursive=True)):
        name = os.path.relpath(path, BASE_DIR).replace("eval_bundle/eval_data/", "core/").removesuffix(".jsonl")
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # the last line of the bundle's narrative_qa.jsonl is cut off
                    record = line
                for text in strings(record):
                    yield name, text
    for row in hf_rows("openai/gsm8k", "main/test-00000-of-00001.parquet"):
        yield "gsm8k_test", row["question"]
        yield "gsm8k_test", row["answer"]
    for subset in ("ARC-Easy", "ARC-Challenge"):
        for row in hf_rows("allenai/ai2_arc", f"{subset}/test-00000-of-00001.parquet"):
            for text in [row["question"]] + row["choices"]["text"]:
                yield "arc_test", text
    for row in hf_rows("cais/mmlu", "all/test-00000-of-00001.parquet"):
        for text in [row["question"]] + row["choices"]:
            yield "mmlu_test", text


class EvalIndex:
    """Sorted hashes of the eval sets' 13-grams (and of their whole 8-12 word strings), with the eval set of each."""

    def __init__(self):
        names, hashes = [], {n: ([], []) for n in range(MIN_WORDS, N + 1)}
        for name, text in eval_texts():
            if not names or names[-1] != name:
                names.append(name)
            ids = word_ids(text)
            if len(ids) >= MIN_WORDS:
                n = min(len(ids), N)
                alpha = np.cumsum([0] + [len(w) >= 2 and w.isalpha() for w in WORD.findall(text.lower())])
                h = ngram_hashes(ids, n)[alpha[n:] - alpha[:-n] >= MIN_ALPHA]
                hashes[n][0].append(h)
                hashes[n][1].append(np.full(len(h), len(names) - 1, dtype=np.int16))
        self.names, self.hashes, self.sets = names, {}, {}
        for n, (h, s) in hashes.items():
            h, s = np.concatenate(h or [np.zeros(0, np.uint64)]), np.concatenate(s or [np.zeros(0, np.int16)])
            if len(h):
                order = np.argsort(h, kind="stable")
                self.hashes[n], self.sets[n] = h[order], s[order]
        print(f"eval index: {len(names)} eval sets, {sum(map(len, self.hashes.values())):,} n-grams")

    def match(self, text):
        """(eval set, the shared words) for the first n-gram of text found in an eval set, or None."""
        ids = word_ids(text)
        for n, table in self.hashes.items():
            h = ngram_hashes(ids, n)
            i = np.minimum(np.searchsorted(table, h), len(table) - 1)
            hit = np.flatnonzero(table[i] == h)
            if len(hit):
                start = int(hit[0])
                return self.names[self.sets[n][i[start]]], " ".join(WORD.findall(text.lower())[start:start + n])
        return None


def filtered(docs, name):
    """Yields the documents that match no eval set, then prints how many were dropped (per eval set)."""
    index = EvalIndex()
    n_docs, dropped = 0, Counter()
    try:
        for doc in docs:
            n_docs += 1
            if (match := index.match(doc)) is None:
                yield doc
            else:
                dropped[match[0]] += 1
    finally:  # also when the consumer stops early (--max_chars)
        print(f"{name}: dropped {sum(dropped.values()):,} of {n_docs:,} documents as contaminated: "
              f"{dict(dropped.most_common())}")
