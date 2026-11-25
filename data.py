"""
Pretraining data: a loader that packs documents into training rows.

On-disk layout (under BASE_DIR), produced by prepare.py (see prepare.SOURCES for the sources):
    raw/<source>/{train,val}/**.parquet                 downloaded documents, in a "text" column
    tokens/<tokenizer>/<source>/{train,val}/*.bin       uint16 token streams in which every document starts with BOS

Packing follows nanochat's BOS-aligned best-fit: every row starts at a document boundary, documents are placed
largest-that-fits-first from a buffer, and when nothing fits, the shortest one is cropped to fill the row exactly.
"""

import glob
import os
import queue
import threading
from collections import deque

import numpy as np

from config import BASE_DIR

def raw_files(source, split):
    return sorted(glob.glob(os.path.join(BASE_DIR, "raw", source, split, "**", "*.parquet"), recursive=True))


def token_files(tokenizer, source, split):
    return sorted(glob.glob(os.path.join(BASE_DIR, "tokens", tokenizer, source, split, "*.bin")))


def parse_mix(mix):
    """'climbmix:0.9,finemath:0.1' -> {'climbmix': 0.9, 'finemath': 0.1}"""
    return {name: float(weight) for name, weight in (item.split(":") for item in mix.split(","))}


class DocStream:
    """The documents of one source in order, cycling over its .bin files forever.
    Its position (file index, token offset) is saved in checkpoints; on resume we skip at most one chunk."""

    def __init__(self, files, bos_id, file_idx=0, offset=0, chunk_tokens=2**22):
        assert files, "no token files found, run prepare.py first"
        self.files, self.bos_id, self.chunk_tokens = files, bos_id, chunk_tokens
        self.file_idx, self.offset = file_idx, offset
        self.docs = deque()

    def state(self):
        return [self.file_idx, self.offset]

    def next_doc(self):
        while not self.docs:
            self._read_chunk()
        return self.docs.popleft()

    def _read_chunk(self):
        tokens = np.memmap(self.files[self.file_idx], dtype=np.uint16, mode="r")
        chunk = tokens[self.offset:self.offset + self.chunk_tokens]
        starts = np.flatnonzero(chunk == self.bos_id)
        if self.offset + len(chunk) < len(tokens) and len(starts) > 1:
            chunk, starts = chunk[:starts[-1]], starts[:-1]  # stop before the last (possibly incomplete) document
        self.docs.extend(np.split(chunk, starts[1:]))
        self.offset += len(chunk)
        if self.offset >= len(tokens):
            self.file_idx, self.offset = (self.file_idx + 1) % len(self.files), 0


class Loader:
    """Yields (tokens, state): tokens is int32 (rows, seq_len + 1); state can be passed back in to resume.
    Train batches use cfg.train_mix, switching to cfg.anneal_mix for the final cfg.anneal_frac of training.
    Val batches come from cfg.val_source and are identical every time a fresh val Loader is made."""

    def __init__(self, cfg, split, rows, bos_id, start_step=0, state=None, buffer_size=1000):
        self.rows, self.row_len, self.buffer_size = rows, cfg.seq_len + 1, buffer_size
        self.long_doc_rows = cfg.long_doc_rows and split == "train"
        if split == "train":
            self.mixes = [parse_mix(cfg.train_mix), parse_mix(cfg.anneal_mix or cfg.train_mix)]
            self.anneal_step = int((1 - cfg.anneal_frac) * cfg.num_steps)
        else:
            self.mixes, self.anneal_step = [{cfg.val_source: 1.0}], float("inf")
        state = state or dict(streams={}, rng=None)
        names = {name for mix in self.mixes for name in mix}
        self.streams = {name: DocStream(token_files(cfg.tokenizer, name, split), bos_id,
                                        *state["streams"].get(name, (0, 0))) for name in sorted(names)}
        self.rng = np.random.default_rng(cfg.seed)
        if state["rng"]:
            self.rng.bit_generator.state = state["rng"]
        self.step = start_step
        self.docs, self.lens, self.sources = [], [], []  # buffer of documents to pack from

    def state(self):
        return dict(streams={name: s.state() for name, s in self.streams.items()}, rng=self.rng.bit_generator.state)

    def __iter__(self):
        return self

    def next_doc(self, mix, pos):
        """The next document to place in a row filled up to pos (removed from the buffer), with its source."""
        names, weights = list(mix), np.array(list(mix.values()))
        while len(self.docs) < self.buffer_size:
            name = names[self.rng.choice(len(names), p=weights / weights.sum())]
            doc = self.streams[name].next_doc()
            self.docs.append(doc)
            self.lens.append(len(doc))
            self.sources.append(name)
        remaining = self.row_len - pos
        lens = np.array(self.lens)
        fits = np.where(lens <= remaining, lens, 0)
        if pos == 0 and self.long_doc_rows and lens.max() > remaining:
            i = int(np.argmax(lens))  # a whole row of the start of the longest document
        elif fits.max() > 0:
            i = int(np.argmax(fits))
        else:
            i = int(np.argmin(lens))  # crop the shortest
        self.lens.pop(i)
        return self.docs.pop(i), self.sources.pop(i)

    def __next__(self):
        mix = self.mixes[-1] if self.step >= self.anneal_step else self.mixes[0]
        out = np.empty((self.rows, self.row_len), dtype=np.int32)
        for row in out:
            pos = 0
            while pos < self.row_len:
                doc, _ = self.next_doc(mix, pos)
                n = min(len(doc), self.row_len - pos)
                row[pos:pos + n] = doc[:n]
                pos += n
        self.step += 1
        return out, self.state()


def prefetch(iterator, size=4):
    """Runs an iterator in a background thread, keeping up to `size` items ready."""
    q = queue.Queue(maxsize=size)

    def worker():
        try:
            for item in iterator:
                q.put(item)
        except Exception as e:
            q.put(e)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, Exception):
            raise item
        yield item
