"""BOS-aligned best-fit packing: every row starts with a document, and only the last document of a row is cropped."""

import numpy as np

import data
from config import Config

BOS = 0


def test_packing(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    docs = [[BOS] + rng.integers(1, 50, n).tolist() for n in rng.integers(2, 40, 300)]
    path = tmp_path / "docs.bin"
    np.array(sum(docs, []), np.uint16).tofile(path)
    monkeypatch.setattr(data, "token_files", lambda *_: [str(path)])
    cfg = Config(seq_len=63, train_mix="docs:1", num_tokens=10, total_batch_tokens=1)
    rows, _ = next(data.Loader(cfg, "train", 32, BOS, buffer_size=20))
    whole = {tuple(d) for d in docs}
    n_whole = 0
    for row in rows:
        starts = list(np.flatnonzero(row == BOS)) + [len(row)]
        pieces = [tuple(row[a:b].tolist()) for a, b in zip(starts, starts[1:])]
        assert starts[0] == 0 and all(p in whole for p in pieces[:-1])
        assert any(d[:len(pieces[-1])] == pieces[-1] for d in whole)
        n_whole += len(pieces) - 1
    assert n_whole > len(rows)  # rows are mostly whole documents
