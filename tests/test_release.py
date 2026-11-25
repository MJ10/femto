"""The release formats: a model saved as model.safetensors + config.json loads to the same params and logits as its
training checkpoint, and a tokenizer saved as tokenizer.json encodes as the one it was saved from."""

import dataclasses
import json
import os

import jax
import numpy as np

import checkpoint
import model
import optim
from config import Config
from tokenizer import SPECIAL_TOKENS, Tokenizer

CFG = Config(vocab_size=320, n_layer=2, d_model=128, head_dim=64, seq_len=64, dtype="float32")


def test_safetensors_model_matches_checkpoint(tmp_path):
    params = model.init_params(CFG, jax.random.key(0))
    ckpt = str(tmp_path / "ckpt.npz")
    checkpoint.save(ckpt, (params, optim.init(params)), dict(step=7, config=dataclasses.asdict(CFG)))
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    arrays = {checkpoint.param_name(k): np.asarray(v) for k, v in leaves}
    assert "blocks.1.wq" in arrays and "resid_lambdas" in arrays
    release = tmp_path / "release"
    release.mkdir()
    checkpoint.write_safetensors(str(release / "model.safetensors"), arrays, dict(name="test"))
    with open(release / "config.json", "w") as f:
        json.dump(dict(step=7, config=dataclasses.asdict(CFG)), f)

    a, cfg_a, _ = checkpoint.load_model(ckpt)
    b, cfg_b, meta = checkpoint.load_model(str(release / "model.safetensors"))
    assert cfg_a == cfg_b == CFG and meta["step"] == 7
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    tokens = jax.random.randint(jax.random.key(1), (2, CFG.seq_len), 0, CFG.vocab_size)
    assert np.array_equal(model.forward(a, tokens, CFG), model.forward(b, tokens, CFG))
    with open(release / "model.safetensors", "rb") as f:  # a standard header: 8-byte size, then JSON
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    assert header["__metadata__"] == {"name": "test"} and header["wte"]["dtype"] == "F32"


def test_tokenizer_json_roundtrip(tmp_path):
    tok = Tokenizer.train(["hello world, the tokenizer of the release", "ls -la | wc -l"] * 20,
                          vocab_size=len(SPECIAL_TOKENS) + 300)
    tok.save_file(str(tmp_path / "tokenizer.json"))
    again = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    text = "hello <|user_start|>ls -la<|user_end|> wörld 123456"
    assert again.vocab_size == tok.vocab_size and again.bos_id == tok.bos_id
    assert again.enc.encode(text, allowed_special="all") == tok.enc.encode(text, allowed_special="all")
    assert again.decode(again.encode(text)) == text
    assert os.path.getsize(tmp_path / "tokenizer.json") > 0
