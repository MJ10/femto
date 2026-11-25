"""
Export training checkpoints as released models, and check them: runs/<run>/ckpt.npz -> runs/<name>/model.safetensors
(the params, fp32) + config.json (the run's config, step and RL settings), and the tokenizer's tokenizer.pkl (the
format before the release) -> tokenizer.json. Then the logits of the exported model must equal the checkpoint's bit
for bit, and the tokenizers must encode a sample of text identically.
    python export.py                          # every model of checkpoint.RELEASED, and tokenizer v2
    python export.py --names=femto            # one of them
"""

import argparse
import json
import os
import pickle

import jax
import numpy as np

import checkpoint
import model
from config import BASE_DIR
from tokenizer import TOKENIZERS_DIR, Tokenizer

SAMPLE = ("Hello world! Numbers: 123, 4567. Unicode: 你好世界 🌍\n<|user_start|>How many lines?<|user_end|>"
          "<|assistant_start|><|tool_start|>bash\nwc -l notes.txt<|tool_end|>\n\tdef f(x):\n        return x ** 2\n")


def export_model(name, run):
    data = np.load(checkpoint.run_path(run))
    meta = json.loads(str(data["__meta__"]))
    params, cfg, _ = checkpoint.load_model(checkpoint.run_path(run))
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    arrays = {checkpoint.param_name(k): np.asarray(v) for k, v in leaves}
    assert all(a.dtype == np.float32 for a in arrays.values())
    out = os.path.join(BASE_DIR, "runs", name)
    os.makedirs(out, exist_ok=True)
    config = dict(name=name, run_name=run, step=meta["step"], n_params=int(sum(a.size for a in arrays.values())),
                  config=meta["config"], **({"rl": meta["rl"]} if "rl" in meta else {}))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    checkpoint.write_safetensors(os.path.join(out, "model.safetensors"), arrays, dict(name=name, run_name=run))
    return out


def check_model(name, run):
    """The exported model's logits equal the checkpoint's exactly (fp32, a full 2k context of random tokens)."""
    a, cfg, _ = checkpoint.load_model(checkpoint.run_path(run), dtype="float32")
    b, cfg_b, meta = checkpoint.load_model(os.path.join(BASE_DIR, "runs", name, "model.safetensors"), dtype="float32")
    assert cfg == cfg_b, (cfg, cfg_b)
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    tokens = jax.random.randint(jax.random.key(0), (2, 2048), 0, cfg.vocab_size)
    forward = jax.jit(lambda p: model.forward(p, tokens, cfg))
    la, lb = np.asarray(forward(a)), np.asarray(forward(b))
    assert np.array_equal(la, lb), float(np.abs(la - lb).max())
    print(f"{name} ({run}, step {meta['step']}): {meta['n_params']:,} params, logits identical "
          f"(mean |logit| {np.abs(la).mean():.3f})", flush=True)


def export_tokenizer(name):
    with open(os.path.join(TOKENIZERS_DIR, name, "tokenizer.pkl"), "rb") as f:
        old = Tokenizer(pickle.load(f))
    old.save(name)
    new = Tokenizer.load(name)
    text = SAMPLE * 3 + open(__file__).read()
    assert new.vocab_size == old.vocab_size and new.bos_id == old.bos_id
    assert new.enc.encode(text, allowed_special="all") == old.enc.encode(text, allowed_special="all")
    assert new.encode(text) == old.encode(text) and new.decode(new.encode(text)) == text
    assert all(new.enc.decode_single_token_bytes(i) == old.enc.decode_single_token_bytes(i)
               for i in range(old.vocab_size))
    print(f"tokenizer {name}: {new.vocab_size} tokens, identical encodings", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default=",".join(checkpoint.RELEASED))
    parser.add_argument("--tokenizer", default="v2")
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    if args.tokenizer:
        export_tokenizer(args.tokenizer)
    for name in args.names.split(","):
        out = export_model(name, checkpoint.RELEASED[name])
        check_model(name, checkpoint.RELEASED[name])
        print("  " + ", ".join(f"{f} {os.path.getsize(os.path.join(out, f)) / 1e6:.1f} MB" for f in os.listdir(out)))


if __name__ == "__main__":
    main()
