"""Checkpoints. Training checkpoints (train.py / rl.py) are one .npz: a pytree of arrays (params and optimizer state)
plus a JSON-able metadata dict. Released models are BASE_DIR/runs/<name>/model.safetensors (the params only, fp32) and
config.json (the run's config and step) next to it, read and written here with numpy alone. Released files missing
locally are downloaded from the Hugging Face Hub (config.HF_REPO), into the same layout under BASE_DIR."""

import dataclasses
import json
import os
import re

import jax
import numpy as np

import model
from config import BASE_DIR, fetch, saved_config

RELEASED = {  # released model -> the run it was exported from (export.py)
    "femto": "rl_web7",  # final model: two RL stages (900 steps) on mid-training
    "femto-mid": "mt_tool20c",  # chat + tool mid-training (0.9B tokens at 4k)
    "femto-base": "final_10b_mila_16k_lr01",  # base model after the 16k context extension
    "femto-base-2k": "final_10b_mila",  # base model at the end of pretraining (10B tokens at 2k)
}


def save(path, tree, meta):
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    arrays = {jax.tree_util.keystr(k): np.asarray(v) for k, v in leaves}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=json.dumps(meta), **arrays)
    os.replace(tmp, path)  # atomic: a job killed mid-save never leaves a corrupt checkpoint


def load(path, template):
    """Loads arrays into the structure of `template` (e.g. a freshly initialized state)."""
    data = np.load(path)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    tree = treedef.unflatten([data[jax.tree_util.keystr(k)] for k, _ in leaves])
    return tree, json.loads(str(data["__meta__"]))


def run_path(run_name):
    return os.path.join(BASE_DIR, "runs", run_name, "ckpt.npz")


def model_path(name):
    """The file to load a model from: the run's training checkpoint if there is one, else its released
    model.safetensors (downloaded if it is a released model)."""
    if os.path.exists(run_path(name)):
        return run_path(name)
    path = os.path.join(BASE_DIR, "runs", name, "model.safetensors")
    if name in RELEASED:
        fetch(f"runs/{name}/config.json")
        fetch(f"runs/{name}/model.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint for {name!r}: neither {run_path(name)} nor {path} exists "
                                f"(released models: {', '.join(RELEASED)})")
    return path


def param_name(key_path):
    """A params pytree path as a flat tensor name: ['blocks'][3]['wq'] -> blocks.3.wq."""
    return ".".join(a or b for a, b in re.findall(r"\['([^']+)'\]|\[(\d+)\]", jax.tree_util.keystr(key_path)))


def load_model(path, **overrides):
    """(params on the default device, cfg, meta) of a train.py / rl.py checkpoint (without its optimizer state) or a
    released model.safetensors. overrides replace config fields, e.g. eval-time settings (dtype, global_attn_temp)."""
    if path.endswith(".safetensors"):
        with open(os.path.join(os.path.dirname(path), "config.json")) as f:
            meta = json.load(f)
        arrays = read_safetensors(path)
        get = lambda key_path: arrays[param_name(key_path)]
    else:
        data = np.load(path)
        meta = json.loads(str(data["__meta__"]))
        get = lambda key_path: data["[0]" + jax.tree_util.keystr(key_path)]
    cfg = dataclasses.replace(saved_config(meta["config"]), **overrides)
    template = jax.eval_shape(lambda: model.init_params(cfg, jax.random.key(0)))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    params = treedef.unflatten([jax.device_put(get(k)) for k, _ in leaves])
    return params, cfg, meta


# safetensors (https://huggingface.co/docs/safetensors): an 8-byte little-endian header size, a JSON header
# {name: {dtype, shape, data_offsets: [begin, end]}, "__metadata__": {str: str}}, then the raw little-endian bytes.
DTYPES = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I64": np.int64}


def write_safetensors(path, arrays, metadata=None):
    header, offset = {"__metadata__": metadata or {}}, 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        dtype = next(k for k, v in DTYPES.items() if v == a.dtype)
        header[name] = dict(dtype=dtype, shape=list(a.shape), data_offsets=[offset, offset + a.nbytes])
        offset += a.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # pad the header to 8 bytes, as the reference implementation does
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(len(blob).to_bytes(8, "little") + blob)
        for a in arrays.values():
            f.write(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes())
    os.replace(tmp, path)


def read_safetensors(path):
    """{name: array} of a safetensors file, memory-mapped."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    buf = np.memmap(path, np.uint8, mode="r", offset=8 + n)
    return {name: buf[begin:end].view(np.dtype(DTYPES[h["dtype"]]).newbyteorder("<")).reshape(h["shape"])
            for name, h in header.items() for begin, end in [h["data_offsets"]]}
