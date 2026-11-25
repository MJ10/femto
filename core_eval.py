"""
CORE metric from the DCLM paper (https://arxiv.org/abs/2406.11794), ported from nanochat's core_eval.py.
22 tasks (HellaSwag, ARC, PIQA, LAMBADA, Winograd, ...) scored as accuracy centered by the random baseline, then averaged:
- multiple choice / schema: the option with the lowest mean loss over its continuation tokens wins
- language modeling: correct only if greedy decoding reproduces the whole continuation
Sequences are padded into a few fixed lengths so the jitted scoring function compiles once per length.
Full CORE of a saved run: eval.py.
"""

import csv
import io
import json
import os
import random
import urllib.request
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import model
from config import BASE_DIR

BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
BUNDLE_DIR = os.path.join(BASE_DIR, "eval_bundle")
LENGTHS = [64, 128, 256, 512, 1024, 2048]  # sequences are padded up to one of these


def render(task_type, item, fewshot, delim):
    """Returns the prompts for one example, following nanochat's (jinja) templates exactly."""
    if task_type == "multiple_choice":
        prefix = "".join(f"{ex['query']}{delim}{ex['choices'][ex['gold']]}\n\n" for ex in fewshot)
        return [f"{prefix}{item['query']}{delim}{choice}" for choice in item["choices"]]
    if task_type == "schema":
        prefix = "".join(f"{ex['context_options'][ex['gold']]}{delim}{ex['continuation']}\n\n" for ex in fewshot)
        return [f"{prefix}{context}{delim}{item['continuation']}" for context in item["context_options"]]
    prefix = "".join(f"{ex['context'].strip()}{delim}{ex['continuation']}\n\n" for ex in fewshot)
    prompt = f"{prefix}{item['context'].strip()}{delim}"
    return [prompt.strip(), prompt + item["continuation"]]  # without / with the continuation


def common_length(seqs, from_end=False):
    """Length of the common prefix (or suffix) of token sequences."""
    n = 0
    for tokens in zip(*(reversed(s) if from_end else s for s in seqs)):
        if any(t != tokens[0] for t in tokens):
            break
        n += 1
    return n


def continuation_spans(task_type, seqs):
    """[start, end) token span to score in each sequence."""
    if task_type == "multiple_choice":  # same context, different continuations
        start = common_length(seqs)
        return [(start, len(s)) for s in seqs]
    if task_type == "schema":  # different contexts, same continuation
        n = common_length(seqs, from_end=True)
        return [(len(s) - n, len(s)) for s in seqs]
    without, with_ = seqs  # language modeling: only score the full prompt
    assert with_[:len(without)] == without
    return [(len(without), len(with_))]


def score_fn(params, tokens, start, end, cfg):
    """Per row: summed loss over target tokens [start, end), and whether greedy decoding gets all of them right."""
    logits = model.forward(params, tokens[:, :-1], cfg)
    targets = tokens[:, 1:]
    losses = model.cross_entropy(logits, targets)
    correct = jnp.argmax(logits, -1) == targets
    pos = jnp.arange(targets.shape[1])
    mask = (pos >= start[:, None] - 1) & (pos < end[:, None] - 1)  # logits at t predict token t + 1
    return jnp.sum(losses * mask, 1), jnp.all(correct | ~mask, 1)


score = jax.jit(score_fn, static_argnums=4)  # shared by all evals, so each (cfg, length) compiles once


def run_batches(fn, arrays, rows, sharding=None):
    """fn(*batch) over the rows of arrays in batches of `rows` rows. The last batch is padded with copies of its last
    row, so every call has the same shape (rows must divide over the devices of sharding). Returns fn's outputs
    concatenated over the batches, without the padding."""
    outs = []
    for i in range(0, len(arrays[0]), rows):
        batch = [a[i:i + rows] for a in arrays]
        n = len(batch[0])
        batch = [np.concatenate([b, np.repeat(b[-1:], rows - n, 0)]) for b in batch]
        batch = jax.device_put(batch, sharding) if sharding else batch
        outs.append([o[:n] for o in jax.device_get(fn(*batch))])
    return [np.concatenate(o) for o in zip(*outs)]


class Scorer:
    """Scores a fixed list of (tokens, start, end) rows with score_fn. Rows are padded into a few fixed lengths so the
    scoring function compiles once per length."""

    def __init__(self, rows, pad_id, cfg):
        lengths = [n for n in LENGTHS if n < cfg.seq_len] + [cfg.seq_len]
        bucket = [next(n for n in lengths if len(r[0]) <= n) for r in rows]  # the smallest length that fits
        self.cfg, self.n_rows, self.groups = cfg, len(rows), {}
        for length in lengths:
            idx = [i for i, b in enumerate(bucket) if b == length]
            if idx:
                tokens = np.full((len(idx), length), pad_id, dtype=np.int32)
                for row, i in zip(tokens, idx):
                    row[:len(rows[i][0])] = rows[i][0]
                self.groups[length] = dict(idx=np.array(idx), tokens=tokens,
                                           start=np.array([rows[i][1] for i in idx], np.int32),
                                           end=np.array([rows[i][2] for i in idx], np.int32))

    def __call__(self, params, sharding=None, tokens_per_batch=2**16):
        """Returns (summed loss, all correct) per row. Each batch holds ~tokens_per_batch tokens (at least one row per
        device); tokens_per_batch // length must be a multiple of the number of devices if it exceeds it."""
        loss_sum, all_correct = np.zeros(self.n_rows), np.zeros(self.n_rows, dtype=bool)
        n_devices = len(sharding.device_set) if sharding else 1
        for length, g in self.groups.items():
            rows = max(tokens_per_batch // length, n_devices)
            fn = lambda *batch: score(params, *batch, self.cfg)
            loss_sum[g["idx"]], all_correct[g["idx"]] = run_batches(fn, [g["tokens"], g["start"], g["end"]], rows, sharding)
        return loss_sum, all_correct


def download_bundle():
    if not os.path.exists(BUNDLE_DIR):
        print(f"downloading the CORE eval bundle to {BUNDLE_DIR}")
        with urllib.request.urlopen(BUNDLE_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(BASE_DIR)


def core_tasks():
    """The 22 CORE tasks: dicts of label, type, num_fewshot, delim, baseline (random-guess accuracy) and path."""
    download_bundle()
    with open(os.path.join(BUNDLE_DIR, "core.yaml")) as f:
        tasks = yaml.safe_load(f)["icl_tasks"]
    with open(os.path.join(BUNDLE_DIR, "eval_meta_data.csv")) as f:
        baselines = {row["Eval Task"]: float(row["Random baseline"]) / 100 for row in csv.DictReader(f)}
    return [dict(label=t["label"], type=t["icl_task_type"], num_fewshot=t["num_fewshot"][0],
                 delim=t.get("continuation_delimiter", " "), baseline=baselines[t["label"]],
                 path=os.path.join(BUNDLE_DIR, "eval_data", t["dataset_uri"])) for t in tasks]


class CoreEval:
    """Renders and tokenizes all tasks once; calling it evaluates params and returns {task: centered accuracy}.
    tasks defaults to CORE's; other task sets in the same format can be scored the same way (see tool_eval.py)."""

    def __init__(self, tokenizer, cfg, max_per_task=500, tasks=None, name="CORE"):
        self.tasks, self.name, rows = [], name, []  # rows: (tokens, start, end) of every sequence to score
        for task in core_tasks() if tasks is None else tasks:
            with open(task["path"]) as f:
                data = [json.loads(line) for line in f]
            random.Random(1337).shuffle(data)
            data = data[:max_per_task] if max_per_task > 0 else data
            task_type, k, delim = task["type"], task["num_fewshot"], task["delim"]
            example_rows = []  # per example, the indices of its rows (one per option)
            for i, item in enumerate(data):
                others = [j for j in range(len(data)) if j != i]
                fewshot = [data[j] for j in random.Random(1234 + i).sample(others, k)]
                seqs = tokenizer.encode_docs(render(task_type, item, fewshot, delim))
                spans = continuation_spans(task_type, seqs)
                seqs = seqs[-1:] if task_type == "language_modeling" else seqs
                example_rows.append(list(range(len(rows), len(rows) + len(seqs))))
                for seq, (start, end) in zip(seqs, spans):
                    crop = max(0, len(seq) - cfg.seq_len)  # too long: drop tokens from the left
                    rows.append((seq[crop:], start - crop, end - crop))
            self.tasks.append(dict(label=task["label"], type=task_type, rows=example_rows,
                                   golds=[item.get("gold") for item in data], baseline=task["baseline"]))
        self.n_tokens = np.array([end - start for _, start, end in rows])
        self.scorer = Scorer(rows, tokenizer.bos_id, cfg)

    def __call__(self, params, sharding=None, tokens_per_batch=2**16):
        """Returns {task: centered accuracy, name: mean}."""
        loss_sum, all_correct = self.scorer(params, sharding, tokens_per_batch)
        mean_loss = loss_sum / self.n_tokens
        results = {}
        for task in self.tasks:
            if task["type"] == "language_modeling":
                hits = [all_correct[r[0]] for r in task["rows"]]
            else:
                hits = [np.argmin(mean_loss[r]) == gold for r, gold in zip(task["rows"], task["golds"])]
            results[task["label"]] = (np.mean(hits) - task["baseline"]) / (1 - task["baseline"])
        results[self.name] = float(np.mean(list(results.values())))
        return results
