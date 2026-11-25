"""
Evals of a saved run on one device, at its training length unless --seq_len overrides it (any other config field can be
overridden too, e.g. --global_attn_temp=0.13 for the long-context evals beyond the training length):
    bpb    val bits per byte on cfg.val_source (ClimbMix), as train.py's in-loop eval
    core   CORE (core_eval.py), all examples by default (--max_per_task=500 is the in-loop subset)
    tool   TOOL + shell / man / python bits per byte (tool_eval.py)
    long   bits per byte by position, passkey and needles up to --long_eval_len (default 32768; long_eval.py)
Execution-based agent evals (tool calls really run) are agent_eval.py.
    python evaluate.py --run_name=final_10b_mila_16k_lr01 --seq_len=2048                      # bpb, core, tool at 2k
    python evaluate.py --run_name=final_10b_mila_16k_lr01 --evals=long [--global_attn_temp=0.13]
The results are printed and appended as a JSON line to BASE_DIR/runs/<run_name>/evals.jsonl.
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp

import checkpoint
from config import parse_overrides
from core_eval import CoreEval
from long_eval import LongEval
from tokenizer import Tokenizer
from tool_eval import ToolEval
from train import eval_step, evaluate_bpb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--evals", default="bpb,core,tool", help="of bpb, core, tool, long")
    parser.add_argument("--max_per_task", type=int, default=-1, help="CORE examples per task (-1 = all)")
    args, rest = parser.parse_known_args()
    params, cfg, meta = checkpoint.load_model(checkpoint.model_path(args.run_name), **parse_overrides(rest))
    tok = Tokenizer.load(cfg.tokenizer)
    evals = args.evals.split(",")
    out = dict(run_name=args.run_name, step=meta["step"], overrides=rest)
    if "bpb" in evals:
        eval_fn = jax.jit(lambda *batch: eval_step(*batch, cfg))
        rows = max(1, 2**16 // cfg.seq_len)
        out["val_bpb"] = evaluate_bpb(eval_fn, params, jnp.asarray(tok.token_bytes()), tok.bos_id, rows, None, cfg)
    if "core" in evals:
        out["core"] = CoreEval(tok, cfg, args.max_per_task)(params)
    if "tool" in evals:
        out["tool"] = ToolEval(tok, cfg)(params)
    if "long" in evals:
        out["long"] = LongEval(tok, cfg, cfg.long_eval_len or 32768)(params)
    out = json.loads(json.dumps(out, default=float))  # numpy scalars -> floats
    for key, value in out.items():
        for k, v in (value.items() if isinstance(value, dict) else [(key, value)]):
            if "/" not in k or key == "tool":
                print(f"{k:<35} {v:.4f}" if isinstance(v, float) else f"{k:<35} {v}")
    with open(os.path.join(os.path.dirname(checkpoint.run_path(args.run_name)), "evals.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
