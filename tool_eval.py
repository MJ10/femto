"""
Secondary evals for the agentic substrate (bash + python), built by build_evals.py under BASE_DIR/evals as raw text:
- TOOL: 4-way multiple choice tasks scored exactly like CORE (lowest mean continuation loss wins, accuracy centered by
  the 25% random baseline, averaged over tasks): NL -> command (NL2SH-ALFA, nl2bash test splits), and predicting the
  output of shell commands, python expressions and python scripts
- bits per byte on held-out text: shell commands (overall, and the command only given its task), man pages, python.
  Tokenizer independent: summed loss in nats / (ln 2 * UTF-8 bytes of the scored tokens)
TOOL of a saved run: eval.py.
"""

import glob
import json
import math
import os

import numpy as np

import build_evals
from build_evals import EVALS_DIR
from core_eval import CoreEval, Scorer, common_length

TOOL_TASKS = {  # label: (few-shot examples, continuation delimiter)
    "nl2sh_alfa": (3, " "),
    "nl2bash": (3, " "),
    "shell_output": (2, "\n"),
    "python_expr": (3, "\n"),
    "python_script": (2, "\n"),
}


def tool_tasks():
    return [dict(label=label, type="multiple_choice", num_fewshot=k, delim=delim, baseline=1 / build_evals.N_CHOICES,
                 path=os.path.join(EVALS_DIR, "tool", f"{label}.jsonl")) for label, (k, delim) in TOOL_TASKS.items()]


class BpbEval:
    """Bits per byte of each bpb/<set>.jsonl. Documents longer than seq_len are split into windows that overlap by one
    token, so every scored token is predicted exactly once."""

    def __init__(self, tokenizer, cfg):
        token_bytes = tokenizer.token_bytes()
        self.sets, rows, n_bytes = {}, [], []
        for path in sorted(glob.glob(os.path.join(EVALS_DIR, "bpb", "*.jsonl"))):
            with open(path) as f:
                items = [json.loads(line) for line in f]
            first = len(rows)
            for item in items:
                context, full = tokenizer.encode_docs([item["context"], item["context"] + item["text"]])
                start = common_length([context, full])  # first scored token
                for a in range(0, len(full) - 1, cfg.seq_len - 1):
                    window = full[a:a + cfg.seq_len]
                    first_scored = max(start - a, 1)
                    if first_scored < len(window):
                        rows.append((window, first_scored, len(window)))
                        n_bytes.append(token_bytes[window[first_scored:]].sum())
            self.sets[os.path.basename(path)[:-len(".jsonl")]] = slice(first, len(rows))
        self.n_bytes = np.array(n_bytes)
        self.scorer = Scorer(rows, tokenizer.bos_id, cfg)

    def __call__(self, params, sharding=None):
        loss_sum, _ = self.scorer(params, sharding)
        return {name: loss_sum[s].sum() / (math.log(2) * self.n_bytes[s].sum()) for name, s in self.sets.items()}


class ToolEval:
    """Calling it returns {'tool/<task>': centered accuracy, 'tool/TOOL': mean, 'bpb/<set>': bits per byte}."""

    def __init__(self, tokenizer, cfg):
        build_evals.build()  # first use only: downloads and generates whatever is missing
        self.tool = CoreEval(tokenizer, cfg, max_per_task=-1, tasks=tool_tasks(), name="TOOL")
        self.bpb = BpbEval(tokenizer, cfg)

    def __call__(self, params, sharding=None):
        results = {f"tool/{k}": v for k, v in self.tool(params, sharding).items()}
        results.update({f"bpb/{k}": v for k, v in self.bpb(params, sharding).items()})
        return results
