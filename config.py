"""
All hyperparameters in one place. Any field can be overridden from the command line:
    python train.py --n_layer=4 --d_model=256 --num_tokens=50_000_000
"""

import dataclasses
import os
import sys
from dataclasses import dataclass

# data, tokenizers and runs all live here: $FEMTO_DIR, else $SCRATCH/femto (~/scratch/femto without $SCRATCH)
BASE_DIR = os.environ.get("FEMTO_DIR") or os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~/scratch")), "femto")
# the Hugging Face Hub repo of the released models and tokenizer; it mirrors BASE_DIR's layout (runs/, tokenizers/)
HF_REPO = "mokshjain/femto"


def fetch(file):
    """BASE_DIR/<file> (e.g. runs/femto/model.safetensors), downloaded from HF_REPO first if it isn't there."""
    path = os.path.join(BASE_DIR, file)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        print(f"downloading {file} from huggingface.co/{HF_REPO} to {BASE_DIR}", flush=True)
        hf_hub_download(HF_REPO, file, local_dir=BASE_DIR)
    return path


@dataclass(frozen=True)
class Config:
    # model (defaults: ~48M params total = 31.5M transformer + 2 x 8.4M untied embeddings)
    vocab_size: int = 16384  # must match the tokenizer and be a multiple of 64
    n_layer: int = 10
    d_model: int = 512
    head_dim: int = 128
    seq_len: int = 2048
    window_pattern: str = "SSSL"  # tiled over layers, L=global (full context), S=local; last layer is always L
    local_window: int = 512  # sliding window of the S layers, in tokens (fixed, so it stays local at any length)
    rope_base: float = 10_000.0  # RoPE of the local layers; the global layers have no positional encoding (NoPE)
    global_attn_temp: float = 0.0  # >0 (evals beyond the training length): global-layer queries at position p > L are
    global_attn_temp_len: int = 0  # scaled by 1 + t*log(p/L), where L is this reference length, or seq_len if 0
    logit_softcap: float = 15.0
    dtype: str = "bfloat16"  # compute dtype of activations and matmuls; float32 for exact inference / RL (model.py)

    # batch / duration
    total_batch_tokens: int = 2**18  # tokens per optimizer step (across all devices + grad accumulation)
    device_batch_size: int = 32  # sequences per device per micro-step: 32 fits an A100-80GB, use 8 on an L40S
    num_tokens: int = 2_500_000_000  # total training tokens

    # optimizer (nanochat defaults; LRs were tuned at batch 2^19 and d_model 768, see optim.py for rescaling)
    matrix_lr: float = 0.02  # Muon, for all hidden 2D weights
    embedding_lr: float = 0.3  # AdamW, wte
    unembedding_lr: float = 0.008  # AdamW, lm_head
    scalar_lr: float = 0.5  # AdamW, per-layer lambdas
    weight_decay: float = 0.1  # Muon cautious weight decay, cosine-decayed to 0 over training
    lr_scale: float = 1.0  # multiplies all LRs, e.g. < 1 for a continued-training stage (see init_from)
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65  # fraction of training spent linearly decaying the LR
    final_lr_frac: float = 0.05

    # data: a mix is "source:weight,..." over the sources in prepare.SOURCES
    tokenizer: str = "v2"  # BASE_DIR/tokenizers/<tokenizer>, training data from BASE_DIR/tokens/<tokenizer>/<source>
    train_mix: str = "climbmix:0.90,shell:0.04,python:0.03,sysadmin_qa:0.03"  # 10% code (best trade-off in the sweep)
    anneal_mix: str = ""  # mix used for the final anneal_frac of training ("" = keep train_mix)
    anneal_frac: float = 0.15
    val_source: str = "climbmix"
    # loss masking for documents in the chat format (train.assistant_mask): "all" tokens; "assistant": only the
    # assistant's tokens of chat documents (SFT style); "anneal": "all", then "assistant" for the final anneal_frac
    loss_mask: str = "all"
    long_doc_rows: bool = False  # start each row with the longest buffered document if it fills the row (long context)

    # eval / logging / checkpointing
    eval_every: int = 250  # validation bits per byte
    eval_tokens: int = 10 * 2**20
    core_every: int = 2000  # CORE benchmark (22 tasks incl. HellaSwag, ARC, PIQA, LAMBADA); 0 = off
    core_max_per_task: int = 500
    tool_every: int = -1  # TOOL benchmark + shell/man/python bits per byte; -1 = same as core_every, 0 = off
    long_eval_len: int = 0  # >0: long-context evals (long_eval.py) up to this length after the final checkpoint
    sample_every: int = 2000  # greedy samples from fixed prompts; 0 = off
    log_every: int = 10
    save_every: int = 500
    run_name: str = "default"  # checkpoints go to BASE_DIR/runs/run_name; rerunning a name resumes it
    init_from: str = ""  # start a new run (stage) from the params of this run, e.g. a long-context extension
    init_optimizer: bool = False  # with init_from: also load its optimizer state (Muon / Adam moments)
    wandb_project: str = "femto"  # empty = no wandb
    seed: int = 0

    @property
    def out_dir(self) -> str:
        return os.path.join(BASE_DIR, "runs", self.run_name)

    @property
    def num_steps(self) -> int:
        return self.num_tokens // self.total_batch_tokens


def parse_overrides(argv) -> dict:
    """{field: value} from --key=value args, casting each value to the field's type."""
    overrides = {}
    fields = {f.name: f for f in dataclasses.fields(Config)}
    for arg in argv:
        assert arg.startswith("--") and "=" in arg, f"expected --key=value, got {arg!r}"
        key, value = arg[2:].split("=", 1)
        assert key in fields, f"unknown config key {key!r}"
        typ = type(fields[key].default)
        if typ is bool:
            assert value in ("True", "False"), f"expected True or False for {key}, got {value!r}"
            overrides[key] = value == "True"
        else:
            overrides[key] = typ(value.replace("_", "")) if typ in (int, float) else typ(value)
    return overrides


def parse_args(argv=None) -> Config:
    return Config(**parse_overrides(sys.argv[1:] if argv is None else argv))


def saved_config(stored: dict) -> Config:
    """Config of a saved run. Options that no longer exist are dropped: they are ablation knobs whose settled value is
    now the only behaviour (NoPE global layers, no document masking, ...); runs that changed them need the code they were trained with."""
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in stored.items() if k in fields})
