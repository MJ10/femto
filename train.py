"""
Training loop (pretraining and the later stages). Data-parallel over all local devices: params replicated, batch sharded
on the 'data' axis.
    python train.py                                    # defaults from config.py
    python train.py --n_layer=2 --d_model=128 ...      # override anything
Resumes automatically from BASE_DIR/runs/{run_name}/ckpt.npz if it exists (and does nothing if that run is finished).
On SIGTERM (SLURM preemption / time limit) it saves a checkpoint and exits, so a requeued job picks up where it left off.
"""

import dataclasses
import functools
import math
import os
import signal
import threading
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import checkpoint
import model
import optim
from config import parse_args
from core_eval import CoreEval
from data import Loader, prefetch
from long_eval import LongEval
from sample import generate
from tokenizer import Tokenizer
from tool_eval import ToolEval

SAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]
PEAK_BF16_FLOPS = {"H100": 989e12, "A100": 312e12, "L40S": 362e12}  # dense, per device


def assistant_mask(tokens, ids):
    """Loss weights for SFT-style training on documents in the chat format (chat.py); tokens: (B, T), ids: {special
    token: id}. Tokens of chat documents (those starting with a turn) get 1 inside the assistant's messages (text, tool
    calls, <|assistant_end|>, but not <|assistant_start|>) and 0 elsewhere (system / user turns, tool outputs); other
    documents keep 1."""
    pos = jnp.arange(tokens.shape[1])
    is_ = lambda *names: functools.reduce(jnp.logical_or, [tokens == ids[f"<|{name}|>"] for name in names])
    latest = lambda mask: jax.lax.cummax(jnp.where(mask, pos, -1), axis=1)  # position of the latest token in mask
    in_output = (latest(is_("output_start")) > latest(is_("output_end", "assistant_end", "bos"))) | is_("output_end")
    turn_starts = is_("system_start", "user_start", "assistant_start")
    chat_doc = is_("bos") & jnp.pad(turn_starts[:, 1:], ((0, 0), (0, 1)))  # at a BOS: is the document a chat?
    doc_start = latest(is_("bos"))
    in_chat = (doc_start >= 0) & jnp.take_along_axis(chat_doc, jnp.maximum(doc_start, 0), axis=1)
    opened = latest(is_("assistant_start")) > latest(is_("assistant_end", "system_start", "user_start", "bos"))
    assistant = (opened & ~is_("assistant_start") & ~in_output) | is_("assistant_end")
    return jnp.where(in_chat, assistant, 1.0)


def train_loss(params, tokens, step, cfg, ids):
    """Mean loss over the target tokens, masked as cfg.loss_mask says (see assistant_mask)."""
    losses = model.token_losses(params, tokens[:, :-1], tokens[:, 1:], cfg)
    if cfg.loss_mask == "all":
        return losses.mean()
    weights = assistant_mask(tokens, ids)
    if cfg.loss_mask == "anneal":  # assistant-only from the anneal step on
        weights = jnp.where(step >= int((1 - cfg.anneal_frac) * cfg.num_steps), weights, 1.0)
    weights = weights[:, 1:]  # of the targets
    return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def train_step(params, opt_state, batch, step, cfg, special_ids=None):
    """batch: (grad_accum, batch_size, seq_len + 1). Accumulates grads over the first axis, then updates.
    special_ids ({special token: id}) is needed for cfg.loss_mask."""
    grad_fn = jax.value_and_grad(lambda params, tokens: train_loss(params, tokens, step, cfg, special_ids))

    def micro_step(grads, tokens):
        loss, g = grad_fn(params, tokens)
        return jax.tree.map(jnp.add, grads, g), loss

    grads, losses = jax.lax.scan(micro_step, jax.tree.map(jnp.zeros_like, params), batch)
    grads = jax.tree.map(lambda g: g / batch.shape[0], grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    params, opt_state = optim.update(params, grads, opt_state, step, cfg)
    return params, opt_state, dict(loss=losses.mean(), grad_norm=grad_norm, lrm=optim.lr_multiplier(step, cfg))


def eval_step(params, tokens, token_bytes, cfg):
    """Returns (sum of losses in nats, sum of target bytes), skipping special tokens (0 bytes)."""
    targets = tokens[:, 1:]
    losses = model.token_losses(params, tokens[:, :-1], targets, cfg)
    n_bytes = token_bytes[targets]
    return jnp.sum(losses * (n_bytes > 0)), jnp.sum(n_bytes)


def evaluate_bpb(eval_fn, params, token_bytes, bos_id, batch_size, sharding, cfg):
    """Bits per byte on the first cfg.eval_tokens tokens of cfg.val_source (the same rows at any batch size)."""
    val_batches = Loader(cfg, "val", batch_size, bos_id)  # a fresh loader yields the same batches every time
    nats, n_bytes = 0.0, 0
    for _ in range(max(1, cfg.eval_tokens // (batch_size * cfg.seq_len))):
        tokens = jax.device_put(next(val_batches)[0], sharding)
        batch_nats, batch_bytes = eval_fn(params, tokens, token_bytes)
        nats, n_bytes = nats + float(batch_nats), n_bytes + int(batch_bytes)
    return nats / (math.log(2) * n_bytes)


def main():
    cfg = parse_args()
    print(cfg)

    # batch geometry
    n_devices = jax.device_count()
    batch_size = cfg.device_batch_size * n_devices  # sequences per micro-step, across devices
    assert cfg.total_batch_tokens % (batch_size * cfg.seq_len) == 0
    grad_accum = cfg.total_batch_tokens // (batch_size * cfg.seq_len)
    print(f"{n_devices} x {jax.devices()[0].device_kind} | grad accum {grad_accum} | {cfg.num_steps:,} steps")
    tok = Tokenizer.load(cfg.tokenizer)
    assert tok.vocab_size == cfg.vocab_size, f"tokenizer has {tok.vocab_size} tokens, config says {cfg.vocab_size}"
    token_bytes = jnp.asarray(tok.token_bytes())

    mesh = jax.make_mesh((n_devices,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
    replicated = NamedSharding(mesh, P())
    data_sharded = NamedSharding(mesh, P(None, "data"))  # (grad_accum, batch, seq)
    eval_sharded = NamedSharding(mesh, P("data"))  # (batch, seq)

    # model + optimizer state, resumed from the checkpoint if there is one
    params = model.init_params(cfg, jax.random.key(cfg.seed))
    opt_state = optim.init(params)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"params: {n_params:,} total, {n_params - params['wte'].size:,} excluding wte")
    ckpt_path = checkpoint.run_path(cfg.run_name)
    start_step, data_state = 0, None
    if os.path.exists(ckpt_path):
        (params, opt_state), meta = checkpoint.load(ckpt_path, (params, opt_state))
        start_step, data_state = meta["step"], meta["data_state"]
        if start_step == cfg.num_steps:
            print(f"{ckpt_path} is finished (step {start_step})")
            return
        print(f"resumed from {ckpt_path} at step {start_step}")
    elif cfg.init_from and not os.path.exists(checkpoint.run_path(cfg.init_from)):
        # a released model (params only): a fresh optimizer state, every data source from its start
        assert not cfg.init_optimizer, f"{cfg.init_from} has no optimizer state (it is a released model)"
        init_params, _, meta = checkpoint.load_model(checkpoint.model_path(cfg.init_from))
        params = jax.tree.map(lambda p, q: jnp.asarray(q, p.dtype), params, init_params)
        print(f"initialized from the released {cfg.init_from} (step {meta['step']}) with a fresh optimizer state")
    elif cfg.init_from:
        # a new stage: params from that run, a fresh optimizer state unless cfg.init_optimizer (the LR schedule and Muon
        # momentum warmup restart with this run's schedule either way), data sources continue where that run left them
        init_path = checkpoint.run_path(cfg.init_from)
        (params, init_opt_state), meta = checkpoint.load(init_path, (params, opt_state))
        opt_state = init_opt_state if cfg.init_optimizer else opt_state
        init_cfg = meta["config"]
        assert meta["step"] == init_cfg["num_tokens"] // init_cfg["total_batch_tokens"], f"{cfg.init_from} is unfinished"
        data_state = dict(streams=meta["data_state"]["streams"], rng=None)
        opt = "its" if cfg.init_optimizer else "a fresh"
        print(f"initialized from {init_path} (step {meta['step']}) with {opt} optimizer state")
    params, opt_state = jax.device_put((params, opt_state), replicated)

    special_ids = {name: tok.enc.encode_single_token(name) for name in tok.enc.special_tokens_set}
    step_fn = jax.jit(lambda *args: train_step(*args, cfg, special_ids), donate_argnums=(0, 1))
    eval_fn = jax.jit(lambda *args: eval_step(*args, cfg))
    core_eval = CoreEval(tok, cfg, cfg.core_max_per_task) if cfg.core_every > 0 else None
    tool_every = cfg.core_every if cfg.tool_every < 0 else cfg.tool_every
    tool_eval = ToolEval(tok, cfg) if tool_every > 0 else None
    long_eval = LongEval(tok, cfg, cfg.long_eval_len) if cfg.long_eval_len > 0 else None
    train_batches = prefetch(Loader(cfg, "train", grad_accum * batch_size, tok.bos_id, start_step, data_state))

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def save(step):
        # never overwrite the last good checkpoint with a diverged one (a rerun then resumes from before the NaN)
        assert all(bool(jnp.isfinite(p).all()) for p in jax.tree.leaves(params)), f"non-finite params at step {step}"
        meta = dict(step=step, data_state=data_state, config=dataclasses.asdict(cfg))
        checkpoint.save(ckpt_path, (params, opt_state), meta)
        print(f"saved checkpoint at step {step}")

    wandb = None
    if cfg.wandb_project:
        import wandb
        # the run name is the wandb id, so a resumed run keeps logging to the same wandb run
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, id=cfg.run_name, resume="allow",
                   config=dataclasses.asdict(cfg))

    flops_per_step = model.flops_per_token(cfg) * cfg.total_batch_tokens
    peak_flops = next((f for name, f in PEAK_BF16_FLOPS.items() if name in jax.devices()[0].device_kind), None)
    t0, t0_step = time.time(), start_step

    for step in range(start_step, cfg.num_steps + 1):
        last_step = step == cfg.num_steps

        if cfg.eval_every > 0 and (step % cfg.eval_every == 0 or last_step):
            val_bpb = evaluate_bpb(eval_fn, params, token_bytes, tok.bos_id, batch_size, eval_sharded, cfg)
            print(f"step {step:6d} | val bpb {val_bpb:.4f}")
            if wandb:
                wandb.log({"val/bpb": val_bpb}, step=step)
            t0, t0_step = time.time(), step  # don't count eval time towards throughput

        if core_eval and step > 0 and (step % cfg.core_every == 0 or last_step):
            results = core_eval(params, eval_sharded)
            print(f"step {step:6d} | CORE {results['CORE']:.4f} | " +
                  " | ".join(f"{k} {v:.3f}" for k, v in results.items() if k != "CORE"))
            if wandb:
                wandb.log({f"core/{k}": v for k, v in results.items()}, step=step)
            t0, t0_step = time.time(), step

        if tool_eval and step > 0 and (step % tool_every == 0 or last_step):
            results = tool_eval(params, eval_sharded)
            print(f"step {step:6d} | " + " | ".join(f"{k} {v:.3f}" for k, v in results.items()))
            if wandb:
                wandb.log(results, step=step)
            t0, t0_step = time.time(), step

        if cfg.sample_every > 0 and step > 0 and (step % cfg.sample_every == 0 or last_step):
            completions = generate(params, tok, SAMPLE_PROMPTS, cfg)
            for prompt, completion in zip(SAMPLE_PROMPTS, completions):
                print(f"step {step:6d} | {prompt!r} -> {completion!r}")
            if wandb:
                table = wandb.Table(columns=["prompt", "completion"], data=list(zip(SAMPLE_PROMPTS, completions)))
                wandb.log({"samples": table}, step=step)
            t0, t0_step = time.time(), step

        if last_step or stop.is_set() or (step > start_step and step % cfg.save_every == 0):
            save(step)
        if last_step or stop.is_set():
            break

        tokens, data_state = next(train_batches)
        batch = jax.device_put(tokens.reshape(grad_accum, batch_size, -1), data_sharded)
        params, opt_state, metrics = step_fn(params, opt_state, batch, step)

        if step % cfg.log_every == 0:
            metrics = {k: float(v) for k, v in metrics.items()}  # blocks until the step is done
            assert math.isfinite(metrics["loss"]), f"non-finite loss at step {step}: stopping, nothing saved"
            dt = (time.time() - t0) / max(step + 1 - t0_step, 1)
            t0, t0_step = time.time(), step + 1
            metrics["tok_per_sec"] = cfg.total_batch_tokens / dt
            if peak_flops:
                metrics["mfu"] = flops_per_step / dt / (peak_flops * n_devices)
            print(f"step {step:6d}/{cfg.num_steps} | " + " | ".join(f"{k} {v:.4g}" for k, v in metrics.items()))
            if wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    if long_eval and not stop.is_set():
        results = long_eval(params, eval_sharded)
        print(f"step {step:6d} | long context | " + " | ".join(f"{k} {v:.3f}" for k, v in results.items() if "/" not in k))
        if wandb:
            wandb.log({f"long/{k}": v for k, v in results.items()}, step=step)


if __name__ == "__main__":
    main()
