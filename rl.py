"""
RL with tool execution in the loop: on-policy, GRPO-style policy gradient. nanochat's chat_rl plus truncated importance
sampling, zero-variance filtering and a pass-rate curriculum.
Every step:
1. Sample n_prompts prompts from the task pool and run `group` rollouts of each as agent episodes (agent.py): the
   model's tool calls really run, in a fresh sandbox per episode, for several turns, within call and token budgets.
   Sampling is Sampling(): temperature 1, full softmax, so the engine's per-token logprobs are the behaviour policy's.
2. Reward = the task's checker, in [0, 1] (1 = solved; the web envs' rewards are graded, envs.py). Advantage
   A = r - mean(r of the group), divided by the group's std with --adv_norm=std (the final recipe). Groups whose rewards
   are all equal carry no signal and are dropped.
3. Loss = -sum over the model's sampled tokens (Sequence.mask; prompt and tool outputs are 0) of
   sg(min(pi_train / pi_rollout, is_cap)) * A * log pi_train, divided by the number of such tokens (token-level).
   No KL term. pi_rollout = the engine's logprobs from step 1 (token in, token out: the exact sampled ids).
   The objective is pluggable (OBJECTIVES: a function of trainer / rollout logprobs, loss mask, rewards and group ids),
   and the update only sees Trajectory records, so off-policy samples (tempered, a replay buffer) can be fed in too.
4. One optimizer step (optim.py: Muon + AdamW at lr_scale x the pretraining LRs, linear decay to 0, no weight decay).
   The engine holds the same params object and samples the next batch with the updated params.
Both sides run in fp32 (cfg.dtype=float32): the engine's and the trainer's logprobs then agree to ~1e-5, so the IS
weight is ~1 and the mismatch metrics (logged every step, from the loss's own forward pass) are a check.
Curriculum: a prompt whose group's mean reward is >= drop_pass_rate is retired (ScaleRL's no-positive-resampling).
Devices: all local GPUs. Rollouts: one engine per device, each running a contiguous part of the groups (agent.Agent).
Update: data parallel (params replicated, microbatch rows sharded, XLA all-reduces the gradients), as train.py.
--guide=terminal: a group without a pass gets the task's reference solution as one of its rollouts (off-policy, IS
weight 1), which unlocks task types the policy never samples a success for.
Environments (train splits disjoint from agent_eval.py's evals):
- terminal  terminal_tasks.py generators with training seeds (1000+), minus any task identical to an eval one (seed 0)
- gsm8k     GSM8K train with the python tool, scored like agent_eval.gsm8k (the last number of the answer)
- search, funcall, weather   the web / function tools (envs.py): open-domain QA with search (+ a tier answerable from
            the prompt), mock APIs declared in the system prompt, templated weather questions; graded rewards
- chat      requests that need no tool, with the tools listed (envs.py): no call and a real reply
- files     write a file with given contents (envs.py); nl2sh: single-command shell tasks (envs.py)
pass1 / passG count an episode as passed when its reward is >= PASS (= solved for the binary terminal / gsm8k checks).
Checkpoints are train.py's format (runs/<run_name>/ckpt.npz), so agent_eval.py / engine.py load them; reruns resume.
The final recipe is two stages, rl_web5 (600 steps from mt_tool20c) and rl_web7 (300 more with nl2sh): run.sh.
"""

import argparse
import concurrent.futures
import dataclasses
import functools
import json
import keyword
import os
import re
import signal
import threading
import time
import types
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

import agent_eval
import chat
import checkpoint
import envs
import model
import optim
import terminal_tasks
from agent import Agent, Episode
from engine import Engine, Sampling, bucket
from tokenizer import Tokenizer
from tools import Sandbox

LENGTH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]  # padded sequence lengths of the training microbatches
PASS = 0.5  # an episode with at least this reward counts as passed (metrics, envs.py's gate)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--init_from", default="mt_tool20c")
    p.add_argument("--tasks", default="terminal:1",
                   help="env:weight,... over terminal, gsm8k, search, funcall, weather, chat, files, nl2sh")
    p.add_argument("--tiers", default="1,2", help="terminal task tiers to train on")
    p.add_argument("--train_seeds", type=int, default=50, help="terminal tasks: seeds 1000 ... 1000 + this")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n_prompts", type=int, default=16, help="prompts per step")
    p.add_argument("--group", type=int, default=16, help="rollouts per prompt (G)")
    p.add_argument("--lr_scale", type=float, default=0.05, help="x the pretraining LRs, decayed linearly to 0")
    p.add_argument("--objective", default="grpo", help="a key of OBJECTIVES")
    p.add_argument("--adv_norm", default="mean", help="mean: A = r - group mean; std: also divided by the group's std")
    p.add_argument("--guide", default="",
                   help="envs (terminal, funcall, nl2sh, ...) whose groups without a pass get the prompt's reference "
                        "solution as one of their rollouts")
    p.add_argument("--is_cap", type=float, default=2.0, help="truncated importance sampling cap")
    p.add_argument("--drop_zero_variance", type=int, default=1, help="drop groups whose rewards are all equal")
    p.add_argument("--drop_pass_rate", type=float, default=0.9, help="retire prompts that pass this often")
    p.add_argument("--require_answer", type=int, default=1,
                   help="no reward for an episode that stopped without answering (call budget, length)")
    p.add_argument("--max_calls", type=int, default=8, help="tool calls per episode (agent_eval's default)")
    p.add_argument("--max_tokens", type=int, default=512, help="tokens per generation between tool calls")
    p.add_argument("--max_len", type=int, default=2048, help="tokens per episode (the engine's cache length)")
    p.add_argument("--n_slots", type=int, default=256)
    p.add_argument("--decode_steps", type=int, default=16)
    p.add_argument("--workers", type=int, default=16, help="threads running tool calls")
    p.add_argument("--micro_tokens", type=int, default=16384, help="tokens per training microbatch, per device")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_suites", default="terminal",
                   help="held-out agent_eval suites (greedy), e.g. terminal,gsm8k,qa,bfcl,weather")
    p.add_argument("--eval_limit", type=int, default=300, help="items per eval suite except terminal (qa: per dataset)")
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--wandb_project", default="femto_rl")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------------
# environments

@dataclasses.dataclass
class Prompt:
    id: str
    task: str  # for per-task metrics: the generator's name, or "gsm8k"
    instruction: str
    files: dict  # the sandbox's starting files
    check: Callable  # (agent.Episode) -> reward in [0, 1] (or solved?)
    system: str = ""  # system prompt (the web envs list the tools in it)
    functions: dict = None  # declared functions {name: handler(args) -> result} the episode's sandbox serves
    solution: str = ""  # reference solution for --guide: a tool call's code (terminal tasks: a bash command) ...
    solution_tool: str = "bash"
    solution_answer: str = ""  # ... and the final answer after its output (default: the output)

    def turns(self):
        return ([chat.system(self.system)] if self.system else []) + [chat.user(self.instruction)]


def task_key(t):
    return t.instruction, tuple(sorted(t.files.items()))


def terminal_pool(tiers, n_seeds):
    """Terminal tasks of the given tiers from training seeds, without duplicates and without any eval (seed 0) task."""
    seen = {task_key(t) for t in terminal_tasks.tasks(0)}
    pool = []
    for seed in range(1000, 1000 + n_seeds):
        for i, t in enumerate(terminal_tasks.tasks(seed)):
            if t.tier in tiers and task_key(t) not in seen:
                seen.add(task_key(t))
                pool.append(Prompt(f"terminal/{seed}/{i}", t.name, t.instruction, t.files, t.check, solution=t.solution))
    return pool


def gsm8k_check(gold):
    def check(ep):
        pred = agent_eval.last_number(terminal_tasks.answer_text(ep))
        return pred is not None and abs(pred - gold) < 1e-4 * max(1, abs(gold))
    return check


def gsm8k_pool():
    """GSM8K train with the python tool, checked as agent_eval.gsm8k checks the test split; the reference solution
    (--guide) is the program of the human solution's calculator steps, then "The answer is <gold>."."""
    import pyarrow.parquet as pq
    rows = pq.read_table(agent_eval.hf_file("openai/gsm8k", "main/train-00000-of-00001.parquet")).to_pylist()
    return [Prompt(f"gsm8k/{i}", "gsm8k", r["question"].strip(), {},
                   gsm8k_check(agent_eval.to_number(r["answer"].split("####")[-1])), solution=gsm8k_program(r["answer"]),
                   solution_tool="python", solution_answer=f"The answer is {r['answer'].split('####')[-1].strip()}.")
            for i, r in enumerate(rows)]


def gsm8k_program(solution):
    """A GSM8K solution's calculator annotations (<<48/2=24>>) as a TinyGSM-style program, for --guide: one variable
    per step, named after the word that follows the step's result in the text, earlier results replaced by their
    variables. "" if there are no annotations (a wrong program is caught by the check)."""
    lines, names, by_value = [], set(), {}
    for m in re.finditer(r"<<([^=<>]+)=([^<>]*)>>", solution):
        expr = re.sub(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", lambda n: by_value.get(n.group(), n.group()), m.group(1))
        word = re.match(r"[^\sA-Za-z]*\s*([A-Za-z]{3,})", solution[m.end():].lstrip("0123456789.,$%"))
        name = word.group(1).lower() if word and not keyword.iskeyword(word.group(1).lower()) else "step"
        name = name if name not in names and name != "result" else f"{name}_{len(lines) + 1}"
        names.add(name)
        lines.append(f"    {name} = {expr.strip()}\n")
        by_value[m.group(2).strip()] = name
    if not lines:
        return ""
    return (f"def simple_math_problem() -> int:\n{''.join(lines)}    result = {name}\n    return result\n\n\n"
            "print(simple_math_problem())")


def reward(prompt, ep, require_answer):
    """The task's check as a float. With require_answer, an episode that stopped without answering (call budget,
    length) gets 0, even if its files are right (otherwise repeating a command until the call budget runs out pays)."""
    try:
        return float(prompt.check(ep)) if ep.finish == "answer" or not require_answer else 0.0
    except Exception:
        return 0.0


def reference_trajectory(agent, prompt, group, args):
    """The prompt's reference solution as a trajectory (--guide): one tool call, then solution_answer, or else the
    call's output, as the answer ("Done." if it prints nothing); without a solution, solution_answer alone (no call).
    Its own tokens are off-policy, logprob nan (the loss then uses IS weight 1). None if it doesn't get the full
    reward."""
    with Sandbox(prompt.files) as box:
        output, code = box.run(prompt.solution_tool, prompt.solution) if prompt.solution else ("", None)
        answer = prompt.solution_answer or output.strip() or "Done."
        call = agent.encode(f"<|tool_start|>{prompt.solution_tool}\n{chat.defuse(prompt.solution)}<|tool_end|>") \
            if prompt.solution else []
        end = agent.encode(chat.defuse(answer) + "<|assistant_end|>")
        out = agent.output_tokens(output, code) if prompt.solution else []
        ep = types.SimpleNamespace(finish="answer", answer=answer, last_output=output, sandbox=box,
                                   calls=[dict(tool=prompt.solution_tool, code=prompt.solution, output=output,
                                               exit_code=code)] if prompt.solution else [])
        r = reward(prompt, ep, args.require_answer)
        if prompt.solution_answer and prompt.solution:  # the given answer must be what the call computed
            r = min(r, reward(prompt, types.SimpleNamespace(**dict(vars(ep), answer=output.strip())), 1))
    tokens = agent.prompt(prompt.turns())
    logprobs = [None] * len(tokens) + [float("nan")] * len(call) + [None] * len(out) + [float("nan")] * len(end)
    return Trajectory(tokens + call + out + end, logprobs, r, group) if r >= 1 else None


# ---------------------------------------------------------------------------------------------------------------------
# trajectories and objectives

@dataclasses.dataclass
class Trajectory:
    """A training sequence, wherever it was sampled (this step's rollouts here; tempered sampling or a replay buffer
    would produce the same records). Token in, token out: the exact ids the engine produced."""
    tokens: list
    logprobs: list  # the behaviour policy's log probability of each token it sampled; None for given tokens
    reward: float
    group: int  # trajectories of the same prompt share a group (group baselines, per-group logZ, ...)

    @property
    def mask(self):
        return [int(lp is not None) for lp in self.logprobs]


def group_mean(x, groups, n_groups):
    total = jax.ops.segment_sum(x, groups, n_groups)
    count = jax.ops.segment_sum(jnp.ones_like(x), groups, n_groups)
    return (total / jnp.maximum(count, 1))[groups]


def grpo(logp, rollout_logp, mask, rewards, groups, n_groups, n_tokens, is_cap=2.0, adv_norm="mean"):
    """GRPO as nanochat (+ TIS): A = r - mean(r of the group); token-level policy gradient
    -sum sg(min(pi / pi_rollout, is_cap)) * A * log pi over the sampled tokens / n_tokens, no KL term.
    adv_norm="std": A / std(r of the group) (the original GRPO; the final recipe)."""
    mean = group_mean(rewards, groups, n_groups)
    adv = rewards - mean
    if adv_norm == "std":
        adv = adv / (jnp.sqrt(group_mean((rewards - mean)**2, groups, n_groups)) + 1e-4)
    weight = jnp.minimum(jnp.exp(jax.lax.stop_gradient(logp) - rollout_logp), is_cap)
    return -jnp.sum(weight * adv[:, None] * mask * logp) / n_tokens


# An objective maps (trainer logprobs (R, T), rollout logprobs (R, T), loss mask (R, T), rewards (R,), group ids (R,),
# number of group ids, number of loss tokens in the whole batch) to a scalar loss; microbatches hold whole groups, so
# group statistics are exact. E.g. trajectory balance would be (logZ + sum(mask * logp, 1) - log R)^2, VarGrad the same
# with logZ = group_mean(log R - sum(mask * logp, 1)) (stop-gradient).
OBJECTIVES = {"grpo": grpo}


def loss_and_stats(params, batch, n_tokens, cfg, objective, is_cap):
    """The objective on one microbatch, plus sums for the logged metrics from the same forward pass: on-policy, these
    params sampled the batch, so pi_train / pi_rollout measures the trainer-engine mismatch."""
    inputs, targets, mask, rollout_logp, rewards, groups = batch
    logits = model.forward(params, inputs, cfg)
    logz = jax.nn.logsumexp(logits, axis=-1)
    logp = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0] - logz
    rollout_logp = jnp.where(jnp.isnan(rollout_logp), jax.lax.stop_gradient(logp), rollout_logp)  # off-policy: weight 1
    loss = objective(logp, rollout_logp, mask, rewards, groups, groups.shape[0] + 1, n_tokens)
    log_ratio = jax.lax.stop_gradient(logp) - rollout_logp
    ratio = jnp.exp(log_ratio)
    sg_logits = jax.lax.stop_gradient(logits)
    entropy = jax.lax.stop_gradient(logz) - jnp.sum(jax.nn.softmax(sg_logits, axis=-1) * sg_logits, axis=-1)
    on = mask > 0
    stats = dict(tokens=jnp.sum(mask), kl=jnp.sum(mask * (ratio - 1 - log_ratio)),
                 abs_logp_diff=jnp.sum(mask * jnp.abs(log_ratio)),
                 max_abs_logp_diff=jnp.max(jnp.where(on, jnp.abs(log_ratio), 0.0)),
                 is_weight_max=jnp.max(jnp.where(on, ratio, 0.0)), is_clipped=jnp.sum(mask * (ratio > is_cap)),
                 outside_05_2=jnp.sum(mask * ((ratio < 0.5) | (ratio > 2))), entropy=jnp.sum(mask * entropy))
    return loss, stats


@functools.partial(jax.jit, static_argnames=("cfg", "objective", "is_cap"), donate_argnames="grads")
def accumulate(params, grads, batch, n_tokens, cfg, objective, is_cap):
    with model.precision(cfg):
        (loss, stats), g = jax.value_and_grad(loss_and_stats, has_aux=True)(params, batch, n_tokens, cfg, objective,
                                                                            is_cap)
    return jax.tree.map(jnp.add, grads, g), loss, stats


@functools.partial(jax.jit, static_argnames="cfg", donate_argnames=("params", "opt_state"))
def apply_update(params, opt_state, grads, step, cfg):
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    params, opt_state = optim.update(params, grads, opt_state, step, cfg, weight_decay=False)
    return params, opt_state, grad_norm


def microbatches(trajectories, micro_tokens, multiple=1):
    """Whole groups, longest first, packed into microbatches of ~micro_tokens tokens (at least one group) padded to a
    bucketed length and to a multiple of `multiple` rows (the devices): (inputs, targets, mask, rollout logprobs,
    rewards, group ids) arrays. Group ids are renumbered 0, 1, ... per microbatch; padding rows get their own id
    (= rows) and mask 0."""
    by_group = {}
    for t in trajectories:
        by_group.setdefault(t.group, []).append(t)
    groups = sorted(by_group.values(), key=lambda g: -max(len(t.tokens) for t in g))
    largest = max(len(g) for g in groups)
    while groups:
        length = bucket(max(len(t.tokens) for t in groups[0]) - 1, LENGTH_BUCKETS)
        rows = -(-max(micro_tokens // length, largest) // multiple) * multiple
        chunk = []
        while groups and len(chunk) + len(groups[0]) <= rows:
            g = groups.pop(0)
            chunk += [(t, g[0].group) for t in g]
        ids = {gid: i for i, gid in enumerate(dict.fromkeys(gid for _, gid in chunk))}
        inputs, targets = np.zeros((rows, length), np.int32), np.zeros((rows, length), np.int32)
        mask, old = np.zeros((rows, length), np.float32), np.zeros((rows, length), np.float32)
        rewards, group_ids = np.zeros(rows, np.float32), np.full(rows, rows, np.int32)
        for r, (t, gid) in enumerate(chunk):
            n = len(t.tokens) - 1
            inputs[r, :n], targets[r, :n], mask[r, :n] = t.tokens[:-1], t.tokens[1:], t.mask[1:]
            old[r, :n] = [lp or 0.0 for lp in t.logprobs[1:]]
            rewards[r], group_ids[r] = t.reward, ids[gid]
        yield inputs, targets, mask, old, rewards, group_ids


def train_step(params, opt_state, trajectories, step, cfg, objective, args, sharding=None):
    """One optimizer step on the trajectories (on- or off-policy alike), microbatch rows split between the devices of
    sharding (None: the default device). Returns params, opt_state and metrics."""
    n_tokens = float(sum(sum(t.mask[1:]) for t in trajectories))
    grads = jax.tree.map(jnp.zeros_like, params)
    losses, stats = [], []
    n_devices = len(sharding.device_set) if sharding else 1
    for batch in microbatches(trajectories, args.micro_tokens * n_devices, n_devices):
        batch = jax.device_put(batch, sharding) if sharding else batch
        grads, loss, st = accumulate(params, grads, batch, n_tokens, cfg=cfg, objective=objective, is_cap=args.is_cap)
        losses.append(loss)
        stats.append(st)
    params, opt_state, grad_norm = apply_update(params, opt_state, grads, step, cfg=cfg)
    stats = jax.device_get(stats)
    total = lambda k: sum(float(s[k]) for s in stats)
    n = total("tokens")
    metrics = {"train/loss": float(sum(jax.device_get(losses))), "train/grad_norm": float(grad_norm),
               "train/lrm": float(optim.lr_multiplier(step, cfg)), "train/tokens": n,
               "train/entropy": total("entropy") / n, "mismatch/kl": total("kl") / n,
               "mismatch/mean_abs_logp_diff": total("abs_logp_diff") / n,
               "mismatch/max_abs_logp_diff": max(float(s["max_abs_logp_diff"]) for s in stats),
               "mismatch/is_weight_max": max(float(s["is_weight_max"]) for s in stats),
               "mismatch/is_clipped_frac": total("is_clipped") / n,
               "mismatch/outside_05_2_frac": total("outside_05_2") / n}
    return params, opt_state, metrics


def drop_zero_variance(trajectories):
    """Groups whose rewards are all equal: no GRPO gradient (A = 0), so they are dropped (also from the token count)."""
    rewards = {}
    for t in trajectories:
        rewards.setdefault(t.group, []).append(t.reward)
    return [t for t in trajectories if np.std(rewards[t.group]) > 0]


# ---------------------------------------------------------------------------------------------------------------------
# rollouts and metrics

def rollout(agent, prompts, args, timing=None):
    """args.group episodes per prompt (sampled by the agent's engine with the current params). Returns, per prompt,
    its episodes and their rewards, and the trajectories to train on (group = the prompt's index)."""
    timing = {} if timing is None else timing
    t0 = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(args.workers)  # sandboxes and checks run commands: in parallel
    group_prompts = [p for p in prompts for _ in range(args.group)]
    boxes = list(pool.map(lambda p: Sandbox(p.files, functions=p.functions), group_prompts))
    tokens = {p.id: agent.prompt(p.turns()) for p in prompts}
    episodes = [Episode(tokens[p.id], box, args.max_calls, args.max_tokens) for p, box in zip(group_prompts, boxes)]
    t1 = time.time()
    agent.run(episodes)
    t2 = time.time()
    results = list(pool.map(lambda p, ep: reward(p, ep, args.require_answer), group_prompts, episodes))
    list(pool.map(Sandbox.close, boxes))
    pool.shutdown()
    timing.update({"time/rollout_sandboxes": t1 - t0 + time.time() - t2, "time/rollout_agent": t2 - t1})
    # a tool output can run past max_len (the episode then stops); that tail was never model input, so it is cropped
    # (otherwise a 4096-long microbatch of a whole group can run out of memory)
    trajectories = [Trajectory(ep.seq.tokens[:args.max_len], ep.seq.logprobs[:args.max_len], r, i // args.group)
                    for i, (ep, r) in enumerate(zip(episodes, results))]
    g = args.group
    timing["pool/guided"] = 0
    for i, p in enumerate(prompts):  # --guide: the reference solution replaces the group's last rollout
        if p.id.split("/")[0] in args.guide.split(",") and (p.solution or p.solution_answer) and \
                max(results[i * g:(i + 1) * g]) < PASS:
            ref = reference_trajectory(agent, p, i, args)
            if ref is not None:
                trajectories[(i + 1) * g - 1] = ref
                timing["pool/guided"] += 1
    return ([episodes[i:i + g] for i in range(0, len(episodes), g)],
            [results[i:i + g] for i in range(0, len(results), g)], trajectories)


def rollout_metrics(prompts, episodes, rewards, trajectories):
    eps = [ep for g in episodes for ep in g]
    solved = [[r >= PASS for r in g] for g in rewards]
    m = {f"format/{k}": v for k, v in agent_eval.health(eps).items()}
    m["episode/tokens"] = float(np.mean([len(ep.seq.tokens) for ep in eps]))
    m["episode/gen_tokens"] = float(np.mean([sum(ep.seq.mask[ep.start:]) for ep in eps]))
    m["reward/mean"] = float(np.mean([t.reward for t in trajectories]))
    m["pass1/all"] = float(np.mean(solved))
    m["passG/all"] = float(np.mean([any(s) for s in solved]))
    for task in sorted({p.task for p in prompts}):
        s = [s for p, s in zip(prompts, solved) if p.task == task]
        m[f"pass1/{task}"], m[f"passG/{task}"] = float(np.mean(s)), float(np.mean([any(x) for x in s]))
        m[f"reward/{task}"] = float(np.mean([r for p, r in zip(prompts, rewards) if p.task == task]))
        m[f"full/{task}"] = float(np.mean([r >= 1 for p, rs in zip(prompts, rewards) if p.task == task for r in rs]))
        m[f"calls/{task}"] = float(np.mean([len(ep.calls) for p, g in zip(prompts, episodes) if p.task == task
                                            for ep in g]))
    return m


def evaluate(engines, tok, args):
    """Held-out agent_eval suites, greedy, on the current params (the eval seed of the terminal tasks)."""
    agent = Agent(engines, tok, Sampling(temperature=0), workers=args.workers)
    out = {}
    for suite in args.eval_suites.split(","):
        limit = 0 if suite == "terminal" else args.eval_limit
        ns = types.SimpleNamespace(seed=0, limit=limit, tiers="", n_samples=1, max_calls=8, max_tokens=512)
        results, episodes, _ = agent_eval.SUITES[suite](agent, ns)
        out.update({f"eval/{suite}/{k}": v for k, v in results.items()})
        out.update({f"eval/{suite}/format/{k}": v for k, v in agent_eval.health(episodes).items()})
    return out


def on_device(tree, device):
    """The copy on device of each (replicated) array of tree, as a single-device array. (A shard's .data can carry the
    whole array's sharding (one device, after a jitted update) or not (before): the engine's jitted functions would
    then compile every shape again after the first update, ~5 s each in fp32 without a cache hit.)"""
    one = jax.sharding.SingleDeviceSharding(device)
    return jax.tree.map(lambda x: jax.device_put(next(s.data for s in x.addressable_shards if s.device == device), one),
                        tree)


# ---------------------------------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    print(vars(args))
    params, base_cfg, _ = checkpoint.load_model(checkpoint.model_path(args.init_from))
    # fp32 on both sides; LRs = lr_scale x the pretraining LRs (batch scaling as in pretraining), linear decay to 0.
    # The optimizer state is fresh, but init_optimizer stays the base run's: from mt_tool20c (True), AdamW then skips its
    # bias correction (optim.update), which damps the first AdamW steps; the final recipe ran that way.
    cfg = dataclasses.replace(base_cfg, dtype="float32", lr_scale=args.lr_scale, warmup_steps=1, warmdown_ratio=1.0,
                              final_lr_frac=0.0, num_tokens=args.steps * base_cfg.total_batch_tokens,
                              run_name=args.run_name, init_from=args.init_from)
    tok = Tokenizer.load(cfg.tokenizer)
    opt_state = optim.init(params)
    ckpt_path = checkpoint.run_path(args.run_name)
    start_step, retired = 0, set()
    if os.path.exists(ckpt_path):
        (params, opt_state), meta = checkpoint.load(ckpt_path, (params, opt_state))
        start_step, retired = meta["step"], set(meta["retired"])
        if start_step == args.steps:
            print(f"{ckpt_path} is finished (step {start_step})")
            return
        print(f"resumed from {ckpt_path} at step {start_step}")
    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    rows_sharded = NamedSharding(mesh, P("data"))  # microbatch rows (every batch array's first axis)
    params, opt_state = jax.device_put((params, opt_state), NamedSharding(mesh, P()))

    weights = {env: float(w) for env, w in (x.split(":") for x in args.tasks.split(","))}
    tiers = {int(t) for t in args.tiers.split(",")}
    builders = {"terminal": lambda: terminal_pool(tiers, args.train_seeds),
                "gsm8k": gsm8k_pool, "search": envs.search_pool, "funcall": envs.funcall_pool,
                "weather": envs.weather_pool, "chat": envs.chat_pool, "files": envs.files_pool,
                "nl2sh": envs.nl2sh_pool}
    pools = {env: builders[env]() for env, w in weights.items() if w > 0}
    print({env: len(pool) for env, pool in pools.items()}, "prompts;", len(retired), "retired")
    if "search" in pools or "qa" in args.eval_suites:
        envs.load_search_index()

    engines = [Engine(on_device(params, d), cfg, n_slots=args.n_slots, max_len=args.max_len,
                      decode_steps=args.decode_steps, row_buckets=[4**i for i in range(8)], device=d)
               for d in devices]  # (few row buckets: few shapes to compile, fp32 compilations are slow)
    agent = Agent(engines, tok, Sampling(), workers=args.workers)  # temperature 1, full softmax
    print(f"{len(devices)} x {devices[0].device_kind}")
    objective = functools.partial(OBJECTIVES[args.objective], is_cap=args.is_cap, adv_norm=args.adv_norm)

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.out_dir, "rl_log.jsonl"), "a")
    samples_file = open(os.path.join(cfg.out_dir, "rl_samples.jsonl"), "a")
    wandb = None
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, id=args.run_name, resume="allow",
                   config=dict(vars(args), model=dataclasses.asdict(cfg)))

    def log(metrics, step):
        log_file.write(json.dumps(dict(step=step, **metrics)) + "\n")
        log_file.flush()
        if wandb:
            wandb.log(metrics, step=step)

    def save(step):
        meta = dict(step=step, config=dataclasses.asdict(cfg), rl=vars(args), retired=sorted(retired))
        checkpoint.save(ckpt_path, (params, opt_state), meta)
        print(f"saved checkpoint at step {step}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    for step in range(start_step, args.steps + 1):
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps) and not stop.is_set():
            t0 = time.time()
            ev = evaluate(engines, tok, args)
            ev["time/eval"] = time.time() - t0
            print(f"step {step:4d} | eval | " + " | ".join(f"{k} {v:.3f}" for k, v in ev.items()
                                                           if "/task/" not in k and "format" not in k), flush=True)
            log(ev, step)
        if step == args.steps or stop.is_set() or (step > start_step and step % args.save_every == 0):
            save(step)
        if step == args.steps or stop.is_set():
            break

        # 1. rollouts with the current params
        t0 = time.time()
        rng = np.random.default_rng([args.seed, step])
        for i, engine in enumerate(engines):
            key = jax.random.key(args.seed * 1_000_003 + step)
            engine.key = jax.random.fold_in(key, i) if i else key
        drawn = rng.choice(list(pools), size=args.n_prompts, p=np.array([weights[e] for e in pools]) / sum(
            weights[e] for e in pools))
        prompts = []
        for env in pools:
            k = int(np.sum(drawn == env))
            active = [p for p in pools[env] if p.id not in retired]
            active = active if len(active) >= k else pools[env]  # (a small pool can run out)
            prompts += [active[i] for i in rng.choice(len(active), size=k, replace=False)]
        timing = {}
        episodes, rewards, trajectories = rollout(agent, prompts, args, timing)
        t_rollout = time.time() - t0
        metrics = rollout_metrics(prompts, episodes, rewards, trajectories) | timing
        metrics["time/rollout"] = t_rollout
        metrics["time/rollout_gen_tok_per_s"] = metrics["episode/gen_tokens"] * len(trajectories) / t_rollout
        for p, r in zip(prompts, rewards):  # curriculum: retire prompts that are (nearly) always solved
            if np.mean(r) >= args.drop_pass_rate:
                retired.add(p.id)
        metrics["pool/retired"] = len(retired)

        # 2.-4. objective on the trajectories (zero-variance groups dropped), one optimizer step
        t0 = time.time()
        if args.drop_zero_variance:
            trajectories = drop_zero_variance(trajectories)
        metrics["zero_var_frac"] = 1 - len(trajectories) / (len(prompts) * args.group)
        if trajectories:
            params, opt_state, train_metrics = train_step(params, opt_state, trajectories, step, cfg, objective, args,
                                                          rows_sharded)
            for engine in engines:  # the next rollouts use the updated params (the old buffers were donated)
                engine.params = on_device(params, engine.device)
            metrics.update(train_metrics)
        metrics["time/train"] = time.time() - t0
        metrics["time/step"] = metrics["time/train"] + t_rollout
        print(f"step {step:4d} | reward {metrics['reward/mean']:.3f} | pass1 {metrics['pass1/all']:.3f} | passG "
              f"{metrics['passG/all']:.3f} | zero-var {metrics['zero_var_frac']:.2f} | gen tok {metrics['episode/gen_tokens']:.0f}"
              f" | calls {metrics['format/calls']:.2f} | malformed {metrics['format/malformed_calls']:.3f} | runaway "
              f"{metrics['format/runaway']:.3f} | kl {metrics.get('mismatch/kl', 0):.2e} | |dlogp| "
              f"{metrics.get('mismatch/mean_abs_logp_diff', 0):.2e} (max {metrics.get('mismatch/max_abs_logp_diff', 0):.2e})"
              f" | entropy {metrics.get('train/entropy', 0):.3f} | rollout {t_rollout:.1f}s train {metrics['time/train']:.1f}s",
              flush=True)
        log(metrics, step)
        if step % 10 == 0:  # a few transcripts, to watch the format
            for p, eps, s in list(zip(prompts, episodes, rewards))[:2]:
                replies = [[ep.reply, ep.finish, ok] for ep, ok in zip(eps[:4], s)]
                samples_file.write(json.dumps(dict(step=step, task=p.task, instruction=p.instruction,
                                                   replies=replies)) + "\n")
            samples_file.flush()


if __name__ == "__main__":
    main()
