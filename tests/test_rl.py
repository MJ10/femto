"""rl.py: the loss is REINFORCE on the sampled tokens (whole groups per microbatch) with IS weight ~1 in fp32, its
mismatch metrics are ~0 on engine rollouts, the update reaches the engine, and the training tasks are disjoint from the
eval tasks; the web envs' rewards pass oracles and resist the obvious hacks."""

import dataclasses
import json
import types

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

import engine
import envs
import model
import optim
import rl
import terminal_tasks
import weather
from test_engine import CONFIGS, params_for
from tools import Sandbox

CFG = dataclasses.replace(CONFIGS[0], dtype="float32", num_tokens=10 * 2**18, total_batch_tokens=2**18, lr_scale=0.05,
                          warmup_steps=1, warmdown_ratio=1.0, final_lr_frac=0.0)


def rollout_samples(params):
    eng = engine.Engine(params, CFG, n_slots=4, max_len=128, decode_steps=8)
    prompts = [[1, 2, 3], list(range(10, 40)), [7] * 9, [5, 9]]
    seqs = eng.generate(prompts, max_tokens=50)
    for s in seqs:  # as if the middle were a tool output: given tokens, mask 0
        s.logprobs[len(s.tokens) // 2:len(s.tokens) // 2 + 5] = [None] * 5
    rewards = [1.0, 0.0, 0.5, 0.0]
    return eng, seqs, [rl.Trajectory(s.tokens, s.logprobs, r, i // 2) for i, (s, r) in enumerate(zip(seqs, rewards))]


def test_loss_is_reinforce_and_on_policy():
    params = params_for(CFG)
    _, seqs, samples = rollout_samples(params)
    n_tokens = float(sum(sum(t.mask[1:]) for t in samples))
    assert n_tokens == 4 * 45
    batches = list(rl.microbatches(samples, micro_tokens=512))  # 2 rows of 256 tokens: one group each
    assert len(batches) == 2 and sum(int(b[2].sum()) for b in batches) == n_tokens
    assert [b[4].tolist() for b in batches] == [[1.0, 0.0], [0.5, 0.0]] and [b[5].tolist() for b in batches] == [[0, 0]] * 2
    grads = jax.tree.map(jnp.zeros_like, params)
    for b in batches:
        grads, loss, stats = rl.accumulate(params, grads, b, n_tokens, cfg=CFG, objective=rl.grpo, is_cap=2.0)

    def reinforce(params):  # -sum A * log p over the sampled tokens / their number, A = r - mean(r of the group)
        total = 0.0
        for (inputs, targets, mask, *_), adv in zip(batches, [[0.5, -0.5], [0.25, -0.25]]):
            logp = -model.token_losses(params, inputs, targets, CFG)
            total -= jnp.sum(jnp.array(adv)[:, None] * mask * logp)
        return total / n_tokens
    with model.precision(CFG):
        ref = jax.grad(reinforce)(params)
    diff = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(ref)))
    scale = max(float(jnp.abs(b).max()) for b in jax.tree.leaves(ref))
    assert diff < 1e-3 * scale, (diff, scale)


def test_mismatch_metrics_zero_and_update_reaches_engine():
    params = params_for(CFG)
    eng = engine.Engine(params, CFG, n_slots=4, max_len=128, decode_steps=8)
    seqs = eng.generate([[1, 2, 3], list(range(10, 40))], max_tokens=60)
    samples = [rl.Trajectory(s.tokens, s.logprobs, r, 0) for s, r in zip(seqs, [1.0, 0.0])]
    n_tokens = float(sum(sum(t.mask[1:]) for t in samples))
    grads = jax.tree.map(jnp.zeros_like, params)
    for b in rl.microbatches(samples, 1024):
        grads, loss, stats = rl.accumulate(params, grads, b, n_tokens, cfg=CFG, objective=rl.grpo, is_cap=2.0)
    assert float(stats["tokens"]) == n_tokens == 120
    assert float(stats["max_abs_logp_diff"]) < 1e-4 and float(stats["kl"]) / n_tokens < 1e-8
    assert abs(float(stats["is_weight_max"]) - 1) < 1e-4 and float(stats["is_clipped"]) == 0
    before = jax.device_get(params["lm_head"])
    params, opt_state, grad_norm = rl.apply_update(params, optim.init(params), grads, 0, cfg=CFG)
    assert float(grad_norm) > 0 and not np.allclose(before, params["lm_head"])
    eng.params = params
    seqs = eng.generate([[1, 2, 3]], max_tokens=20)
    assert engine.mismatch(params, CFG, seqs)["max_abs_logp_diff"] < 1e-4  # sampled by the updated params


def test_devices_split_rollouts_and_update():
    """Two devices (tests/conftest.py): an engine per device generates what the default engine does, and the
    data-parallel update (params replicated, microbatch rows sharded) is the single-device one."""
    devices = jax.local_devices()
    assert len(devices) == 2
    params = params_for(CFG)
    mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    replicated = jax.device_put(params, NamedSharding(mesh, P()))
    prompts = [[1, 2, 3], list(range(10, 40))]
    greedy = engine.Sampling(temperature=0)
    ref = engine.Engine(params, CFG, n_slots=2, max_len=128).generate(prompts, max_tokens=30, sampling=greedy)
    for d in devices:
        eng = engine.Engine(rl.on_device(replicated, d), CFG, n_slots=2, max_len=128, device=d)
        assert [s.tokens for s in eng.generate(prompts, max_tokens=30, sampling=greedy)] == [s.tokens for s in ref]
        assert eng.cache[0][0].devices() == {d}

    _, _, samples = rollout_samples(params)
    args = types.SimpleNamespace(micro_tokens=256, is_cap=2.0)
    one, _, m1 = rl.train_step(params, optim.init(params), samples, 0, CFG, rl.grpo, args)
    replicated = jax.device_put(params_for(CFG), NamedSharding(mesh, P()))  # (the update donated params)
    two, _, m2 = rl.train_step(replicated, optim.init(replicated), samples, 0, CFG, rl.grpo, args,
                               NamedSharding(mesh, P("data")))
    assert abs(m1["train/loss"] - m2["train/loss"]) < 1e-6 and m1["train/tokens"] == m2["train/tokens"]
    diff = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree.leaves(one), jax.tree.leaves(two)))
    assert diff < 5e-5, diff  # gradient all-reduce order differs by CPU; Muon's step amplifies it to ~1e-5


def test_training_tasks_disjoint_from_eval():
    pool = rl.terminal_pool({1, 2}, n_seeds=2)
    eval_keys = {rl.task_key(t) for t in terminal_tasks.tasks(0)}
    assert pool and all(rl.task_key(p) not in eval_keys for p in pool)
    assert len({rl.task_key(p) for p in pool}) == len(pool)
    assert {p.task for p in pool} == {t.name for t in terminal_tasks.tasks(0) if t.tier in (1, 2)}


def test_web_env_rewards():
    """The funcall / weather rewards are 1 for an oracle (the gold call really run, its value stated) and low for the
    obvious hacks (no call, pasted JSON, a retry)."""
    def episode(answer, calls=()):
        return types.SimpleNamespace(answer=answer, calls=list(calls), finish="answer")
    for i in range(60):
        task, item = envs.funcall_item(envs.SEED0, i)
        if task.startswith("unserved"):
            declared = dict(tool=next(iter(item.handlers)), code="{}", output="{}")
            assert item.check(episode("I can't do that.")) == 1 and item.check(episode("ok", [declared])) == 0
            assert item.check(episode("ok", [dict(tool=None, code="", output="error")])) == 0  # (an invented tool)
            # a search is fine when the prompt lists the built-ins, not when it lists only the functions (as BFCL)
            assert item.check(episode("ok", [dict(tool="search", code="x", output="")])) == (task == "unserved")
            continue
        if task == "none":
            assert item.check(episode(f"{item.value}")) == 1 and item.check(episode("12345")) == 0.5
            declared = dict(tool=next(iter(item.handlers)), code="{}", output="{}")
            assert item.check(episode(f"{item.value}", [dict(tool="python", code="", output="")])) == 1
            assert item.check(episode(f"{item.value}", [declared])) == 0
            continue
        name, allowed = next(iter(item.gold[0].items()))
        code = json.dumps({p: v[0] for p, v in allowed.items()})
        with Sandbox(functions=item.handlers) as box:
            output, _ = box.run(name, code)
        call = dict(tool=name, code=code, output=output)
        assert item.check(episode(f"It is {item.value}.", [call])) == 1, (task, output)
        assert item.check(episode(f"It is {item.value}.")) == 0
        assert item.check(episode(output, [call])) == 0.5 and item.check(episode(f"{item.value}", [call] * 2)) == 0
    cities = sorted(weather.CITIES)[:30]  # (any cities: the RL ones need tool_sources.rl_split)
    for i in range(40):
        item = envs.weather_item(cities, envs.SEED0, i)
        calls = [dict(tool="weather", code=place) for place in item.places]
        assert item.check(episode(item.oracle, calls)) == 1 and item.check(episode("", calls)) == 0, item
        assert item.check(episode(item.oracle, calls[:1] if len(calls) > 1 else [])) == 0  # (guessed)


def test_guide_reference_trajectories():
    """--guide: every terminal task's reference solution, and a GSM8K annotation program, as a trajectory, gets the full
    reward (a program that computes the wrong number doesn't, even with the gold answer); its own tokens are
    trained on (mask 1, logprob nan: IS weight 1) and the tool output is not."""
    from agent import Agent
    from tokenizer import SPECIAL_TOKENS, Tokenizer
    tok = Tokenizer.train(["ls wc -l sort uniq awk grep mv cp mkdir echo cat python3 files"] * 3,
                          vocab_size=len(SPECIAL_TOKENS) + 300)
    agent = Agent(None, tok)
    args = types.SimpleNamespace(require_answer=1)
    solution = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nShe sold 48+24 = <<48+24=72>>72 clips in all.\n#### 72"
    code = rl.gsm8k_program(solution)
    assert "clips = 48/2" in code and "clips_2 = 48+clips" in code
    gsm8k = [rl.Prompt("gsm8k/0", "gsm8k", "How many?", {}, rl.gsm8k_check(gold), solution=code, solution_tool="python",
                       solution_answer=f"The answer is {gold}.") for gold in (72, 73)]
    assert rl.reference_trajectory(agent, gsm8k[0], 0, args).reward == 1
    assert rl.reference_trajectory(agent, gsm8k[1], 0, args) is None  # (the program prints 72)
    for p in rl.terminal_pool({1, 2, 3, 4}, n_seeds=1)[::5]:
        t = rl.reference_trajectory(agent, p, 3, args)
        assert t is not None and t.reward == 1 and t.group == 3, p.task
        trained = [x for x, m in zip(t.tokens, t.mask) if m]
        assert tok.decode(trained).startswith("<|tool_start|>bash\n") and tok.decode(trained).endswith("<|assistant_end|>")
        assert all(np.isnan(lp) for lp in t.logprobs if lp is not None)
