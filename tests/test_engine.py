"""The engine's incremental forward pass (KV cache, chunked prefill, decoding) gives model.forward's logits, and its
generations and log probabilities match the training forward pass."""

import dataclasses

import jax
import numpy as np

import engine
import model
from config import Config

# local layers with a 16-token window, global NoPE layers with (or without) the log-length temperature from position 32
CFG = Config(vocab_size=256, n_layer=4, d_model=128, head_dim=64, seq_len=128, local_window=16, global_attn_temp=0.13,
             global_attn_temp_len=32)
CONFIGS = [CFG, dataclasses.replace(CFG, global_attn_temp=0.0)]


def params_for(cfg):
    params = model.init_params(cfg, jax.random.key(0))  # zero-init output projections: perturb so every layer matters
    return jax.tree.map(lambda p: p + 0.05 * jax.random.normal(jax.random.key(1), p.shape), params)


def test_cached_logits_match_forward():
    tokens = np.random.default_rng(0).integers(0, CFG.vocab_size, 100).tolist()
    for cfg in CONFIGS:
        params = params_for(cfg)
        for dtype, tol in [("float32", 1e-4), ("bfloat16", 0.1)]:
            c = dataclasses.replace(cfg, dtype=dtype)
            for pieces in [(100,), (40, 1), (17, 5, 1)]:  # prefill only / prefill + decode / chunked prefill + decode
                gap = engine.logits_gap(params, c, tokens, pieces)
                print(f"{dtype} {pieces}: max |logits diff| {gap.max():.2g}")
                assert gap.max() < tol, (dtype, pieces, gap.max())


def test_generate_matches_trainer():
    cfg = dataclasses.replace(CONFIGS[0], dtype="float32")
    params = params_for(cfg)
    prompts = [[1, 2, 3], list(range(10, 60)), [7] * 20]
    eng = engine.Engine(params, cfg, n_slots=2, max_len=128)  # 3 prompts, 2 slots: one waits for a free slot
    greedy = eng.generate(prompts, max_tokens=60, sampling=engine.Sampling(temperature=0))
    sampled = eng.generate(prompts, max_tokens=60)
    for seq, prompt in zip(greedy, prompts):  # greedy decoding = argmax of the full forward pass at every step
        logits = np.asarray(model.forward(params, np.array([seq.tokens[:-1]]), cfg))[0]
        assert seq.tokens[len(prompt):] == logits[len(prompt) - 1:].argmax(-1).tolist()
        assert seq.mask == [0] * len(prompt) + [1] * 60
    metrics = engine.mismatch(params, cfg, greedy + sampled)
    print(metrics)
    assert metrics["n_tokens"] == 360 and metrics["max_abs_logp_diff"] < 1e-4


def test_multistep_decode_matches_single_step():
    """decode_steps tokens per call (on-device loop with stop handling) = one token per call, under greedy decoding."""
    cfg = dataclasses.replace(CONFIGS[0], dtype="float32")
    params = params_for(cfg)
    prompts = [[1, 2, 3], list(range(10, 60)), [7] * 20, [5, 9]]
    free = engine.Engine(params, cfg, n_slots=4, max_len=128, decode_steps=1).generate(
        prompts, max_tokens=40, sampling=engine.Sampling(temperature=0))
    stop = {free[0].tokens[len(prompts[0]) + 12], free[2].tokens[len(prompts[2]) + 5]}  # some rows stop early
    out = {}
    for k, frac in [(1, 0.5), (5, 0.5), (16, 0.0), (16, 2.0)]:  # all_slots_frac: decode over all slots / gathered rows
        eng = engine.Engine(params, cfg, n_slots=3, max_len=128, decode_steps=k, all_slots_frac=frac)  # one waits
        out[k, frac] = eng.generate(prompts, max_tokens=40, stop=stop, sampling=engine.Sampling(temperature=0))
    out[16] = out[16, 0.0]
    for key in [(5, 0.5), (16, 0.0), (16, 2.0)]:
        for a, b in zip(out[1, 0.5], out[key]):
            assert a.tokens == b.tokens and a.finish_reason == b.finish_reason
            assert np.allclose([x or 0 for x in a.logprobs], [x or 0 for x in b.logprobs], atol=1e-5)
    assert "stop" in {s.finish_reason for s in out[16]}
    for a, b in zip(free, out[16]):  # stopped early, at the first stop token
        n = len(b.tokens) - len(a.tokens) + 40
        assert b.tokens == a.tokens[:len(b.tokens)] and not set(b.tokens[-n:-1]) & stop
    sampled = engine.Engine(params, cfg, n_slots=4, max_len=128, decode_steps=8).generate(prompts, max_tokens=60)
    assert engine.mismatch(params, cfg, sampled)["max_abs_logp_diff"] < 1e-4


def test_shared_prompts_and_prefill_shapes():
    """Sequences with the same prompt prefill it once (the others copy its cache slot), and a prefill step's shape is
    chosen by pending tokens: the generations are those of separate engines, under greedy decoding and sampling."""
    cfg = dataclasses.replace(CONFIGS[0], dtype="float32")
    params = params_for(cfg)
    prompts = [list(range(10, 60))] * 3 + [[1, 2, 3]] * 2 + [[7] * 20, [5]]
    calls = []
    step_fn = engine.step_fn
    engine.step_fn = lambda *args, **kw: calls.append(args[3].shape) or step_fn(*args, **kw)
    try:
        eng = engine.Engine(params, cfg, n_slots=8, max_len=128, prefill_tokens=256)
        shared = eng.generate(prompts, max_tokens=30, sampling=engine.Sampling(temperature=0))
    finally:
        engine.step_fn = step_fn
    assert sum(rows for rows, chunk in calls) < len(prompts) and not eng.leaders and not eng.followers
    for prompt, seq in zip(prompts, shared):
        alone = engine.Engine(params, cfg, n_slots=1, max_len=128).generate(
            [prompt], max_tokens=30, sampling=engine.Sampling(temperature=0))[0]
        assert seq.tokens == alone.tokens
        assert np.allclose([x or 0 for x in seq.logprobs], [x or 0 for x in alone.logprobs], atol=1e-5)
    sampled = engine.Engine(params, cfg, n_slots=4, max_len=128).generate(prompts, max_tokens=40)  # some wait
    assert len({tuple(s.tokens) for s in sampled[:3]}) > 1
    assert engine.mismatch(params, cfg, sampled)["max_abs_logp_diff"] < 1e-4


def test_shared_prompts_bookkeeping():
    """Rounds of groups of identical prompts, continued for a few turns, with too few slots: every follower runs
    (whatever chunk finishes its leader's prefill) and nothing is left over."""
    cfg = dataclasses.replace(CONFIGS[0], dtype="float32")
    eng = engine.Engine(params_for(cfg), cfg, n_slots=12, max_len=128, decode_steps=4, row_buckets=[1, 4, 16])
    rng = np.random.default_rng(0)
    for _ in range(10):
        prompts = [rng.integers(0, 256, rng.integers(2, 40)).tolist() for _ in range(4)]
        seqs = [eng.add(engine.Sequence(p, max_tokens=int(rng.integers(1, 10)))) for p in prompts for _ in range(4)]
        turns = {id(s): int(rng.integers(0, 3)) for s in seqs}
        while eng.has_work():
            for s in eng.step():
                if turns[id(s)] and len(s.tokens) < 100:
                    turns[id(s)] -= 1
                    s.extend(rng.integers(0, 256, rng.integers(1, 12)).tolist())
                    eng.add(s)
                else:
                    eng.release(s)
        assert all(s.done for s in seqs) and not eng.leaders and not eng.followers and len(eng.free) == 12
