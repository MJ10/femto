"""
Inference engine: batched generation with a KV cache, for the chat CLI, the agent loop (agent.py), evals and RL.
A much simplified nano-vllm (github.com/GeeeekExplorer/nano-vllm):
- Each sequence gets a slot of a fixed-capacity KV cache (n_slots x max_len per layer) and keeps it until it is
  released, so a stopped sequence can be continued (a tool output, the next user turn) without recomputing it.
- Every step either prefills (the pending tokens of new or continued sequences, in chunks) or decodes for every
  running sequence, as nano-vllm's scheduler does. A decode step runs up to decode_steps tokens in one jitted call (a
  lax.while_loop with on-device sampling and stop handling, ending early once every row has stopped), so the host
  round trip is paid once per decode_steps tokens. Shapes are bucketed (rows, chunk length, attended cache length), so
  the jitted functions compile a bounded number of times. A prefill step takes the (rows, chunk) shape that runs the
  most pending tokens per padded token (prefill_batch).
- Shared prompts: sequences added with the same tokens (the rollouts of one RL prompt) prefill them once; the others
  wait, then get a copy of that cache slot (copy_slots) and only run their last token.
- The forward pass is model.forward itself, given the cache (same code, dtypes and masks; attention is
  jax.nn.dot_product_attention's XLA implementation with an explicit mask by absolute position). With
  cfg.dtype="float32" on both sides, logits match the full forward pass to ~4e-5 (logprobs ~2e-6 on average). In bf16
  they match up to bf16 rounding: on GPU mostly from cuDNN attention in training (mean |dlogp| ~0.01), on CPU from
  XLA's kernels for steps of < 16 tokens. See tests/test_engine.py and `python engine.py --run_name=<run>`;
  mismatch() measures it on any batch of generations (for RL metrics).
- Sampling: greedy, temperature, top-k, top-p, min-p. Every sampled token comes with its log probability under the
  model (temperature 1, no truncation), i.e. -model.token_losses of the same tokens. For RL sample with the default
  Sampling() (temperature 1, full softmax), so that these are the behaviour policy's log probabilities.
  The chat CLI can also penalize repetition (repetition_penalty, over the tokens the model sampled); this decodes one
  token per call and is not for evals or RL, whose samples must come from the model's distribution.
Local layers only need the last local_window keys, but get a full cache for now (simple, and small at this size).
One engine runs on one device (Engine(device=...)); for several devices, run one engine per device (agent.Agent).

    python engine.py --run_name=rl_web7     # parity with model.forward on a real checkpoint + throughput
"""

import argparse
import collections
import dataclasses
import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import checkpoint
import model

ROW_BUCKETS = [2**i for i in range(13)]  # sequences per step (Engine(row_buckets=...): coarser = fewer compilations)
CHUNK_BUCKETS = [16, 64, 256, 1024]  # tokens per sequence in a prefill step (longer prompts are prefilled in chunks)
MIN_KV_LEN = 256  # attended cache lengths: 256, 512, ... up to max_len
CALL_COST = 1024  # the fixed cost of a prefill call, in padded tokens (for choosing its shape)


def bucket(n, sizes):
    return next(s for s in sizes if s >= n)


@dataclasses.dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0  # 0: greedy
    top_k: int = 0  # 0: off
    top_p: float = 1.0  # 1: off
    min_p: float = 0.0  # keep tokens with at least min_p x the top token's probability; 0: off
    repetition_penalty: float = 1.0  # CTRL's: logits of tokens already sampled are divided by it (x it if < 0); 1: off

    @property
    def penalized(self):
        return self.repetition_penalty != 1


def seen_tokens(seqs, n_rows, vocab_size):
    """(n_rows, vocab_size) bool: the tokens each sequence sampled so far (not its prompt, tool outputs or stop tokens)."""
    seen = np.zeros((n_rows, vocab_size), bool)
    for i, seq in enumerate(seqs):
        seen[i, [t for t, m in zip(seq.tokens, seq.mask) if m]] = True
        seen[i, list(seq.stop)] = False
    return seen


class Sequence:
    """A token sequence being generated: stops at a token in stop, or after max_tokens new tokens, or when the cache
    is full. logprobs[i] is the model's log probability of tokens[i] if it was sampled, else None (prompt, tool
    outputs), so the sampled tokens are the ones to train on."""

    def __init__(self, tokens, max_tokens=256, stop=()):
        self.tokens, self.logprobs = list(tokens), [None] * len(tokens)
        self.max_tokens, self.stop = max_tokens, frozenset(stop)
        self.n_new = 0  # tokens sampled since the sequence was last (re)started
        self.slot, self.n_cached = None, 0  # its cache slot, and how many of its tokens are in it
        self.finish_reason = None  # "stop": a stop token, "length": max_tokens or the cache is full

    def extend(self, tokens, max_tokens=None):
        """Appends given tokens (a tool output, the next user turn); Engine.add then continues generating."""
        self.tokens += tokens
        self.logprobs += [None] * len(tokens)
        self.max_tokens = max_tokens or self.max_tokens

    @property
    def done(self):
        return self.finish_reason is not None

    @property
    def mask(self):
        """1 for the tokens the model sampled (to train on), 0 for given ones."""
        return [int(lp is not None) for lp in self.logprobs]


def init_cache(cfg, n_slots, max_len, device=None):
    """Keys and values of every layer, (n_slots, max_len, heads, head_dim) each, in the compute dtype."""
    shape = (n_slots, max_len, cfg.d_model // cfg.head_dim, cfg.head_dim)
    return [(jnp.zeros(shape, cfg.dtype, device=device), jnp.zeros(shape, cfg.dtype, device=device))
            for _ in range(cfg.n_layer)]


@functools.partial(jax.jit, donate_argnames="cache")
def copy_slots(cache, src, dst):
    """Copies cache slots src[i] to dst[i] (all positions, every layer)."""
    return [(k.at[dst].set(k[src]), v.at[dst].set(v[src])) for k, v in cache]


def sample(logits, key, sampling, seen=None):
    if seen is not None:  # repetition penalty on the tokens of seen_tokens
        p = sampling.repetition_penalty
        logits = jnp.where(seen, jnp.where(logits > 0, logits / p, logits * p), logits)
    if sampling.temperature == 0:
        return jnp.argmax(logits, axis=-1)
    logits = logits / sampling.temperature
    if sampling.min_p > 0:
        logits = jnp.where(logits < logits.max(axis=-1, keepdims=True) + jnp.log(sampling.min_p), -jnp.inf, logits)
    if sampling.top_k:
        kth = jax.lax.top_k(logits, sampling.top_k)[0][:, -1:]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    if sampling.top_p < 1:  # keep the most likely tokens until their probability reaches top_p
        desc = -jnp.sort(-logits, axis=-1)
        probs = jax.nn.softmax(desc, axis=-1)
        keep = jnp.cumsum(probs, axis=-1) - probs < sampling.top_p
        threshold = jnp.min(jnp.where(keep, desc, jnp.inf), axis=-1, keepdims=True)
        logits = jnp.where(logits < threshold, -jnp.inf, logits)
    return jax.random.categorical(key, logits, axis=-1)


@functools.partial(jax.jit, static_argnames=("cfg", "kv_len", "sampling"), donate_argnames="cache")
def step_fn(params, cache, rows, tokens, start, n_new, key, cfg, kv_len, sampling, seen=None):
    """Runs forward on the new tokens of each row and samples the token after its last one (n_new - 1). Returns the
    cache, the sampled tokens and their log probabilities (computed as in model.token_losses, without penalties)."""
    logits, cache = model.forward(params, tokens, cfg, cache=(cache, rows, start, kv_len))
    last = logits[jnp.arange(tokens.shape[0]), n_new - 1]  # (R, V)
    next_tokens = sample(last, key, sampling, seen)
    logprobs = -(jax.nn.logsumexp(last, axis=-1) - jnp.take_along_axis(last, next_tokens[:, None], axis=-1)[:, 0])
    return cache, next_tokens, logprobs


@functools.partial(jax.jit, static_argnames=("cfg", "kv_len", "sampling", "n_steps"), donate_argnames="cache")
def decode_fn(params, cache, rows, tokens, start, budget, stop_ids, key, cfg, kv_len, sampling, n_steps):
    """Up to n_steps decode steps in one call. Row b feeds tokens[b] at position start[b] (then its own samples) and
    samples until it samples one of stop_ids[b] (-1 = padding) or has sampled budget[b] tokens; the loop ends once every
    row has stopped. Each step is step_fn's (same forward, sampling and log probabilities). A stopped row keeps running
    on its own cache slot past its end, which is harmless: those positions are rewritten before anything attends to them.
    Returns the cache, the sampled tokens and log probabilities (R, n_steps) and the number sampled per row (R,)."""
    R = tokens.shape[0]

    def body(state):
        i, cache, token, done, n, out_tokens, out_logprobs = state
        logits, cache = model.forward(params, token[:, None], cfg, cache=(cache, rows, start + i, kv_len))
        last = logits[:, 0]
        next_tokens = sample(last, jax.random.fold_in(key, i), sampling)
        logprobs = -(jax.nn.logsumexp(last, axis=-1) - jnp.take_along_axis(last, next_tokens[:, None], axis=-1)[:, 0])
        out_tokens, out_logprobs = out_tokens.at[:, i].set(next_tokens), out_logprobs.at[:, i].set(logprobs)
        n = n + ~done
        done = done | jnp.any(next_tokens[:, None] == stop_ids, axis=-1) | (n >= budget)
        return i + 1, cache, next_tokens, done, n, out_tokens, out_logprobs

    state = (jnp.int32(0), cache, tokens, budget <= 0, jnp.zeros(R, jnp.int32),
             jnp.zeros((R, n_steps), jnp.int32), jnp.zeros((R, n_steps), jnp.float32))
    _, cache, _, _, n, out_tokens, out_logprobs = jax.lax.while_loop(
        lambda state: (state[0] < n_steps) & ~jnp.all(state[3]), body, state)
    return cache, out_tokens, out_logprobs, n


class Engine:
    """Generates many sequences at once. Either generate() a batch of prompts, or drive it: add() sequences, call
    step() until nothing is waiting or running, and release() finished sequences once they won't be continued."""

    def __init__(self, params, cfg, n_slots=32, max_len=None, prefill_tokens=16384, decode_steps=16,
                 all_slots_frac=0.5, row_buckets=ROW_BUCKETS, seed=0, device=None):
        """device: where the cache lives and the steps run (params must be there too); None: the default device."""
        self.params, self.cfg, self.n_slots, self.device = params, cfg, n_slots, device
        self.max_len = max_len or cfg.seq_len
        self.prefill_tokens = prefill_tokens  # at most this many tokens (rows x chunk) per prefill step
        self.decode_steps = decode_steps  # at most this many tokens per sequence per decode step (one jitted call)
        self.all_slots_frac = all_slots_frac  # decode over all slots when at least this fraction of them are running
        self.row_buckets = [b for b in row_buckets if b < n_slots] + [n_slots + 1]  # rows per step
        self.kv_buckets = sorted({min(MIN_KV_LEN * 2**i, self.max_len) for i in range(20)})
        self.cache = init_cache(cfg, n_slots + 1, self.max_len, device)  # the extra slot takes padding rows' writes
        self.free = list(range(n_slots))[::-1]
        self.waiting, self.running = collections.deque(), []
        self.leaders = {}  # tuple(tokens) -> the sequence prefilling them, while it does (shared prompts)
        self.followers = {}  # id(leader) -> (its tokens, the sequences waiting for its prefill)
        self.key = jax.random.key(seed)

    def add(self, seq):
        """Queues a new sequence, or continues a finished one in its slot (after seq.extend). A sequence that doesn't
        fit in the cache is finished right away (finish_reason "length")."""
        seq.n_new, seq.finish_reason = 0, None
        if len(seq.tokens) >= self.max_len:
            seq.finish_reason = "length"
        elif seq.slot is None:
            self.waiting.append(seq)
        else:
            self.running.append(seq)
        return seq

    def release(self, seq):
        if seq.slot is not None:
            self.free.append(seq.slot)
            seq.slot, seq.n_cached = None, 0

    def has_work(self):
        return bool(self.waiting or self.running)

    def admit(self):
        """Gives waiting sequences free slots. One with the same tokens as a sequence still prefilling them becomes its
        follower: it is not run until the leader's prefill is done (share_prefill)."""
        while self.waiting and self.free:
            seq = self.waiting.popleft()
            seq.slot, seq.n_cached = self.free.pop(), 0
            key = tuple(seq.tokens)
            if key in self.leaders:
                self.followers[id(self.leaders[key])][1].append(seq)
                continue
            if len(seq.tokens) > 1:  # (a 1-token sequence has nothing to prefill)
                self.leaders[key], self.followers[id(seq)] = seq, (key, [])
            self.running.append(seq)

    def share_prefill(self, seqs):
        """Copies the cache slots of the leaders among seqs that have cached all their prompt but its last token (the
        prefill's last chunk can leave it to a decode step) to their followers, which then also have only their last
        token to run (in the next decode step, which samples their first token)."""
        src, dst = [], []
        for seq in seqs:
            if id(seq) not in self.followers or seq.n_cached < len(self.followers[id(seq)][0]) - 1:
                continue
            key, followers = self.followers.pop(id(seq))
            del self.leaders[key]
            for f in followers:
                src.append(seq.slot)
                dst.append(f.slot)
                f.n_cached = len(f.tokens) - 1
                self.running.append(f)
        if dst:
            n = bucket(len(dst), self.row_buckets)
            pad = [self.n_slots] * (n - len(dst))  # the extra slot onto itself
            self.cache = copy_slots(self.cache, np.array(src + pad, np.int32), np.array(dst + pad, np.int32))

    def prefill_batch(self, seqs):
        """The sequences and chunk length of the next prefill step: for every chunk length, the sequences with the most
        pending tokens, as many as fit in prefill_tokens (rows bucketed); the chunk that runs the most pending tokens
        per padded token wins (a call costs CALL_COST more)."""
        seqs = sorted(seqs, key=lambda s: s.n_cached - len(s.tokens))  # most pending first
        best = None
        for chunk in CHUNK_BUCKETS:
            fits = [b for b in self.row_buckets if b * chunk <= self.prefill_tokens]
            n = min(len(seqs), max(fits or [1]))
            useful = sum(min(len(s.tokens) - s.n_cached, chunk) for s in seqs[:n])
            score = useful / (bucket(n, self.row_buckets) * chunk + CALL_COST)
            if best is None or score > best[0]:
                best = score, seqs[:n], chunk
        return best[1], best[2]

    def step(self, sampling=Sampling()):
        """One prefill or decode step. Returns the sequences that finished in it (they keep their slots)."""
        self.admit()
        prefill = [s for s in self.running if len(s.tokens) - s.n_cached > 1]
        if prefill:
            batch, chunk = self.prefill_batch(prefill)
        else:
            chunk, batch = 1, self.running
        if not batch:
            return []
        if chunk == 1 and self.decode_steps > 1 and not sampling.penalized:
            return self.decode(batch, sampling)

        n_rows = bucket(len(batch), self.row_buckets)
        rows = np.full(n_rows, self.n_slots, np.int32)  # padding rows write into the extra slot
        tokens, start, n_new = np.zeros((n_rows, chunk), np.int32), np.zeros(n_rows, np.int32), np.ones(n_rows, np.int32)
        for i, seq in enumerate(batch):
            new = seq.tokens[seq.n_cached:seq.n_cached + chunk]
            rows[i], start[i], n_new[i], tokens[i, :len(new)] = seq.slot, seq.n_cached, len(new), new
        kv_len = bucket(int((start + n_new).max()), self.kv_buckets)
        seen = seen_tokens(batch, n_rows, self.cfg.vocab_size) if sampling.penalized else None
        self.key, key = jax.random.split(self.key)
        self.cache, next_tokens, logprobs = step_fn(self.params, self.cache, rows, tokens, start, n_new, key,
                                                    cfg=self.cfg, kv_len=kv_len, sampling=sampling, seen=seen)
        next_tokens, logprobs = np.asarray(next_tokens), np.asarray(logprobs)

        finished = []
        for i, seq in enumerate(batch):
            seq.n_cached += int(n_new[i])
            if seq.n_cached < len(seq.tokens):
                continue  # more to prefill
            token = int(next_tokens[i])
            seq.tokens.append(token)
            seq.logprobs.append(float(logprobs[i]))
            seq.n_new += 1
            if token in seq.stop:
                seq.finish_reason = "stop"
            elif seq.n_new >= seq.max_tokens or len(seq.tokens) >= self.max_len:
                seq.finish_reason = "length"
            if seq.done:
                finished.append(seq)
        self.running = [s for s in self.running if not s.done]
        self.share_prefill(batch)
        return finished

    def decode(self, batch, sampling):
        """Decodes up to decode_steps tokens for every sequence of batch in one call (decode_fn). With many running
        sequences, row b is cache slot b for every slot (idle ones write nowhere: their positions are past the cache),
        which saves gathering the attended keys and values; otherwise the rows are the batch's slots."""
        all_slots = len(batch) >= self.all_slots_frac * self.n_slots
        n_rows, n_steps = self.n_slots + 1 if all_slots else bucket(len(batch), self.row_buckets), self.decode_steps
        n_stop = bucket(max(len(s.stop) for s in batch), [8, 32, 128, 512])
        index = [seq.slot for seq in batch] if all_slots else list(range(len(batch)))  # row of each sequence
        rows = np.full(n_rows, self.n_slots, np.int32)
        tokens, budget = np.zeros(n_rows, np.int32), np.zeros(n_rows, np.int32)
        start = np.full(n_rows, self.max_len if all_slots else 0, np.int32)
        stop_ids = np.full((n_rows, n_stop), -1, np.int32)
        for i, seq in zip(index, batch):
            rows[i], tokens[i], start[i] = seq.slot, seq.tokens[-1], seq.n_cached
            budget[i] = min(seq.max_tokens - seq.n_new, self.max_len - len(seq.tokens))
            stop_ids[i, :len(seq.stop)] = sorted(seq.stop)
        kv_len = bucket(min(max(seq.n_cached for seq in batch) + n_steps, self.max_len), self.kv_buckets)
        self.key, key = jax.random.split(self.key)
        self.cache, out_tokens, out_logprobs, n = decode_fn(
            self.params, self.cache, None if all_slots else rows, tokens, start, budget, stop_ids, key, cfg=self.cfg,
            kv_len=kv_len, sampling=sampling, n_steps=n_steps)
        out_tokens, out_logprobs, n = np.asarray(out_tokens), np.asarray(out_logprobs), np.asarray(n)

        finished = []
        for i, seq in zip(index, batch):
            new = out_tokens[i, :n[i]].tolist()
            seq.tokens += new
            seq.logprobs += out_logprobs[i, :n[i]].tolist()
            seq.n_cached += len(new)  # every sampled token but the last was fed back in
            seq.n_new += len(new)
            if new and new[-1] in seq.stop:
                seq.finish_reason = "stop"
            elif seq.n_new >= seq.max_tokens or len(seq.tokens) >= self.max_len:
                seq.finish_reason = "length"
            if seq.done:
                finished.append(seq)
        self.running = [s for s in self.running if not s.done]
        return finished

    def generate(self, prompts, max_tokens=256, stop=(), sampling=Sampling()):
        """Continues each prompt (a list of token ids) and returns its Sequence."""
        seqs = [self.add(Sequence(p, max_tokens, stop)) for p in prompts]
        while self.has_work():
            for seq in self.step(sampling):
                self.release(seq)
        return seqs


# ---------------------------------------------------------------------------------------------------------------------
# trainer-inference mismatch

@functools.partial(jax.jit, static_argnames="cfg")
def _trainer_logprobs(params, tokens, cfg):
    """Log probability of every token given the previous ones, by the training forward pass (model.token_losses)."""
    return -model.token_losses(params, tokens[:, :-1], tokens[:, 1:], cfg)


def trainer_logprobs(params, cfg, seqs, batch_size=8):
    """Per sequence, the training forward pass's log probability of each of its tokens (teacher forced), aligned with
    seq.logprobs (None at index 0). Sequences are padded to a multiple of 256 tokens, in batches of batch_size."""
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        length = -(-max(len(s.tokens) for s in batch) // 256) * 256
        tokens = np.zeros((batch_size, length + 1), np.int32)
        for row, s in zip(tokens, batch):
            row[:len(s.tokens)] = s.tokens
        lp = np.asarray(_trainer_logprobs(params, tokens, cfg))
        out += [[None] + lp[j, :len(s.tokens) - 1].tolist() for j, s in enumerate(batch)]
    return out


def mismatch(params, cfg, seqs, batch_size=8):
    """verl-style rollout-vs-trainer metrics over the sampled tokens of seqs (engine logprobs vs teacher-forced
    training forward pass): with log ratio r = logp_train - logp_engine and IS weight w = exp(r),
    kl (k3 estimate of KL(engine || trainer)), mean / max |r|, mean / max w, chi2_token = mean(w^2) - 1, the fraction of
    tokens with w outside [0.5, 2], and the largest per-sequence |sum r|."""
    train = trainer_logprobs(params, cfg, seqs, batch_size)
    r, per_seq = [], []
    for s, t in zip(seqs, train):
        rs = [b - a for a, b in zip(s.logprobs, t) if a is not None]
        r += rs
        per_seq.append(abs(sum(rs)))
    r = np.array(r, np.float64)
    w = np.exp(r)
    return dict(n_tokens=len(r), kl=float(np.mean(w - 1 - r)), mean_abs_logp_diff=float(np.abs(r).mean()),
                max_abs_logp_diff=float(np.abs(r).max()), is_weight_mean=float(w.mean()), is_weight_max=float(w.max()),
                chi2_token=float(np.mean(w**2) - 1), frac_outside_05_2=float(np.mean((w < 0.5) | (w > 2))),
                max_seq_abs_sum_diff=float(max(per_seq)))


def logits_gap(params, cfg, tokens, pieces):
    """Max |engine logits - model.forward logits| per position, for a sequence fed to the engine's forward pass in
    pieces of the given sizes (the last size repeats): e.g. (64, 1) = a 64-token prefill, then one token at a time."""
    T = len(tokens)
    ref = jax.jit(model.forward, static_argnums=2)(params, jnp.asarray([tokens]), cfg)[0]
    fwd = jax.jit(lambda p, c, rows, x, start, cfg, kv_len: model.forward(p, x, cfg, cache=(c, rows, start, kv_len)),
                  static_argnames=("cfg", "kv_len"))
    cache, out, pos, sizes = init_cache(cfg, 1, T), [], 0, list(pieces)
    while pos < T:
        n = min(sizes.pop(0) if len(sizes) > 1 else sizes[0], T - pos)
        logits, cache = fwd(params, cache, jnp.zeros(1, jnp.int32), jnp.asarray([tokens[pos:pos + n]]),
                            jnp.asarray([pos]), cfg=cfg, kv_len=T)
        out.append(logits[0])
        pos += n
    return np.asarray(jnp.abs(jnp.concatenate(out) - ref).max(axis=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--n_slots", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--parity_len", type=int, default=2048)
    parser.add_argument("--decode_steps", type=int, default=16, help="tokens per decode call (throughput: vs 1)")
    args = parser.parse_args()

    from tokenizer import Tokenizer
    params, cfg, _ = checkpoint.load_model(checkpoint.model_path(args.run_name))
    tok = Tokenizer.load(cfg.tokenizer)
    print(f"{jax.default_backend()} | {args.run_name}: {cfg}")

    for dtype in ("bfloat16", "float32"):
        c = dataclasses.replace(cfg, dtype=dtype)
        # a sampled continuation of parity_len tokens; RL-style sampling (temperature 1, full softmax)
        engine = Engine(params, c, n_slots=8, max_len=args.parity_len, decode_steps=args.decode_steps)
        prompts = tok.encode_docs(["The history of the Roman Empire", "<|user_start|>List the files in the current "
                                   "directory, largest first.<|user_end|><|assistant_start|>"], allow_special=True)
        seqs = engine.generate(prompts * 4, max_tokens=args.parity_len)
        for pieces in [(args.parity_len,), (64, 1), (1000, 37, 1)]:
            gap = logits_gap(params, c, seqs[0].tokens[:args.parity_len], pieces)
            print(f"{dtype}: engine vs model.forward logits, fed in pieces {pieces}: max |diff| {gap.max():.3g} "
                  f"(positions < 512: {gap[:512].max():.3g}, >= 512: {gap[512:].max():.3g})")
        print(f"{dtype}: mismatch over {len(seqs)} sampled sequences:",
              json.dumps({k: round(v, 6) for k, v in mismatch(params, c, seqs).items()}))

    # throughput: n_slots sequences of max_tokens new tokens each, one token per decode call vs decode_steps
    for decode_steps, frac in [(1, 2.0), (args.decode_steps, 2.0), (args.decode_steps, 0.5)]:
        for dtype, n_slots in [(d, n) for d in ("bfloat16", "float32") for n in (args.n_slots, 4 * args.n_slots)]:
            engine = Engine(params, dataclasses.replace(cfg, dtype=dtype), n_slots=n_slots, max_len=1024,
                            decode_steps=decode_steps, all_slots_frac=frac)
            prompts = [tok.encode_docs([f"Story number {i}: once upon a time"])[0] for i in range(n_slots)]
            for i in range(2):  # the first time includes compilation
                t0 = time.time()
                seqs = engine.generate(prompts, args.max_tokens)
                n, dt = sum(s.n_new for s in seqs), time.time() - t0
                print(f"{dtype} decode_steps={decode_steps} all_slots_frac={frac} "
                      f"{'(with compilation)' if i == 0 else ''}: {len(seqs)} sequences x {args.max_tokens} tokens: "
                      f"{n} tokens in {dt:.1f}s = {n / dt:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
