"""Greedy sampling without a KV cache, for the training-time samples (engine.py is the real inference engine). The token
buffer has a fixed shape, so the forward pass compiles only once."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

import model


@functools.partial(jax.jit, static_argnums=3)
def next_tokens(params, tokens, lengths, cfg):
    logits = model.forward(params, tokens, cfg)
    return jnp.argmax(logits[jnp.arange(tokens.shape[0]), lengths - 1], axis=-1)


def generate(params, tokenizer, prompts, cfg, max_tokens=16):
    """Greedily continues each prompt by max_tokens tokens and returns the decoded completions."""
    seqs = tokenizer.encode_docs(prompts)
    lengths = np.array([len(s) for s in seqs])
    buf_len = -(-(lengths.max() + max_tokens) // 64) * 64  # round up to a multiple of 64
    tokens = np.full((len(seqs), buf_len), tokenizer.bos_id, dtype=np.int32)
    for row, s in zip(tokens, seqs):
        row[:len(s)] = s
    for _ in range(max_tokens):  # causal attention: the padding after each prompt doesn't affect it
        tokens[np.arange(len(seqs)), lengths] = np.asarray(next_tokens(params, tokens, lengths, cfg))
        lengths += 1
    return [tokenizer.decode(row[len(s):n].tolist()) for row, s, n in zip(tokens, seqs, lengths)]
