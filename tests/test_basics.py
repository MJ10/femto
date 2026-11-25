"""Sanity checks: the model starts at chance, and model + optimizer can memorize a single batch."""

import math

import jax

import model
import optim
from config import Config
from train import train_step

CFG = Config(vocab_size=256, n_layer=2, d_model=128, head_dim=64, seq_len=64, total_batch_tokens=8 * 64,
             num_tokens=300 * 8 * 64, warmup_steps=5)


def test_init_loss_is_uniform():
    params = model.init_params(CFG, jax.random.key(0))
    tokens = jax.random.randint(jax.random.key(1), (8, CFG.seq_len + 1), 0, CFG.vocab_size)
    loss = model.token_losses(params, tokens[:, :-1], tokens[:, 1:], CFG).mean()
    assert abs(float(loss) - math.log(CFG.vocab_size)) < 0.01


def test_overfits_single_batch():
    params = model.init_params(CFG, jax.random.key(0))
    opt_state = optim.init(params)
    batch = jax.random.randint(jax.random.key(1), (1, 8, CFG.seq_len + 1), 0, CFG.vocab_size)
    step_fn = jax.jit(lambda *args: train_step(*args, CFG))
    for step in range(CFG.num_steps):
        params, opt_state, metrics = step_fn(params, opt_state, batch, step)
    assert float(metrics["loss"]) < 0.5
