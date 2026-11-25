# Femto

Femto is a 48M-parameter language model that uses tools. It was trained from scratch in pure JAX (no Flax, no optax),
in the spirit of [nanochat](https://github.com/karpathy/nanochat). The pipeline has four stages: 10B tokens of
pretraining at 2k context, a short 16k context extension, mid-training on chat and tool traces, and GRPO RL with real
tool execution. The tools are bash, python, Wikipedia search / fetch, weather and declared functions. You chat with it
in a terminal CLI that runs on a laptop CPU.

This repository has the code of the whole pipeline and the released checkpoints. The story behind it is the blog post
[Training a small language model from scratch](https://mj10.github.io/blog/2026/femto/).

Femto is a small research model, not an assistant to rely on. It can write a one-line shell command, run a short
Python computation or look a fact up, and it gets plenty wrong (see [What it can and can't do](#what-it-can-and-cant-do)).

## Quickstart

```bash
git clone https://github.com/MJ10/femto && cd femto
uv sync                     # Python 3.12-3.13; on Linux this installs jax[cuda12], which also runs on CPU
uv run python cli.py        # downloads femto (193 MB) and the tokenizer on first use, then chat
```

The model and the tokenizer download from the Hugging Face Hub (`config.HF_REPO`) into `~/scratch/femto` (or
`$FEMTO_DIR`, or `$SCRATCH/femto`).

The CLI samples at temperature 0.7, top-p 0.9 and repetition penalty 1.1. On CPU it runs in fp32.

Some options:
```bash
uv run python cli.py --run_name=femto-mid                 # another released model (femto, femto-mid, femto-base, femto-base-2k)
uv run python cli.py --temperature=0                      # greedy decoding (it loops on open-ended prompts)
uv run python cli.py --functions=functions.json           # declare your own functions: [{name, description, parameters, result?}]
uv run python cli.py --live                               # live Wikipedia and Open-Meteo instead of the simulator
uv run python cli.py --system="..." --tools=bash,python   # a system prompt; which tools to list
```
In the chat, `/reset` starts a new conversation with a fresh sandbox, and `/quit` exits.

**Tools and the sandbox.** Every conversation gets a fresh temporary work directory. Inside it, the model sees
`/tmp/work` on Linux and the real path on macOS.
- **Linux:** bash and python run in user, mount, pid and network namespaces, with the home, scratch and tmp trees
  hidden behind empty tmpfs mounts.
- **macOS:** they run under `sandbox-exec`, with no writes outside the work dir and no network.
- **Both:** CPU time and memory are limited, and each call has a 10 s timeout.

The other tools:
- `search` and `fetch` use a local BM25 index over 20.8M Wikipedia passages if you have built one
  (`python search.py build` then `upgrade`, ~33 GB). Otherwise they query live Wikipedia.
- `weather` is a seeded simulator that answers in Open-Meteo's format. `--live` uses the real API.

The sandbox keeps a small model from accidentally damaging your machine. It is not a security boundary against a
determined adversary.

## The released models

| Name | Run (training name) | Stage | Context | What it is for |
|---|---|---|---|---|
| `femto` | `rl_web7` | RL, 600 + 300 GRPO steps | 4k | the final model: chat with tools |
| `femto-mid` | `mt_tool20c` | chat + tool mid-training | 4k | the start of RL; tool and chat format without RL |
| `femto-base` | `final_10b_mila_16k_lr01` | base, after the 16k extension | 16k | a base model for your own post-training |
| `femto-base-2k` | `final_10b_mila` | base, end of pretraining | 2k | the base model the post's CORE number is for |

Every model has the same architecture:
- 10 layers, d_model 512, 4 heads of dimension 128, ReLU² MLP, QK-norm, logit softcap 15;
- untied embeddings with a 16,384-token BPE vocabulary;
- per-layer residual and x0 mixing lambdas;
- three sliding-window layers (512 tokens, RoPE) for each global layer, and the global layers use no positional
  encoding (NoPE), so they extrapolate in length.

That is 48,234,516 parameters. The details are in `model.py` and `config.py`.

Each model is a directory: `runs/<name>/model.safetensors` (the fp32 parameters, named like `blocks.3.wq`) and
`config.json` (the training config, the step and, for femto, the RL settings). The tokenizer is
`tokenizers/v2/tokenizer.json`, which holds the split pattern, the special tokens and the base64 BPE ranks, as tiktoken
uses them.

`checkpoint.py` reads both files with numpy alone, so there is no torch and no safetensors package:
```python
import jax.numpy as jnp
import checkpoint, model
from tokenizer import Tokenizer
params, cfg, meta = checkpoint.load_model(checkpoint.model_path("femto"), dtype="float32")
tok = Tokenizer.load(cfg.tokenizer)
logits = model.forward(params, jnp.array([tok.encode_docs(["Hello"])[0]]), cfg)
```
`engine.py` is the batched KV-cache inference engine, and `agent.py` runs tool-calling episodes. `model_path` and
`load_model` also accept the training checkpoints `runs/<run>/ckpt.npz`, so every script works on your own runs too.

## What it can and can't do

These transcripts are from the blog post: femto, seed 0, CLI sampling.

| Prompt | What femto does | |
|---|---|---|
| Who wrote the novel Pride and Prejudice? | searches, answers "Jane Austen." | ✓ |
| Create notes.txt containing hello world, then tell me how many bytes it is. | `echo "hello world" > notes.txt && wc -c < notes.txt`, answers 12 | ✓ |
| How many lines in /etc/passwd contain the word nologin? | `grep -c 'nologin' /etc/passwd` | ✓ |
| What is 12 times 6? (with an unrelated `reserve_table` function declared) | computes it in python instead of calling the function | ✓ |
| What's the weather in Montreal right now? | right temperature, precipitation read from the wrong field | half |
| What is the capital of Australia? (with `reserve_table` declared) | refuses: "None of the tools provided…" (without the function it searches and says Canberra) | ✗ |
| Write a short haiku about autumn. | prose, not a haiku | ✗ |

**It can:**
- write and run one-step shell commands and short Python programs;
- answer factual questions when a search result contains the answer;
- call declared functions in simple cases;
- report the simulated weather.

**It can't, or not reliably:**
- **Multi-step tasks:** it stops after one or two tool calls. On terminal task types it was not trained on, it solves
  2 of 6.
- **Grade-school math:** 15% on GSM8K, even with python.
- **Open-ended writing:** it has no sense of form.
- **Knowledge without search:** about 4% on QA without tools.
- **Deciding when not to call a declared function:** BFCL irrelevance is 0.48. Calls are over-eager whenever any
  function is declared.

**Other limits:**
- Greedy decoding loops on open-ended prompts, which is why the CLI samples.
- It is English only.
- It has no safety tuning.

## Results

The table has the post's numbers (single seeds, greedy, execution-checked). The last column was re-measured on the
released `femto` files with this code (`agent_eval.py`, `evaluate.py`, bf16 on one A100).

| Metric | femto-mid | after 600 RL steps | **femto** | femto, re-measured |
|---|---|---|---|---|
| Terminal tasks: held-out instances of the 26 trained types (312) | 0.138 | 1.00 | **1.00** | 1.00 |
| Terminal tasks: 6 held-out types ¹ | | | 0.33 | 0.33 |
| NL2SH-ALFA test: execution match (300) | | | **0.303** | 0.297 |
| No call when none is needed (weather suite) | 0.15 | 0.98 | **0.99** | 0.99 |
| Weather: stated temperature within ±0.5 °C (400) | 0.10 | 0.56 | **0.635** | 0.635 |
| QA with search: NQ / TriviaQA / PopQA / HotpotQA, contains (2000) ² | 0.062 | 0.201 | **0.21** | 0.211 |
| GSM8K test with python (1319) | 0.092 | 0.137 | **0.145** | 0.146 |
| BFCL v3 non-live, our call format (1240) ³ | 0.32 | 0.33 | **0.33** | 0.325 |
| BFCL irrelevance: no call when no function fits | 0.58 | 0.44 | **0.48** | 0.479 |
| CORE (22 tasks, centered) | 0.123 | 0.125 | **0.131** | 0.131 |

The re-measured column differs from the post by at most 0.007 (NL2SH, 2 of 300 items). That is bf16 run-to-run noise
on GPU, not the release: re-running the post's own code on the original checkpoint today gives BFCL 0.326, GSM8K 0.146
and NL2SH 0.300, and the released safetensors hold the original parameters bit for bit (`export.py` checks the
logits). The base models re-measure exactly: CORE 0.1273 and 0.1343, val bpb 0.9090 and 0.9114, needles as below.

1. The held-out terminal types are `terminal_new`. Two of its eight types, `count_words` and `backup_csv`, overlapped
   templates of the nl2sh RL environment, so they are left out (all 8: 0.50).
2. femto-mid was scored on an older search index, which on its own costs about 0.035.
3. BFCL here is BFCL v3's non-live split in Femto's own call format, so it is not comparable to the leaderboard.

**The base models:**

| | femto-base-2k | femto-base (16k) |
|---|---|---|
| CORE | **0.127** (GPT-2 124M: 0.114) | 0.134 |
| TOOL ⁴ | 0.304 | 0.297 |
| ClimbMix val bpb | 0.909 | 0.911 |
| 4 numbers hidden in real 32k-token documents | 0.18 | **0.76** |

4. TOOL is our likelihood-scored shell / python benchmark (`tool_eval.py`).

Before RL, the base model emits well-formed tool calls in every terminal episode and solves 7% of the tasks.
Mid-training takes GSM8K-with-python from 0.2% to 9% and costs about 0.013 CORE.

## The pipeline

| Stage | Run | Data | Tokens | Time on 4× A100-80GB | FLOPs |
|---|---|---|---|---|---|
| tokenizer | `v2` | 16k BPE on ~90% ClimbMix, ~10% code and shell | | ~3 min | |
| pretraining | `final_10b_mila` | 80% ClimbMix, 20% code / shell / sysadmin Q&A; the last 15% anneals on high-quality text plus 9% agentic traces | 10B at 2k | ~2 h 20 min | 3.0e18 |
| 16k extension | `final_10b_mila_16k_lr01` | a long-document-rich mix, LR ×0.1 | 315M at 16k | ~10 min | 1.8e17 |
| mid-training | `mt_tool20c` | ~27% replay, ~20% agentic, ~16% chat, ~7% math as python, ~9% multiple choice, ~21% tool use | 0.9B at 4k | ~15 min | 3.0e17 |
| RL 1 | `rl_web5` | terminal, GSM8K, search QA, mock functions, weather, no-tool chat, file writing | 600 steps × 512 episodes | ~1 h 5 min | |
| RL 2 | `rl_web7` | the same plus single-command shell tasks (nl2sh) | 300 steps × 512 episodes | ~35 min | 3.4e16 (both) |

**Pretraining.** Muon trains the hidden matrices and AdamW the embeddings, head and scalars. The schedule is WSD with a
65% linear warmdown, and the batch is 2^18 tokens.

**Mid-training** loads the 16k run's optimizer state. It uses no weight decay and a 50% warmdown to 0. The loss covers
every token, then only the assistant's tokens for the last 20%.

**RL** is GRPO as in nanochat:
- rewards are execution-checked, and advantages are normalized by the group's std;
- truncated importance sampling;
- no KL term;
- groups whose rewards are all equal are dropped, and prompts solved ≥90% of the time are retired;
- a group of terminal, function or nl2sh tasks with no pass gets the task's reference solution as one of its rollouts
  (`--guide`).

RL runs in fp32 on both the engine and the trainer side, so their log-probs agree to ~1e-5. Every rollout's bash and
python run in a fresh sandbox.

Mid-training and RL together are about 10% of the FLOPs. RL took about half as many GPU-hours as pretraining, because
it decodes token by token and executes every tool call.

## Reproduce it

`run.sh` is the whole recipe, from nothing to `femto`, followed by the final evals. It is resumable: rerun it after a
crash or a time limit and it skips or resumes every stage whose output exists.
```bash
bash run.sh           # the full recipe into $FEMTO_DIR (default $SCRATCH/femto)
bash run.sh smoke     # every stage for a few steps on a small data subset: 1 GPU, ~1 h
python export.py      # the runs as released models: runs/<name>/model.safetensors + config.json
```
**Hardware:**
- one node with 4× A100-80GB (H100s work too: ~2.4× faster);
- 16+ CPUs and ~160 GB of RAM, for the search index build;
- ~90 GB of disk for the data and checkpoints, plus ~33 GB for the search index.

End to end it takes about 7 h 30 min: ~2 h of data, search index and tokenizer, then the training times above, then
~45 min of final evals. The original runs used 2 GPUs for pretraining (4 h) and 1 for RL. The whole project, ablations
included, used roughly 150 GPU-hours.

On SLURM, `slurm/speedrun.sh` runs `run.sh` as one job, and `slurm/job.sh` / `slurm/train.sh` run single scripts. Each
job runs a pinned commit from node-local storage. With fewer GPUs, `train.py` accumulates gradients to the same
2^18-token batch, and `rl.py` uses every local GPU. On a GPU with less memory, lower `--device_batch_size`: with
several GPUs, running out of memory hangs in NCCL instead of crashing.

**The stages** (exact flags in `run.sh`):
1. **Eval sets first** (`build_evals.py`, `core_eval.py`, `long_eval.py`, ~5 min), because decontamination and the
   executed traces drop whatever overlaps them.
2. **Data** (`prepare.py download <source>`, ~1 h on 8+ CPUs). Every source is downloaded into tokenizer-independent
   parquet files and keeps every 100th document for validation. `--decontaminate` drops documents that share a 13-gram
   with any eval set (`decontam.py`). The search index is built from Wikipedia (`search.py build`, then `upgrade` for
   the reranking files). The search / weather / irrelevant-function traces are rendered from it (`tool_sources.py`).
3. **Tokenizer** (`tokenizer.py`), then `prepare.py tokenize <source>` writes uint16 `.bin` files. To keep the released
   models compatible, skip the training and use the released `tokenizers/v2/tokenizer.json`: `Tokenizer.load("v2")`
   downloads it when it is missing.
4. **Pretraining, 16k extension and mid-training** are all `train.py` with different flags. `--init_from` starts a
   stage from a finished run, and its data streams continue where that run left them.
5. **RL** is `rl.py`, in two stages.
6. **Final evals:**
   - `evaluate.py`: val bpb, CORE, TOOL and long context;
   - `agent_eval.py`: terminal, NL2SH, GSM8K, ARC, chat repetition, BFCL, QA with and without search, weather.

To post-train from a released checkpoint:
- `rl.py --init_from=femto-mid` runs the RL stages as they ran.
- `train.py --init_from=femto-base` starts from the params with a fresh optimizer state. The released mid-training
  loaded the 16k run's optimizer moments (`--init_optimizer=True`), which the release doesn't ship, so that
  continuation is close to the original but not identical.

What to expect: the same numbers within run-to-run noise, not bit for bit. The datasets are downloaded as they are
today, and RL stage 1 runs with the final environments (`run.sh`'s header lists the differences). Every stage was run
with a single seed.

## Code

The code is flat: each script imports the others by name. Any `Config` field can be overridden with `--key=value`.

| Area | Files |
|---|---|
| model and training | `config.py` (all hyperparameters), `model.py` (the GPT as plain functions over a dict of params), `optim.py` (Muon + AdamW and schedules), `train.py` (data-parallel loop for every stage), `data.py` (BOS-aligned best-fit packing), `checkpoint.py`, `export.py` |
| tokenizer and format | `tokenizer.py` (rustbpe to train, tiktoken to encode), `chat.py` (the chat / tool-call document format and its special tokens) |
| data | `prepare.py` (the source registry: download, decontaminate, tokenize), `chat_sources.py`, `tool_sources.py`, `exec_traces.py` (shell and python tasks, executed), `decontam.py` |
| inference and tools | `engine.py` (KV cache, batched prefill / decode, sampling, log-probs), `agent.py` (the tool loop: token in, token out), `tools.py` (sandbox and tools), `search.py` (BM25 + rerank over Wikipedia), `weather.py`, `cli.py` |
| RL | `rl.py` (GRPO with tool execution), `envs.py` (search, function, weather, chat, files and nl2sh environments), `terminal_tasks.py` (shell task generators: RL and the terminal eval) |
| evals | `evaluate.py` (bpb, `core_eval.py`, `tool_eval.py`, `long_eval.py`), `agent_eval.py`, `build_evals.py`, `scoring.py`, `sample.py` |

```bash
uv run pytest    # ~1 min on CPU: engine/trainer parity, loss masks, chat format, packing, agent loop, RL loss, release formats
```

## Data, licence, citation

- **Data:** every dataset, with its licence and how it was made, is in [DATA.md](DATA.md). Several of the agentic,
  chat, math and function-calling sources were generated by larger LLMs; nothing was distilled by us.
- **Model card:** [MODEL_CARD.md](MODEL_CARD.md).
- **Licence:** the code is under the MIT licence ([LICENSE](LICENSE)). The weights were trained on data under their
  own licences, including some non-commercial ones (see DATA.md).
- **Credits:** the design follows nanochat (Karpathy, 2025), Muon (Jordan et al., 2024) and DCLM's CORE metric.

```bibtex
@misc{jain2026femto,
  author = {Moksh Jain},
  title  = {Training a small language model from scratch},
  year   = {2026},
  url    = {https://mj10.github.io/blog/2026/femto/}
}
```
