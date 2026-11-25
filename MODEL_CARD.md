---
license: mit
library_name: jax
pipeline_tag: text-generation
language:
  - en
tags:
  - jax
  - tool-use
  - agent
  - grpo
  - from-scratch
  - nanochat
---

# Femto

Femto is a 48M-parameter decoder-only language model that uses tools. It was trained from scratch in pure JAX:
- 10B tokens of pretraining at 2k context;
- a 315M-token 16k context extension;
- 0.9B tokens of chat and tool mid-training;
- 900 steps of GRPO with real tool execution.

The tools are bash, python, Wikipedia search / fetch, weather and declared JSON functions.

The code, the training recipe and the CLI are in [github.com/MJ10/femto](https://github.com/MJ10/femto). The write-up
is the blog post [Training a small language model from scratch](https://mj10.github.io/blog/2026/femto/).

## Files

| Path | Model | |
|---|---|---|
| `runs/femto/` | final model (training run `rl_web7`) | chat with tools |
| `runs/femto-mid/` | `mt_tool20c` | after mid-training, before RL |
| `runs/femto-base/` | `final_10b_mila_16k_lr01` | base model, 16k context |
| `runs/femto-base-2k/` | `final_10b_mila` | base model at the end of pretraining, 2k context |
| `tokenizers/v2/tokenizer.json` | | 16,384-token byte-level BPE (tiktoken ranks, base64), 32 special tokens |

Each model directory holds `model.safetensors`, with the fp32 params (48,234,516, ~193 MB), and `config.json`, with the
training config, the step and the RL settings. The repository mirrors the code's data directory, so the code
downloads these files on first use:
```bash
uv run python cli.py                     # chat with femto on CPU or GPU
uv run python agent_eval.py --run_name=femto --suites=terminal,gsm8k
```
The code loads the weights with numpy alone (`checkpoint.load_model`). The parameter names are those of `model.py`'s
param dict: `wte`, `lm_head`, `resid_lambdas`, `x0_lambdas`, `blocks.<i>.{wq,wk,wv,wo,w1,w2}`. The model is not a
transformers architecture, so it needs this code.

## Architecture

- 10 layers, d_model 512, 4 heads × 128, a 4× ReLU² MLP, QK-norm, parameter-free RMSNorm.
- Untied embeddings, logit softcap 15, per-layer residual and x0 lambdas.
- Attention pattern SSSL: 7 sliding-window layers (512 tokens, RoPE base 10k) and 3 global layers with no positional
  encoding.
- Trained with Muon on the hidden matrices and AdamW on the rest.

## Chat format

```
<|bos|><|system_start|>…tools…<|system_end|><|user_start|>question<|user_end|><|assistant_start|>text
<|tool_start|>bash
command<|tool_end|><|output_start|>output
[exit code 0]<|output_end|>answer<|assistant_end|>
```
`chat.py` renders it, and `agent.py` runs the loop. Tool outputs are not the model's tokens.

## Results

All results are single seeds, greedy and execution-checked; there is no LLM judge.

| | femto-mid | femto |
|---|---|---|
| Terminal tasks, held-out instances of 26 trained types | 0.138 | 1.00 |
| Terminal tasks, 6 untrained types | | 0.33 |
| NL2SH-ALFA test (execution match) | | 0.30 |
| GSM8K test with python | 0.092 | 0.145 |
| QA with search (NQ / TriviaQA / PopQA / HotpotQA) | 0.062 | 0.21 |
| Weather (temperature within ±0.5 °C) | 0.10 | 0.635 |
| BFCL v3 non-live, Femto's call format (not leaderboard-comparable) | 0.32 | 0.33 |
| BFCL irrelevance | 0.58 | 0.48 |
| CORE | 0.123 | 0.131 |

The base models:
- `femto-base-2k`: CORE 0.127 (GPT-2 124M: 0.114).
- `femto-base`: CORE 0.134. Retrieval of 4 numbers hidden in 32k-token documents rises from 0.18 to 0.76 over
  femto-base-2k.

## Intended use and limitations

Femto is a research and teaching artifact: a complete small-model pipeline you can read, rerun and modify. It is
**not** a capable assistant.
- It solves one-step shell and python tasks. It fails at most multi-step tasks and most math, and it hallucinates
  freely without search.
- It over-calls declared functions: with an unrelated function declared, it refuses questions it would otherwise
  search for.
- Greedy decoding loops on open-ended prompts. Sample at T 0.7, top-p 0.9, repetition penalty 1.1, as the CLI does.
- It is English only, with no safety tuning and no refusal training.
- The tools execute code. The bundled sandbox (Linux namespaces or macOS `sandbox-exec`) limits writes and the network
  but is not a hardened security boundary. Don't give it credentials or point it at systems you care about.

## Training data

Full list with licences: `DATA.md` in the repository.

**Pretraining:**
- ClimbMix web text;
- Stack-Edu shell and Python;
- Stack Exchange sysadmin threads;
- an anneal on FineWeb-Edu, FinePDFs-Edu, FineMath and Wikipedia;
- 9% agentic traces in the anneal.

**Mid-training and RL:**
- agentic terminal trajectories;
- SmolTalk chat;
- GSM8K and TinyGSM math as python;
- MMLU auxiliary train and ARC multiple choice;
- function-calling datasets;
- search and weather traces generated with our own tools;
- programmatic RL environments with execution-checked rewards.

**LLM-generated data:** we distilled nothing, but many of the existing datasets in the recipe were generated by larger
LLMs:
- the Nemotron-Terminal, OpenThoughts-Agent and AgentTrove trajectories;
- SmolTalk;
- TinyGSM;
- APIGen, Glaive, ToolACE and Hermes function calling;
- parts of ClimbMix.

**Licences:** ClimbMix's upstream licence is CC-BY-NC-4.0, and several sources are CC-BY-SA. Check `DATA.md` before
any commercial use.

**Decontamination:** the post-training sources were 13-gram decontaminated against all eval sets.

## Compute

The final recipe takes ~7.5 h on one node with 4× A100-80GB (`run.sh`). It uses ~3.5e18 FLOPs in total, of which
pretraining is 3.0e18. The whole project, ablations included, used about 150 GPU-hours.

## Citation

```bibtex
@misc{jain2026femto,
  author = {Moksh Jain},
  title  = {Training a small language model from scratch},
  year   = {2026},
  url    = {https://mj10.github.io/blog/2026/femto/}
}
```
