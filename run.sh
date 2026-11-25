#!/bin/bash
# From nothing to the final model: data, tokenizer, 10B-token pretraining, 16k context extension, chat + tool
# mid-training, RL with tool execution (terminal, python, search, functions, weather), final evals. One node with 4 GPUs
# (A100-80GB), ~16+ CPUs and ~160 GB of RAM (the search index build): ~7 h 30 min, ~90 GB of disk.
#   bash run.sh            # the full recipe, in $FEMTO_DIR (default $SCRATCH/femto)
#   bash run.sh smoke      # every stage for a few steps on a small data subset (1 GPU is enough, ~1 h), in $SCRATCH/femto_smoke
# Rerunning skips (or resumes) every stage whose output exists, so after a crash, time limit or preemption just run it
# again. On SLURM as one job: slurm/speedrun.sh.
# These are the settings of the released models (run names: final_10b_mila -> femto-base-2k, final_10b_mila_16k_lr01
# -> femto-base, mt_tool20c -> femto-mid, rl_web7 -> femto; checkpoint.RELEASED), except that:
# - 4 GPUs replace pretraining's 2 (same global batch of 2^18 tokens, without gradient accumulation), and RL's 1 (the
#   same 32 x 16 rollouts and update per step, split across the GPUs);
# - nl2cmd_chat is decontaminated from the start (the released pretraining still had 147 of its 12,118 documents);
# - the downloads are of today's HF datasets / tldr pages;
# - RL stage 1 (rl_web5) runs with the final envs: the unserved-function reward follows the prompt's tool list and a
#   quarter of the search items declare irrelevant functions (both only in stage 2 for the released model), and the
#   nl2sh templates that overlapped terminal_new's held-out types are gone.
# So expect the same numbers within noise, not bit for bit. Single seeds throughout.
set -euo pipefail
cd "$(dirname "$0")"
[ -n "${VIRTUAL_ENV:-}" ] || { uv sync --frozen && source .venv/bin/activate; }

DL="" TOK="" SEARCH="" PRE="" EXT="" MID="" RL="" EVAL="" LONG="" AGENT=""
if [ "${1:-}" = smoke ]; then  # tiny settings, appended to the real ones (the last --key=value wins)
    export FEMTO_DIR=${SCRATCH:-$HOME/scratch}/femto_smoke WANDB_MODE=disabled
    DL="--num_files=1 --max_chars=3e6"
    TOK="--max_chars=30000000"
    SEARCH="--max_files=1"  # (the index of 1 of the dump's 41 files)
    PRE="--num_tokens=5242880 --eval_tokens=1048576 --core_max_per_task=20 --long_eval_len=4096"  # 20 steps
    EXT="--num_tokens=1048576 --eval_tokens=1048576 --core_max_per_task=20 --long_eval_len=4096"
    MID="--num_tokens=2097152 --eval_tokens=1048576"
    RL="--steps=2 --n_prompts=4 --group=4 --eval_every=2 --eval_limit=20 --drop_zero_variance=0"  # (all rewards are 0)
    EVAL="--max_per_task=20" LONG="--long_eval_len=4096" AGENT="--limit=20"
fi
BASE=${FEMTO_DIR:-${SCRATCH:-$HOME/scratch}/femto}  # = config.BASE_DIR
stage() { echo; echo "=== $(date '+%F %T') $*"; }
download() { python prepare.py download "$@" $DL; }

# 1. eval sets, first: decontamination and exec_traces drop what overlaps them (CORE bundle, TOOL and code bpb sets
# (man pages need man-db, e.g. a Linux node), long documents). ~5 min.
stage "eval sets"
python -c "import os, build_evals, core_eval, long_eval
core_eval.download_bundle(); build_evals.build(); os.path.exists(long_eval.DOCS_PATH) or long_eval.download_docs()"

# 2. data (~1 h on 8+ CPUs, most of it TinyGSM's executed programs and the ClimbMix download; ~30 GB raw + tokens).
# Every 100th document of a source is held out for validation (ClimbMix: its last file).
stage "data"
download climbmix --num_files=80                       # ~4.6B tokens, nanochat's pretraining data
download shell --num_files=3                           # Stack-Edu Shell, int_score >= 4
download python --num_files=13                         # Stack-Edu Python, int_score >= 4
download sysadmin_qa --num_files=12                    # unix/askubuntu/serverfault/superuser threads, score >= 3
download tldr --num_shards=1
download nl2sh --num_shards=1                          # NL2SH-ALFA + NL2Bash train splits
download fineweb_edu_hq --num_files=3 --max_chars=1e9  # the high-quality anneal sources
download finepdfs_edu --num_files=1 --max_chars=1e9
download finemath --num_files=3 --max_chars=1e9
download wikipedia --num_files=3 --max_chars=1e9
download terminal_nemotron --num_files=9 --max_chars=4e8         # agentic sources (chat / tool format, chat.py)
download terminal_openthoughts --num_files=3 --max_chars=1.5e8
download nl2cmd_chat --num_shards=1 --decontaminate
download exec_traces --max_chars=8e7                             # synthetic shell / python tasks, executed
download smoltalk --num_files=4 --decontaminate                  # chat mid-training sources
download gsm8k --num_shards=1 --decontaminate
download tinygsm --num_files=1 --max_chars=3e8 --decontaminate
download mmlu_aux --num_shards=4 --decontaminate
download arc --num_files=2 --num_shards=1 --decontaminate
download tool_docs --num_shards=1 --decontaminate
download terminal_nemotron_medium --num_files=17 --max_chars=1e9 --decontaminate
download terminal_nemotron_adapters --num_files=3 --max_chars=1e9 --decontaminate
download terminal_agenttrove --num_files=38 --max_chars=1e9 --decontaminate
download fc_apigen --num_shards=2 --decontaminate                # function calling
download fc_glaive --num_shards=2 --decontaminate
download fc_toolace --num_shards=1 --decontaminate
download fc_hermes --num_shards=1 --decontaminate
# the search tool's Wikipedia index: BM25 over 20.8M passages (~7 min on 32 CPUs after the download, 160 GB RAM), then
# the reranking files (search_v2, search.INDEX_DIR)
[ -f "$BASE/search/meta.json" ] || python search.py build --dir="$BASE/search" $SEARCH
[ -f "$BASE/search_v2/meta.json" ] || python search.py upgrade --src="$BASE/search" --dir="$BASE/search_v2"
download search_traces --num_shards=4 --decontaminate            # QA answered with search / fetch (~30 min, 16 CPUs)
download weather_traces --num_shards=2 --decontaminate
download notool_chat --num_files=3 --num_shards=2 --decontaminate
download fc_irrelevant --num_shards=2 --decontaminate            # (after the three above: it re-declares their rows)

# 3. tokenizer v2: 16k BPE on ~90% ClimbMix + ~10% code / shell characters, chat and tool special tokens (~3 min),
# then every source tokenized with it (~20 min)
stage "tokenizer"
[ -f "$BASE/tokenizers/v2/tokenizer.json" ] ||
    python tokenizer.py --name=v2 --mix=climbmix:0.9,shell:0.025,python:0.02,sysadmin_qa:0.045,tldr:0.005,nl2sh:0.005 $TOK
for source in $(ls "$BASE/raw"); do python prepare.py tokenize "$source" --tokenizer=v2; done

# 4. pretraining: 10B tokens, 10% code; the last 15% anneal on high-quality + 9% agentic data (WSD, 65% linear
# warmdown). ~2 h 15 min on 4x A100 (1.4M tok/s) plus in-loop evals (val bpb, CORE-500, TOOL, long context at 16k)
stage "pretraining"
python train.py --run_name=final_10b_mila --num_tokens=10_000_000_000 --weight_decay=0.026 \
    --train_mix=climbmix:0.80,shell:0.08,python:0.06,sysadmin_qa:0.06 \
    --anneal_mix=climbmix:0.26,fineweb_edu_hq:0.15,finepdfs_edu:0.15,finemath:0.10,wikipedia:0.10,shell:0.05,python:0.05,sysadmin_qa:0.05,terminal_nemotron:0.035,terminal_openthoughts:0.015,exec_traces:0.03,nl2cmd_chat:0.01 \
    --anneal_frac=0.15 --long_eval_len=16384 $PRE

# 5. context extension to 16k: 315M tokens (1,200 steps of 16 rows of 16k) on a long-document-rich mix, LR x0.1, fresh
# optimizer, linear decay to 5%. ~5 min on 4x A100 (1.05M tok/s) plus evals (full CORE, long context at 32k): ~8 min
stage "16k context extension"
python train.py --run_name=final_10b_mila_16k_lr01 --init_from=final_10b_mila --seq_len=16384 --device_batch_size=4 \
    --num_tokens=314572800 --long_doc_rows=True --lr_scale=0.1 --warmdown_ratio=1.0 \
    --train_mix=finepdfs_edu:0.15,climbmix:0.27,wikipedia:0.12,fineweb_edu_hq:0.10,finemath:0.08,python:0.08,shell:0.04,sysadmin_qa:0.04,terminal_nemotron:0.04,terminal_openthoughts:0.02,exec_traces:0.04,nl2cmd_chat:0.02 \
    --eval_every=200 --core_every=1200 --save_every=400 --long_eval_len=32768 $EXT

# 6. chat + tool mid-training (mt_tool20c): 0.9B tokens at 4k, LR x0.5, the optimizer state loaded, no weight decay,
# 50% linear warmdown to 0; full loss, then assistant-only loss for the last 20%. Mix by tokens: ~27% pretraining
# replay, ~20% agentic, ~16% SmolTalk, ~7% math as python tool calls, ~9% multiple choice, ~21% tool use (function
# calling, search / weather traces, no-tool chat, 1.4% declared functions that don't fit the request). ~13 min on 4x A100
stage "chat + tool mid-training"
python train.py --run_name=mt_tool20c --init_from=final_10b_mila_16k_lr01 --init_optimizer=True --lr_scale=0.5 \
    --loss_mask=anneal --seq_len=4096 --device_batch_size=16 --num_tokens=900_000_000 --weight_decay=0 \
    --warmup_steps=40 --warmdown_ratio=0.5 --final_lr_frac=0 --anneal_frac=0.2 \
    --train_mix=climbmix:0.08964,fineweb_edu_hq:0.02658,finepdfs_edu:0.01724,finemath:0.01408,wikipedia:0.01655,shell:0.009911,python:0.01363,sysadmin_qa:0.008439,terminal_nemotron:0.01507,terminal_nemotron_medium:0.0151,terminal_nemotron_adapters:0.01153,terminal_openthoughts:0.01053,terminal_agenttrove:0.0262,exec_traces:0.02194,nl2cmd_chat:0.009087,tldr:0.009262,tool_docs:0.004578,smoltalk:0.1092,gsm8k:0.01005,tinygsm:0.2412,mmlu_aux:0.1038,arc:0.005183,fc_apigen:0.03709,fc_glaive:0.04527,fc_toolace:0.0068,fc_hermes:0.000987,search_traces:0.0744,weather_traces:0.02295,notool_chat:0.02251,fc_irrelevant:0.0135 \
    --eval_every=500 --core_every=0 --tool_every=0 --sample_every=0 --save_every=1000 $MID

# 7. RL, GRPO with real tool execution (envs.py): 32 prompts x 16 samples per step, LR x0.05,
# fp32, std-normalized advantages, no reward for episodes that stop without answering; groups of terminal / function
# tasks with no pass get the reference solution (--guide). All 4 GPUs (rl.py uses every local device: one engine per GPU
# for the rollouts, a data-parallel update), the search index in RAM (~8 GB).
# Stage 1 (rl_web5): terminal tasks (tiers 1-4, twice the weight), GSM8K with python, search QA, mock functions,
# weather, no-tool chat and file writing; 600 steps, ~1 h 5 min (~5.3 s / step + evals; 2 h 45 min on 1 GPU)
stage "RL 1"
python rl.py --run_name=rl_web5 --init_from=mt_tool20c --tiers=1,2,3,4 --n_prompts=32 --group=16 --lr_scale=0.05 \
    --tasks=terminal:2,gsm8k:1,search:1,funcall:1,weather:1,chat:1,files:0.5 --steps=600 --require_answer=1 \
    --guide=terminal,funcall --adv_norm=std --eval_every=50 --save_every=50 \
    --eval_suites=terminal,terminal_new,gsm8k,qa,bfcl,weather --eval_limit=100 $RL
# Stage 2 (rl_web7): 300 more steps from rl_web5 with single-command shell tasks (nl2sh) added; ~35 min (1 h 40 min on
# 1 GPU)
stage "RL 2"
python rl.py --run_name=rl_web7 --init_from=rl_web5 --tiers=1,2,3,4 --n_prompts=32 --group=16 --lr_scale=0.05 \
    --tasks=terminal:2,gsm8k:1,search:1,funcall:1,weather:1,chat:1,files:0.5,nl2sh:1 --steps=300 --require_answer=1 \
    --guide=terminal,funcall,nl2sh --adv_norm=std --eval_every=50 --save_every=50 \
    --eval_suites=terminal,terminal_new,gsm8k,nl2sh,qa,bfcl,weather --eval_limit=100 $RL

# 8. final evals (one GPU, ~45 min): the base model (val bpb, CORE and TOOL at 2k; long context to 32k), and the final
# model: agent evals with tool execution (greedy), chat repetition under the CLI's sampling, CORE / TOOL (forgetting)
stage "final evals"
if [ ! -f "$BASE/runs/rl_web7/final_evals.done" ]; then
    python evaluate.py --run_name=final_10b_mila_16k_lr01 --seq_len=2048 $EVAL
    python evaluate.py --run_name=final_10b_mila_16k_lr01 --evals=long $LONG
    python agent_eval.py --run_name=rl_web7 $AGENT  # terminal, NL2SH, GSM8K, ARC
    python agent_eval.py --run_name=rl_web7 --suites=terminal_new $AGENT  # held-out terminal task types
    python agent_eval.py --run_name=rl_web7 --suites=chat --temperature=0.7 --top_p=0.9 --repetition_penalty=1.1 $AGENT
    python agent_eval.py --run_name=rl_web7 --suites=bfcl,qa,qa_closed,weather $AGENT  # search / functions / weather
    python agent_eval.py --run_name=final_10b_mila_16k_lr01 --suites=qa_fewshot $AGENT  # the no-search QA floor
    python evaluate.py --run_name=rl_web7 --evals=core,tool $EVAL
    touch "$BASE/runs/rl_web7/final_evals.done"
fi
stage "done: chat with the model with  python cli.py --run_name=rl_web7"
# (python export.py writes the runs above as released models: runs/<name>/model.safetensors + config.json)
