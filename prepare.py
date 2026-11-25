"""
Download and tokenize a pretraining source (see SOURCES):
    python prepare.py download shell                              # -> raw/shell/{train,val}/*.parquet with a "text" column
    python tokenizer.py --name=v2 --mix=climbmix:0.9,shell:0.1    # once, trains a tokenizer on the downloaded sources
    python prepare.py tokenize shell --tokenizer=v2               # -> tokens/v2/shell/{train,val}/*.bin, BOS before each doc
Downloads keep every 100th document for validation (ClimbMix: its last file), and are skipped if already there.
With --decontaminate, documents sharing a 13-gram with an eval set are dropped (decontam.py).
run.sh has the exact downloads of the final model's data.
"""

import argparse
import csv
import fnmatch
import functools
import glob
import gzip
import itertools
import json
import os
import random
import re
import subprocess
import tempfile
import time
import zlib
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

import chat
import chat_sources
import decontam
import exec_traces
import tool_sources
from config import BASE_DIR
from data import raw_files
from tokenizer import Tokenizer

VAL_EVERY = 100


@dataclass(frozen=True)
class Source:
    repo: str  # HuggingFace dataset repo
    pattern: str  # space-separated globs of its parquet, jsonl(.gz) or json (a list of rows) files, in sorted order
    text: str = "text"  # the field holding the document
    keep: Callable[[dict], bool] | None = None  # row filter
    docs: Callable[[dict], Iterable[str]] | None = None  # turns a row into documents (instead of row[text])
    by_file: bool = False  # keep the repo's (shuffled) files as they are: the first num_files train, the last one val


def stack_edu(language, min_int_score=0):
    """Keeps Stack-Edu files of one language (its shards are sorted by language) with an edu score >= min_int_score."""
    return lambda row: row["metadata"]["language"] == language and row["int_score"] >= min_int_score


def tldr_docs():
    """One document per English tldr page (concise markdown examples of common and Linux commands)."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth=1", "https://github.com/tldr-pages/tldr", tmp], check=True)
        for path in sorted(glob.glob(os.path.join(tmp, "pages", "common", "*.md")) +
                           glob.glob(os.path.join(tmp, "pages", "linux", "*.md"))):
            with open(path) as f:
                yield f.read()


def nl2sh(split):
    with open(hf_hub_download("westenfelder/NL2SH-ALFA", f"{split}.csv", repo_type="dataset")) as f:
        return list(csv.DictReader(f))


def nl2sh_held_out():
    """Lowercased tasks and the commands of the NL2SH-ALFA and NL2Bash test splits (reserved for evals)."""
    nl2bash = hf_hub_download("jiacheng-ye/nl2bash", "default/test/0000.parquet", repo_type="dataset",
                              revision="refs/convert/parquet")
    test = nl2sh("test")
    held_out = {r["nl"].strip().lower() for r in test} | {r[k].strip() for r in test for k in ("bash", "bash2")}
    return held_out | {r[k].strip().lower() if k == "nl" else r[k].strip()
                       for r in pq.read_table(nl2bash).to_pylist() for k in ("nl", "bash")}


def nl2sh_pairs():
    """Natural language -> bash pairs from the train splits of NL2SH-ALFA and NL2Bash, shuffled. Pairs whose task or
    command appears in the test splits are dropped."""
    nl2bash = hf_hub_download("jiacheng-ye/nl2bash", "default/train/0000.parquet", repo_type="dataset",
                              revision="refs/convert/parquet")
    held_out = nl2sh_held_out()
    pairs = [(" ".join(r["nl"].split()), r["bash"].strip()) for r in nl2sh("train") + pq.read_table(nl2bash).to_pylist()]
    pairs = [p for p in dict.fromkeys(pairs) if p[0].lower() not in held_out and p[1] not in held_out]
    random.Random(0).shuffle(pairs)
    return pairs


def nl2sh_docs(pairs_per_doc=8):
    """NL2SH pairs as '# <task>\\n$ <command>' blocks, a few per document."""
    pairs = nl2sh_pairs()
    for i in range(0, len(pairs), pairs_per_doc):
        yield "\n\n".join(f"# {nl}\n$ {bash}" for nl, bash in pairs[i:i + pairs_per_doc])


# ---------------------------------------------------------------------------------------------------------------------
# agentic sources, rendered in the chat / tool format (chat.py)

def tldr_pairs():
    """(task, command) for each example of each tldr page, e.g. ('Extract an archive, using tar', 'tar xf source.tar')."""
    for page in tldr_docs():
        name = page.split("\n", 1)[0].removeprefix("# ").strip()
        for task, cmd in re.findall(r"^- (.+?):?\n\n`(.+)`$", page, re.M):
            task = re.sub(r"\[(\w+)\]", r"\1", task)  # "[c]reate" -> "create" (marks the letter of a short option)
            cmd = re.sub(r"\{\{\[([^|\]]+)\|[^\]]*\]\}\}", r"\1", cmd)  # {{[-f|--file]}} -> -f
            yield f"{task}, using {name}", re.sub(r"\{\{(.*?)\}\}", r"\1", cmd)  # {{path/to/file}} -> path/to/file


def nl2cmd_chat_docs(pairs_per_doc=6):
    """NL -> bash pairs (NL2SH-ALFA / NL2Bash train splits, tldr examples) as conversations in which each user task is
    answered with a bash tool call. There is no output: the commands are not run."""
    held_out = nl2sh_held_out()
    pairs = nl2sh_pairs() + [p for p in tldr_pairs() if p[0].lower() not in held_out and p[1] not in held_out]
    random.Random(1).shuffle(pairs)
    for i in range(0, len(pairs), pairs_per_doc):
        yield chat.render([turn for nl, cmd in pairs[i:i + pairs_per_doc]
                           for turn in (chat.user(nl), chat.tool_call("bash", cmd))])


# terminus-2 trajectories: the agent answers in JSON {"analysis", "plan", "commands": [{"keystrokes"}], "task_complete"},
# the user turns show the terminal, where each command is echoed after a shell prompt like "root@host:/app# " (which
# directly follows the previous output if that didn't end with a newline)
PROMPT = re.compile(r"(?:\([\w.-]+\) )?root@[\w.-]+:[~/][^\s#$]*[#$](?: |$)", re.M)


def trajectory(row):
    """The messages of a terminus-2 trajectory (AgentTrove has them under either name)."""
    return row.get("conversations") or row["messages"]


def sampled(row, frac):
    """Keeps a deterministic ~frac of the trajectories."""
    key = row.get("trial_name") or trajectory(row)[-1]["content"][:1000]
    return zlib.crc32(key.encode()) < frac * 2**32


def subsample(frac):
    return lambda row: sampled(row, frac)


def terminus_action(text):
    text = text.rsplit("</think>", 1)[-1]
    try:
        action = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return action if isinstance(action, dict) else None
    except ValueError:
        return None


def terminal_screen(text):
    for marker in ("New Terminal Output:", "Current Terminal Screen:"):
        if marker in text:
            return text[text.rindex(marker) + len(marker):].split("\n\nAre you sure you want to mark")[0].strip("\n")
    return None


def strip_echo(part, command):
    """part: what follows a shell prompt, i.e. the echo of the typed command (wrapped at the terminal width, '> ' before
    continuation lines) and then its output. Returns the output, or None if part doesn't start with that echo."""
    j = 0
    for c in command.rstrip("\n"):
        if c == "\n":
            if not part.startswith("\n>", j):
                return None
            j += 3 if part.startswith("\n> ", j) else 2
            continue
        if part.startswith("\n", j):  # wrapped line
            j += 1
        if not part.startswith(c, j):
            return None
        j += 1
    return part[j + 1:] if part.startswith("\n", j) or j == len(part) else None


def command_outputs(screen, commands):
    """The output of each command on a terminal screen, or None if the screen can't be split into them."""
    parts = PROMPT.split(screen)[1:]
    if parts and not parts[-1].strip():
        parts.pop()  # the prompt waiting for the next command
    if len(parts) < len(commands):
        return None
    outputs = [strip_echo(part, cmd) for part, cmd in zip(parts[len(parts) - len(commands):], commands)]
    return None if None in outputs else outputs


def brief(text, max_chars=300):
    """The first sentences of text, up to ~max_chars characters."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text.rfind(". ", 0, max_chars)
    return text[:cut + 1] if cut > 0 else text[:max_chars].rsplit(" ", 1)[0] + " ..."


def terminus_docs(row):
    """Compacts a terminus-2 trajectory into chat documents: the task, then for each step its plan (brief) and one bash
    tool call per command with its truncated output, then the final analysis if the task was marked complete. The
    system prompt, thinking and unparseable steps are dropped. A trajectory longer than MAX_DOC_CHARS is split at step
    boundaries into several documents; the later ones restate the beginning of the task."""
    messages = trajectory(row)
    if "Task Description:" not in messages[0]["content"]:
        return
    task = messages[0]["content"].split("Task Description:", 1)[1].split("Current terminal state:")[0].strip()
    first, again = chat.user(chat.truncate(task, 1000, 300)), chat.user(chat.truncate(task, 300, 0))
    steps = []
    for msg, reply in zip(messages[1::2], messages[2::2] + [None]):
        action = terminus_action(msg["content"]) if msg["role"] == "assistant" else None
        if action is None:
            continue
        if action.get("task_complete") is True:
            steps.append([chat.say(brief(str(action.get("analysis", ""))))])
            break
        commands = action.get("commands")
        commands = [c.get("keystrokes") for c in commands if isinstance(c, dict)] if isinstance(commands, list) else []
        commands = [c for c in commands if isinstance(c, str) and c.strip()]
        screen = terminal_screen(reply["content"]) if reply else None
        if not commands or screen is None or any(not c.endswith("\n") for c in commands):
            continue  # nothing run, or interactive keystrokes (C-c, q, ...)
        outputs = command_outputs(screen, commands)
        if outputs is None:  # one tool call for the whole batch, showing the terminal as is
            calls = [("".join(commands), PROMPT.sub("$ ", screen))]
        else:
            calls = list(zip(commands, outputs))
        plan = brief(str(action.get("plan") or action.get("analysis") or ""))
        steps.append([chat.say(plan)] * bool(plan) +
                     [turn for cmd, out in calls for turn in (chat.tool_call("bash", cmd), chat.tool_output(out))])
    doc = [first]
    for step in steps:
        if len(chat.render(doc + step)) > chat.MAX_DOC_CHARS:
            if len(doc) > 1:
                yield chat.render(doc)
            doc = [again]
            if len(chat.render(doc + step)) > chat.MAX_DOC_CHARS:
                continue  # a step too long on its own
        doc += step
    if len(doc) > 1:
        yield chat.render(doc)


# AgentTrove task sources worked on in the shell, and the fraction of their trajectories kept: NL2Bash-derived
# terminal tasks, bash scripting exercises, and StackOverflow / StackExchange questions (staqc, stack_exchange, and
# exp_tas, which has GLM-4.7 traces). Its other sources are SWE, competitive programming, web, dialogue, freelance jobs...
AGENTTROVE_SOURCES = {"nl2bash": 0.5, "bash textbook": 1.0, "staqc": 1.0, "stack_exchange": 0.25, "exp_tas": 0.1}


def agenttrove_keep(row):
    """A sample of the trajectories of AGENTTROVE_SOURCES that have at least 5 agent turns."""
    frac = AGENTTROVE_SOURCES.get(row["original_source"], 0)
    return frac > 0 and sampled(row, frac) and sum(m["role"] == "assistant" for m in trajectory(row)) >= 5


@functools.cache
def held_out_commands():
    return frozenset(cmd for cmd in nl2sh_held_out() if " " in cmd and len(cmd) >= 15)


def agenttrove_docs(row):
    """terminus_docs, minus documents that run a command of the NL2SH-ALFA / NL2Bash test splits verbatim (AgentTrove's
    nl2bash tasks are derived from NL2Bash, whose test split is reserved for evals)."""
    for doc in terminus_docs(row):
        calls = re.findall(r"<\|tool_start\|>bash\n(.*?)<\|tool_end\|>", doc, re.S)
        if not any(line.strip() in held_out_commands() for call in calls for line in call.split("\n")):
            yield doc


SOURCES = {
    "climbmix": Source("karpathy/climbmix-400b-shuffle", "shard_*.parquet", by_file=True),  # nanochat's pretraining data
    # code and the command line
    "shell": Source("common-pile/stackv2_edu_filtered", "stack-edu-008[6-8].json.gz",
                    keep=stack_edu("Shell", min_int_score=4)),
    "python": Source("common-pile/stackv2_edu_filtered", "stack-edu-007[2-9].json.gz stack-edu-008[0-4].json.gz",
                     keep=stack_edu("Python", min_int_score=4)),
    "sysadmin_qa": Source("HuggingFaceTB/stackexchange_2025_md", "unix.stackexchange.com/* askubuntu.com/* "
                          "serverfault.com/* superuser.com/*", text="ThreadText", keep=lambda row: row["Score"] >= 3),
    "tldr": tldr_docs,
    "nl2sh": nl2sh_docs,
    # agentic data in the chat / tool format (chat.py)
    "terminal_nemotron": Source("nvidia/Nemotron-Terminal-Corpus", "synthetic_tasks/skill_based/*/*/data_filtered.parquet",
                                docs=terminus_docs),  # downloaded with --num_files=9: the easy tasks
    "terminal_openthoughts": Source("open-thoughts/OpenThoughts-Agent-SFT-100K", "data/*.parquet", docs=terminus_docs),
    "nl2cmd_chat": nl2cmd_chat_docs,
    "exec_traces": exec_traces.docs,  # synthetic shell / python tasks, really executed
    # more agentic data for the chat mid-training stage (all trajectories, failed ones too)
    "terminal_nemotron_medium": Source("nvidia/Nemotron-Terminal-Corpus", "synthetic_tasks/skill_based/medium/*/*.parquet "
                                       "synthetic_tasks/skill_based/mixed/*/*.parquet", keep=subsample(0.25),
                                       docs=terminus_docs),
    "terminal_nemotron_adapters": Source("nvidia/Nemotron-Terminal-Corpus", "dataset_adapters/*.parquet",
                                         keep=subsample(0.1), docs=terminus_docs),  # code, math and SWE tasks
    "terminal_agenttrove": Source("open-thoughts/AgentTrove", "data/*.parquet", keep=agenttrove_keep,
                                  docs=agenttrove_docs),
    "tool_docs": chat_sources.tool_docs,  # --help outputs, Python docs, GNU manuals (plain text)
    # chat, math as python tool calls and multiple choice, for the chat mid-training stage (chat_sources.py)
    "smoltalk": Source("HuggingFaceTB/smol-smoltalk", "data/train-*.parquet", docs=chat_sources.smoltalk_docs),
    "gsm8k": Source("openai/gsm8k", "main/train-*.parquet", docs=chat_sources.gsm8k_docs),
    "tinygsm": Source("TinyGSM/TinyGSM", "data/*.parquet", docs=chat_sources.tinygsm_docs),
    "mmlu_aux": Source("cais/mmlu", "auxiliary_train/*.parquet", docs=chat_sources.mmlu_docs),
    "arc": Source("allenai/ai2_arc", "ARC-Easy/train-*.parquet ARC-Challenge/train-*.parquet", docs=chat_sources.arc_docs),
    # tool use: function calling, search / fetch and weather traces, tools listed but not needed (tool_sources.py)
    "fc_apigen": Source("HuggingFaceTB/smoltalk", "data/apigen-80k/train-*.parquet", docs=tool_sources.apigen_docs),
    "fc_glaive": Source("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json",
                        docs=tool_sources.glaive_docs),
    "fc_toolace": Source("Team-ACE/ToolACE", "data.json", docs=tool_sources.toolace_docs),
    "fc_hermes": Source("NousResearch/hermes-function-calling-v1", "func-calling.json", docs=tool_sources.hermes_docs),
    "search_traces": tool_sources.search_docs,
    "fc_irrelevant": tool_sources.irrelevant_docs,  # (after search_traces, weather_traces, notool_chat)
    "weather_traces": tool_sources.weather_docs,
    "notool_chat": Source("HuggingFaceTB/smoltalk", "data/everyday-conversations/train-*.parquet "
                          "data/smol-rewrite/train-*.parquet data/smol-summarize/train-*.parquet",
                          docs=tool_sources.notool_docs),
    # higher-quality sources, e.g. to anneal on at the end of training
    "fineweb_edu_hq": Source("HuggingFaceFW/fineweb-edu", "sample/100BT/*.parquet",
                             keep=lambda row: row["int_score"] >= 4),
    "finepdfs_edu": Source("HuggingFaceFW/finepdfs-edu", "data/eng_Latn/train/*.parquet"),
    "finemath": Source("HuggingFaceTB/finemath", "finemath-4plus/*.parquet"),
    "wikipedia": Source("wikimedia/wikipedia", "20231101.en/*.parquet"),
}
CHAT_SOURCES = {"terminal_nemotron", "terminal_openthoughts", "nl2cmd_chat", "exec_traces", "terminal_nemotron_medium",
                "terminal_nemotron_adapters", "terminal_agenttrove", "smoltalk", "gsm8k", "tinygsm", "mmlu_aux",
                "arc", "fc_apigen", "fc_glaive", "fc_toolace", "fc_hermes", "search_traces", "weather_traces",
                "notool_chat", "fc_irrelevant"}  # tokenized with the special tokens allowed


def read_rows(path):
    if path.endswith(".parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif path.endswith(".json"):
        with open(path) as f:
            yield from json.load(f)
    else:
        with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
            yield from map(json.loads, f)


def write_docs(name, docs, num_shards, max_chars):
    """Writes documents to raw/<name>/train/<shard>.parquet round robin, so that every shard samples the whole source,
    except every VAL_EVERY-th document, which goes to raw/<name>/val/0000.parquet. Stops after max_chars characters."""
    paths = [os.path.join(BASE_DIR, "raw", name, "val", "0000.parquet")]
    paths += [os.path.join(BASE_DIR, "raw", name, "train", f"{i:04d}.parquet") for i in range(num_shards)]
    writers = []
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        writers.append(pq.ParquetWriter(path + ".tmp", pa.schema([("text", pa.string())])))
    n_docs, n_chars = 0, 0
    for batch in itertools.batched(docs, 1000 * num_shards):
        dest = [0 if i % VAL_EVERY == 0 else 1 + i % num_shards for i in range(n_docs, n_docs + len(batch))]
        for j, writer in enumerate(writers):
            writer.write_table(pa.table({"text": pa.array([doc for doc, k in zip(batch, dest) if k == j], pa.string())}))
        n_docs, n_chars = n_docs + len(batch), n_chars + sum(map(len, batch))
        print(f"{name}: {n_docs:,} documents, {n_chars / 1e6:.0f}M characters")
        if n_chars >= max_chars:
            break
    for writer, path in zip(writers, paths):
        writer.close()
        os.replace(path + ".tmp", path)


def n_procs():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def list_docs(fn, row):
    return list(fn(row))


def rendered(fn, rows, batch_size=4096):
    """The documents fn(row) of every row, in order, rendered by a pool of processes."""
    with Pool(n_procs()) as pool:
        for batch in itertools.batched(rows, batch_size):  # (imap would read all the rows ahead)
            for docs in pool.imap(functools.partial(list_docs, fn), batch, chunksize=16):
                yield from docs


def download(name, num_files, num_shards, max_chars, decontaminate):
    source = SOURCES[name]
    if getattr(source, "by_file", False):  # (resumes an interrupted download)
        files = sorted(fnmatch.filter(list_repo_files(source.repo, repo_type="dataset"), source.pattern))
        for split, split_files in [("train", files[:-1][:num_files]), ("val", files[-1:])]:
            snapshot_download(source.repo, repo_type="dataset", allow_patterns=split_files, max_workers=8,
                              local_dir=os.path.join(BASE_DIR, "raw", name, split))
        return
    if raw_files(name, "train"):  # (written at the very end)
        return print(f"{name} is already downloaded")
    if callable(source):
        docs = source()
    else:
        repo_files = list_repo_files(source.repo, repo_type="dataset")
        files = sorted({f for pattern in source.pattern.split() for f in fnmatch.filter(repo_files, pattern)})[:num_files]
        local_dir = snapshot_download(source.repo, repo_type="dataset", allow_patterns=files, max_workers=8)
        rows = itertools.chain.from_iterable(read_rows(os.path.join(local_dir, file)) for file in files)
        rows = (row for row in rows if source.keep is None or source.keep(row))
        docs = rendered(source.docs, rows) if source.docs else (row[source.text] for row in rows)
    if decontaminate:
        docs = decontam.filtered(docs, name)
    write_docs(name, docs, num_shards, max_chars)


def tokenize_file(args):
    tokenizer, src, dst, max_tokens, allow_special = args
    tok = Tokenizer.load(tokenizer)
    n_tokens = 0
    with open(dst + ".tmp", "wb") as f:
        for batch in pq.ParquetFile(src).iter_batches(batch_size=1024, columns=["text"]):
            docs = tok.encode_docs(batch.column("text").to_pylist(), num_threads=1, allow_special=allow_special)
            tokens = np.fromiter(itertools.chain.from_iterable(docs), dtype=np.uint16, count=sum(map(len, docs)))
            f.write(tokens.tobytes())
            n_tokens += len(tokens)
            if n_tokens >= max_tokens:
                break
    os.replace(dst + ".tmp", dst)
    return n_tokens


def tokenize(name, tokenizer, max_val_tokens):
    assert Tokenizer.load(tokenizer).vocab_size <= 2**16, "tokens are stored as uint16"
    jobs = []
    for split in ["train", "val"]:
        raw_root = os.path.join(BASE_DIR, "raw", name, split)
        out_dir = os.path.join(BASE_DIR, "tokens", tokenizer, name, split)
        os.makedirs(out_dir, exist_ok=True)
        for src in raw_files(name, split):
            dst = os.path.join(out_dir, os.path.relpath(src, raw_root).replace("/", "_").replace(".parquet", ".bin"))
            if not os.path.exists(dst):
                max_tokens = max_val_tokens if split == "val" else float("inf")
                jobs.append((tokenizer, src, dst, max_tokens, name in CHAT_SOURCES))
    print(f"tokenizing {len(jobs)} files with {n_procs()} processes")
    t0, total = time.time(), 0
    with Pool(n_procs()) as pool:
        for job, n_tokens in zip(jobs, pool.imap(tokenize_file, jobs)):
            total += n_tokens
            print(f"{os.path.basename(job[2])}: {n_tokens / 1e6:.1f}M tokens | total {total / 1e9:.3f}B | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download", "tokenize"])
    parser.add_argument("source", choices=list(SOURCES))
    parser.add_argument("--num_files", type=int, default=1, help="number of files to download")
    parser.add_argument("--num_shards", type=int, default=8, help="number of train files to write")
    parser.add_argument("--max_chars", type=float, default=float("inf"), help="stop after this many characters")
    parser.add_argument("--decontaminate", action="store_true", help="drop documents that overlap an eval set")
    parser.add_argument("--tokenizer", default="v2", help="name of the tokenizer to tokenize with")
    parser.add_argument("--max_val_tokens", type=int, default=50_000_000)
    args = parser.parse_args()
    if args.action == "download":
        download(args.source, args.num_files, args.num_shards, args.max_chars, args.decontaminate)
    else:
        tokenize(args.source, args.tokenizer, args.max_val_tokens)
