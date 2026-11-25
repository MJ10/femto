"""
Execution-based agent evals of a saved run, all checked programmatically (no LLM judge). Every item is an agent
episode (agent.py: the model's tool calls really run, in a fresh sandbox per episode), greedy by default:
- terminal   terminal_tasks.py: 312 shell tasks in 4 tiers (one command / pipeline / read-then-act / fix-up), checked
             on the final answer and / or the sandbox's final files
- nl2sh      NL2SH-ALFA test (300 tasks): the model's first bash command and the reference commands run in identical
             fresh sandboxes holding the benchmark's /testbed file tree (a reconstruction); a match is the same output
             and the same final files as either reference. `exec` counts items whose reference doesn't reproduce
             (dates, pids, ...) by exact match; `exec_informative`: only items whose reference succeeds and prints or
             changes something. `exact`: the command is one of the references
- gsm8k      GSM8K test (1319) with the python tool: the last number of the answer
- arc        ARC-Easy / ARC-Challenge test in chat_sources.py's multiple-choice format: the answer's letter
- chat       repetition on open-ended prompts: the first user turn (and system prompt) of the first 100 smol-smoltalk
             test conversations with short prompts (held out: training uses its train split), tools available.
             no_end: replies that don't end with <|assistant_end|> within the budget; rep2 / rep4: mean fraction of
             a reply's token n-grams that repeat an earlier one (Welleck et al. 2019); looping: replies with rep4 > 0.5;
             distinct1 / distinct2: unique n-grams / all n-grams over all replies; length: mean reply tokens (ref_length:
             the dataset's replies). Over the text tokens the model sampled (no special tokens, no tool outputs)
Search, function calling and weather (eval sets: build_evals.build_web; scoring: scoring.py,
shared with the RL rewards):
- bfcl       BFCL v3 non-live (1240: simple, multiple, parallel, parallel_multiple, irrelevance), BFCL-style in our
             format: the schemas in the system prompt (chat.tools_prompt), calls as NAME + JSON arguments, all calls of
             the reply checked with BFCL's AST matching; irrelevance: no call. Not comparable 1:1 with the leaderboard
- qa         NQ-open, TriviaQA, PopQA, HotpotQA (500 each) with search / fetch: EM, token F1, contains (a gold answer
             in the reply), search rate, retrieved (a gold answer in a search output: retrieval vs reading failures)
- qa_closed  the same without tools; qa_fewshot: 5-shot Q/A prompts without the chat format (base models' floor)
- weather    400 requests on held-out cities: weather (the stated temperature vs the simulator), search, or no tool
             (tool choice; choice/none = the "no tool needed" rate)
- format health per suite: tool-call rate, calls per episode, malformed calls, runaway generations (a generation hit
  max_tokens or the context), episodes stopped by the call budget, generated tokens per episode
With --n_samples=n (and a temperature), every item is attempted n times and terminal / gsm8k / arc also report pass@k
(k = 1, 2, 4, ..., n; unbiased estimator of Chen et al. 2021), e.g. to check that RL has something to sharpen.
Results go to BASE_DIR/runs/<run>/agent_eval.json (+ a few transcripts per suite in agent_eval_samples.jsonl).
    python agent_eval.py --run_name=rl_web7 [--suites=terminal,nl2sh,gsm8k,arc] [--limit=20]
    python agent_eval.py --run_name=... --suites=terminal,gsm8k --tiers=1,2 --temperature=1 --n_samples=16
    python agent_eval.py --run_name=... --suites=chat --temperature=0.7 [--top_p=0.9 --repetition_penalty=1.1]
    python agent_eval.py --run_name=... --suites=bfcl,qa,qa_closed,weather
    python agent_eval.py --run_name=final_10b_mila_16k_lr01 --suites=qa_fewshot
Repetition penalties (the chat CLI's) are only allowed for the chat suite.
"""

import argparse
import concurrent.futures
import csv
import gzip
import io
import json
import math
import os
import re
import tarfile
import time

import numpy as np
import pyarrow.parquet as pq

import build_evals
import chat
import checkpoint
import scoring
import terminal_tasks
import tools
import weather
from agent import Agent, Episode
from build_evals import hf_file
from chat_sources import to_number
from engine import Engine, Sampling
from tokenizer import Tokenizer
from tools import WORKDIR, Sandbox, mock


def health(episodes):
    calls = [c for ep in episodes for c in ep.calls]
    mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
    return dict(tool_rate=mean([bool(ep.calls) for ep in episodes]), calls=mean([len(ep.calls) for ep in episodes]),
                malformed_calls=mean([c["tool"] is None for c in calls]),
                malformed_end=mean([ep.finish == "malformed" for ep in episodes]),
                runaway=mean([ep.finish == "length" for ep in episodes]),
                call_budget=mean([ep.finish == "max_calls" for ep in episodes]),
                answered=mean([ep.finish == "answer" for ep in episodes]),
                gen_tokens=mean([sum(ep.seq.mask[ep.start:]) for ep in episodes]))


def sample(ep, **extra):
    return dict(reply=ep.reply, finish=ep.finish, **extra)


def repeat(items, args):
    """Each item n_samples times in a row (the order pass_at_k expects)."""
    return [item for item in items for _ in range(args.n_samples)]


def pass_at_k(ok, n):
    """{pass@k: ...} for k = 1, 2, 4, ..., n from n attempts per item (ok: item-major, as repeat makes it)."""
    correct = np.reshape(ok, (-1, n)).sum(axis=1)
    ks = [2**i for i in range(int(math.log2(n)) + 1)] + ([n] if n & (n - 1) else [])
    return {f"pass@{k}": float(np.mean([1 - math.comb(n - c, k) / math.comb(n, k) for c in correct])) for k in ks}


# ---------------------------------------------------------------------------------------------------------------------
# terminal tasks

def terminal(agent, args, generators=terminal_tasks.GENERATORS):
    tasks = [t for t in terminal_tasks.tasks(args.seed, generators=generators)
             if not args.tiers or str(t.tier) in args.tiers.split(",")]
    tasks = repeat(tasks[:args.limit or None], args)
    boxes = [Sandbox(t.files) for t in tasks]
    episodes = agent.run([Episode(agent.prompt([chat.user(t.instruction)]), box, args.max_calls, args.max_tokens)
                          for t, box in zip(tasks, boxes)])
    ok = []
    for task, ep in zip(tasks, episodes):
        try:
            ok.append(bool(task.check(ep)))
        except Exception:
            ok.append(False)
    for box in boxes:
        box.close()
    results = dict(accuracy=float(np.mean(ok)))
    for tier in sorted({t.tier for t in tasks}):
        results[f"tier{tier}"] = float(np.mean([o for t, o in zip(tasks, ok) if t.tier == tier]))
        if args.n_samples > 1:
            tier_ok = [o for t, o in zip(tasks, ok) if t.tier == tier]
            results.update({f"tier{tier}_{k}": v for k, v in pass_at_k(tier_ok, args.n_samples).items()})
    for name in dict.fromkeys(t.name for t in tasks):
        results[f"task/{name}"] = float(np.mean([o for t, o in zip(tasks, ok) if t.name == name]))
    samples = [sample(ep, task=t.instruction, ok=o) for t, ep, o in zip(tasks, episodes, ok)]
    return results, episodes, samples


# ---------------------------------------------------------------------------------------------------------------------
# NL2SH-ALFA, executed

def archive(members, compress):
    """A tar(.gz) archive with fixed timestamps, so that it is the same in every sandbox."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, text in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(text.encode()), 1704067200
            tar.addfile(info, io.BytesIO(text.encode()))
    return gzip.compress(buf.getvalue(), mtime=0) if compress else buf.getvalue()


TEXT1 = "This is a text file.\nIt has another line.\nhello world\n"
SETUP_SCRIPT = """#!/bin/bash
# sets up the test file system
mkdir -p /testbed/dir1/subdir1
mkdir -p /testbed/dir2/subdir1 /testbed/dir2/subdir2
mkdir -p /testbed/dir3/subdir1/subsubdir1/FooBar
echo "This is a text file." > /testbed/dir1/textfile1.txt
echo "another file with some text" > /testbed/dir1/subdir1/textfile4.txt
echo '{"name": "test", "value": 42}' > /testbed/dir1/subdir1/jsonfile1.json
cp /testbed/dir1/textfile1.txt /testbed/dir2/subdir1/textfile2.txt
ln -s /testbed/dir1 /testbed/link
echo done
"""
FIXTURE = {  # the NL2SH-ALFA test commands' /testbed (the paths they name) and the current directory
    "setup_nl2b_fs_1.sh": SETUP_SCRIPT,
    "notes.txt": "the meeting is at noon\nbring a laptop\n",
    "testbed/dir1/textfile1.txt": TEXT1,
    "testbed/dir1/subdir1/textfile4.txt": "another file\nwith some text\nand a third line\n",
    "testbed/dir1/subdir1/jsonfile1.json": '{"name": "test", "value": 42, "tags": ["a", "b"]}\n',
    "testbed/dir1/subdir1/script1.py": "print('hello from script1')\n",
    "testbed/dir2/subdir1/textfile2.txt": "line one\nline two\nhello\nanother line\nThis is a text file.\n",
    "testbed/dir2/subdir2/script2.py": "import sys\n\nprint(len(sys.argv))\n",
    "testbed/dir2/subdir2/notes.txt": "foo bar baz\n",
    "testbed/dir3/subdir1/subsubdir1/textfile3.txt": "line one\nhello\n",
    "testbed/dir3/subdir1/subsubdir1/FooBar/file.txt": "FooBar file\n",
    "testbed/dir3/subdir1/subsubdir1/tmp": None,
    "testbed/test_dir": None,
    "testbed/hello.php": "<?php\necho 'Hello, World!';\n?>\n",
    "testbed/index.php": "<?php\ninclude 'hello.php';\n?>\n",
    "testbed/textfile7.txt": "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs.\n",
    "testbed/Main.java": "public class Main {\n    public static void main(String[] args) {}\n}\n",
    "testbed/Copy.java": "public class Main {\n    public static void main(String[] args) {}\n}\n",
    "testbed/prog.c": "#include <stdio.h>\n\nint main(void) {\n    printf(\"hi\\n\");\n    return 0;\n}\n",
    "testbed/page.html": "<html><body>foo</body></html>\n",
    "testbed/about.html": "<html><body>about us</body></html>\n",
    "testbed/hashes.txt": "d41d8cd98f00b204e9800998ecf8427e  empty.txt\n5d41402abc4b2a76b9719d911017c592  hello.txt\n",
    "testbed/.hidden": "secret\n",
    "testbed/archive.tar.gz": archive({"dir1/textfile1.txt": TEXT1}, compress=True),
    "testbed/logs.tar.gz": archive({"logs/app.log": "started\nstopped\n"}, compress=True),
    "testbed/hello.tar": archive({"hello.php": "<?php\necho 'Hello, World!';\n?>\n"}, compress=False),
}
FIXTURE_SETUP = "ln -s dir1 testbed/link; find . -exec touch -h -t 202401010000 {} +"


def run_command(command):
    """(output, exit code, final files) of a bash command in a fresh sandbox holding the fixture."""
    with Sandbox(FIXTURE, FIXTURE_SETUP, timeout=5) as box:
        output, code = box.run("bash", command.replace("/testbed", box.workdir + "/testbed"))
        output = "\n".join(line.rstrip() for line in output.replace(WORKDIR + "/testbed", "/testbed").strip("\n").split("\n"))
        return output, code, box.state()


def first_command(reply):
    match = re.search(r"<\|tool_start\|>bash\n(.*?)<\|tool_end\|>", reply, re.S)
    return match[1].strip() if match else None


def nl2sh(agent, args):
    with open(hf_file("westenfelder/NL2SH-ALFA", "test.csv")) as f:
        rows = list(csv.DictReader(f))[:args.limit or None]
    episodes = agent.run([Episode(agent.prompt([chat.user(r["nl"].strip())]), None, max_calls=0, max_tokens=128)
                          for r in rows])
    commands = [first_command(ep.reply) for ep in episodes]
    refs = [list(dict.fromkeys(c for c in (r["bash"].strip(), r["bash2"].strip()) if c)) for r in rows]
    jobs = {(i, "model"): c for i, c in enumerate(commands) if c}
    for i, rs in enumerate(refs):
        jobs.update({(i, f"ref{j}"): c for j, c in enumerate(rs)} | {(i, "again"): rs[0]})
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        out = dict(zip(jobs, pool.map(run_command, jobs.values())))
    _, _, untouched = run_command("true")
    exact, executed, informative, deterministic = [], [], [], []
    for i, (cmd, rs) in enumerate(zip(commands, refs)):
        ref_runs = [out[i, f"ref{j}"] for j in range(len(rs))]
        same = lambda a, b: a[0] == b[0] and a[2] == b[2]
        exact.append(" ".join((cmd or "").split()) in {" ".join(r.split()) for r in rs})
        det = same(out[i, "ref0"], out[i, "again"])
        match = cmd is not None and any(same(out[i, "model"], r) for r in ref_runs)
        deterministic.append(det)
        executed.append(match if det else exact[-1])
        r0 = out[i, "ref0"]
        if det and r0[1] == 0 and (r0[0] or r0[2] != untouched):
            informative.append(match)
    results = dict(exact=float(np.mean(exact)), exec=float(np.mean(executed)),
                   exec_informative=float(np.mean(informative)), n_informative=len(informative),
                   n_deterministic=int(np.sum(deterministic)), command_rate=float(np.mean([c is not None for c in commands])))
    samples = [sample(ep, task=r["nl"], refs=rs, model_output=out.get((i, "model"), ("",))[0][:300],
                      ref_output=out[i, "ref0"][0][:300], exec=e, exact=x)
               for i, (r, rs, ep, e, x) in enumerate(zip(rows, refs, episodes, executed, exact))]
    return results, episodes, samples


# ---------------------------------------------------------------------------------------------------------------------
# GSM8K with the python tool, ARC in the chat format

def last_number(text):
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    return to_number(found[-1]) if found else None


def gsm8k(agent, args):
    rows = pq.read_table(hf_file("openai/gsm8k", "main/test-00000-of-00001.parquet")).to_pylist()[:args.limit or None]
    rows = repeat(rows, args)
    boxes = [Sandbox() for _ in rows]
    episodes = agent.run([Episode(agent.prompt([chat.user(r["question"].strip())]), box, args.max_calls,
                                  args.max_tokens) for r, box in zip(rows, boxes)])
    for box in boxes:
        box.close()
    ok = []
    for r, ep in zip(rows, episodes):
        gold, pred = to_number(r["answer"].split("####")[-1]), last_number(terminal_tasks.answer_text(ep))
        ok.append(pred is not None and abs(pred - gold) < 1e-4 * max(1, abs(gold)))
    return (dict(accuracy=float(np.mean(ok))), episodes,
            [sample(ep, task=r["question"], gold=r["answer"].split("####")[-1].strip(), ok=o)
             for r, ep, o in zip(rows, episodes, ok)])


def arc(agent, args):
    results, episodes, samples = {}, [], []
    for subset in ("ARC-Easy", "ARC-Challenge"):
        rows = pq.read_table(hf_file("allenai/ai2_arc", f"{subset}/test-00000-of-00001.parquet")).to_pylist()
        rows = repeat([r for r in rows if r["answerKey"] in r["choices"]["label"]][:args.limit or None], args)
        prompts = []
        for r in rows:  # chat_sources.mc_doc's letter-only format
            options = "\n".join(f"{letter}. {c.strip()}" for letter, c in zip("ABCDE", r["choices"]["text"]))
            prompts.append(f"{r['question'].strip()}\n\n{options}\n\nAnswer with the letter of the correct choice.")
        boxes = [Sandbox() for _ in rows]
        eps = agent.run([Episode(agent.prompt([chat.user(p)]), box, 2, 64) for p, box in zip(prompts, boxes)])
        for box in boxes:
            box.close()
        ok = []
        for r, ep in zip(rows, eps):
            letter = re.match(r"\s*\(?([A-E])\b", ep.answer) if ep.finish == "answer" else None  # as mc_doc's replies
            ok.append(bool(letter) and "ABCDE".index(letter[1]) == r["choices"]["label"].index(r["answerKey"]))
        results[subset.lower()] = float(np.mean(ok))
        episodes += eps
        samples += [sample(ep, task=p, gold=r["answerKey"], ok=o) for p, r, ep, o in zip(prompts, rows, eps, ok)]
    return results, episodes, samples


# ---------------------------------------------------------------------------------------------------------------------
# repetition on open-ended prompts

def chat_prompts(n=100, max_chars=600):
    """(turns, reference reply) of the first n smol-smoltalk test conversations whose prompts are short."""
    rows = pq.read_table(hf_file("HuggingFaceTB/smol-smoltalk", "data/test-00000-of-00001.parquet")).to_pylist()
    out = []
    for row in rows:
        msgs = [(m["role"], m["content"].strip()) for m in row["messages"]]
        system = [chat.system(text) for role, text in msgs[:1] if role == "system"]
        rest = msgs[len(system):]
        if (len(rest) >= 2 and rest[0][0] == "user" and rest[1][0] == "assistant" and 0 < len(rest[0][1]) <= max_chars
                and sum(len(text) for _, text in system) <= max_chars):
            out.append((system + [chat.user(rest[0][1])], rest[1][1]))
        if len(out) == n:
            break
    return out


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rep_n(tokens, n):
    grams = ngrams(tokens, n)
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def chat_suite(agent, args):
    items = chat_prompts()[:args.limit or None]
    boxes = [Sandbox() for _ in items]
    episodes = agent.run([Episode(agent.prompt(turns), box, 4, args.max_tokens) for (turns, _), box in zip(items, boxes)])
    for box in boxes:
        box.close()
    special = set(agent.special.values())
    texts = [[t for t, m in zip(ep.seq.tokens[ep.start:], ep.seq.mask[ep.start:]) if m and t not in special]
             for ep in episodes]
    distinct = lambda n: len({g for t in texts for g in ngrams(t, n)}) / max(1, sum(len(ngrams(t, n)) for t in texts))
    results = dict(no_end=float(np.mean([ep.finish != "answer" for ep in episodes])),
                   rep2=float(np.mean([rep_n(t, 2) for t in texts])), rep4=float(np.mean([rep_n(t, 4) for t in texts])),
                   looping=float(np.mean([rep_n(t, 4) > 0.5 for t in texts])), distinct1=distinct(1),
                   distinct2=distinct(2), length=float(np.mean([len(t) for t in texts])),
                   ref_length=float(np.mean([len(agent.tok.enc.encode_ordinary(ref)) for _, ref in items])))
    samples = [sample(ep, task=chat.render(turns), rep4=rep_n(t, 4), ok=ep.finish == "answer" and rep_n(t, 4) < 0.2)
               for (turns, _), ep, t in zip(items, episodes, texts)]
    return results, episodes, samples


# ---------------------------------------------------------------------------------------------------------------------
# search, function calling, weather (eval sets: build_evals.build_web, scoring: scoring.py)

def web_items(name, args):
    build_evals.build_web()
    with open(os.path.join(build_evals.EVALS_DIR, "web", f"{name}.jsonl")) as f:
        return repeat([json.loads(line) for line in f][:args.limit or None], args)


BFCL_MOCK = {"status": "success"}  # every declared function's answer: the eval checks the calls only


def run_web(agent, items, tools, args, max_calls=None):
    """Single-turn episodes of the items' questions with tools(item) listed in the system prompt (built-in names and / or
    function schemas, the latter answered by a mock)."""
    boxes = [Sandbox(functions={t["name"]: mock(BFCL_MOCK) for t in tools(item) if isinstance(t, dict)})
             for item in items]
    turns = lambda item: [chat.system(chat.tools_prompt(tools(item)))] * bool(tools(item)) + [chat.user(item["question"])]
    episodes = agent.run([Episode(agent.prompt(turns(item)), box, args.max_calls if max_calls is None else max_calls,
                                  args.max_tokens) for item, box in zip(items, boxes)])
    for box in boxes:
        box.close()
    return episodes


def by_group(name, groups, ok):
    return {f"{name}/{g}": float(np.mean([o for h, o in zip(groups, ok) if h == g])) for g in dict.fromkeys(groups)}


def first_calls(ep):
    """The calls the reply opens with, before it writes any text, without exact repeats: BFCL scores the calls of one
    model response, while here each call gets its mock output at once, so a model may go on calling after the outputs."""
    calls, rest = [], ep.reply
    for c in ep.calls:
        if rest.split("<|tool_start|>", 1)[0].strip():
            break
        if (c["tool"], c["code"]) not in calls:
            calls.append((c["tool"], c["code"]))
        rest = rest.split("<|output_end|>", 1)[1] if "<|output_end|>" in rest else ""
    return calls


def bfcl(agent, args):
    """BFCL v3 non-live, BFCL-style in our format: the functions' JSON schemas in the system prompt, calls as
    NAME + JSON arguments (a mock answers each), checked with BFCL's AST matching (scoring.call_matches) on the calls
    the reply opens with (first_calls). irrelevance: no call before the first text. accuracy: the mean of the 5
    categories (as BFCL's non-live summary)."""
    items = web_items("bfcl", args)
    episodes = run_web(agent, items, lambda item: item["functions"], args)
    ok = [scoring.call_matches([(tool, scoring.parse_args(code)) for tool, code in first_calls(ep)], item["gold"],
                               item["functions"]) for item, ep in zip(items, episodes)]
    results = by_group("category", [item["category"] for item in items], ok)
    results = dict(accuracy=float(np.mean(list(results.values()))), **results)
    samples = [sample(ep, task=item["question"], gold=item["gold"], ok=o) for item, ep, o in zip(items, episodes, ok)]
    return results, episodes, samples


QA_SETS = ("nq_open", "triviaqa", "popqa", "hotpotqa")
FEWSHOT = [("who wrote the novel pride and prejudice", "Jane Austen"), ("what is the capital city of australia", "Canberra"),
           ("how many legs does a spider have", "eight"), ("who painted the mona lisa", "Leonardo da Vinci"),
           ("what is the chemical symbol for gold", "Au")]


def qa_scores(name, items, answers, episodes=None):
    """EM / token F1 of the stated answer and `contains` (a gold answer in it, as words: replies are often sentences);
    with search: the search rate and `retrieved` (a gold answer in some search / fetch output), with `contains` split by
    it, which tells retrieval failures (not retrieved) from reading failures (retrieved, not answered), and
    question_recall@3: a gold answer in the top 3 passages for the question itself (the index's share of the problem)."""
    golds = [item["answers"] for item in items]
    contains = [scoring.has_answer(a, g) for a, g in zip(answers, golds)]
    r = {f"{name}/em": float(np.mean([scoring.exact_match(a, g) for a, g in zip(answers, golds)])),
         f"{name}/f1": float(np.mean([scoring.token_f1(a, g) for a, g in zip(answers, golds)])),
         f"{name}/contains": float(np.mean(contains))}
    if name == "popqa":
        r["popqa/contains_longtail"] = float(np.mean([c for c, i in zip(contains, items) if i["popularity"] < 1000]))
    if episodes is not None:
        lookups = [[c for c in ep.calls if c["tool"] in ("search", "fetch")] for ep in episodes]
        hit = [any(scoring.has_answer(c["output"] or "", g) for c in calls) for calls, g in zip(lookups, golds)]
        recall = [scoring.has_answer(tools.search_output(tools.searcher().search(item["question"], 3)), g)
                  for item, g in zip(items, golds)]  # the index with the question as the query, whatever the model does
        r.update({f"{name}/search_rate": float(np.mean([bool(c) for c in lookups])), f"{name}/retrieved": float(np.mean(hit)),
                  f"{name}/question_recall@3": float(np.mean(recall)),
                  f"{name}/contains_if_retrieved": float(np.mean([c for c, h in zip(contains, hit) if h] or [0])),
                  f"{name}/contains_if_not": float(np.mean([c for c, h in zip(contains, hit) if not h] or [0]))})
    return r


def qa_suite(agent, args, tools):
    results, episodes, samples = {}, [], []
    for name in QA_SETS:
        items = web_items(name, args)
        eps = run_web(agent, items, lambda item: tools, args, max_calls=args.max_calls if tools else 0)
        results.update(qa_scores(name, items, [ep.answer if ep.finish == "answer" else "" for ep in eps],
                                 eps if tools else None))
        episodes += eps
        samples += [sample(ep, task=item["question"], gold=item["answers"][:5],
                           ok=scoring.has_answer(ep.answer, item["answers"])) for item, ep in zip(items, eps)]
    results = dict(contains=float(np.mean([v for k, v in results.items() if k.endswith("/contains")])), **results)
    return results, episodes, samples


def qa(agent, args):
    """Open-domain QA with the search and fetch tools (a BM25 index of Wikipedia)."""
    return qa_suite(agent, args, ["search", "fetch"])


def qa_closed(agent, args):
    """The same questions without tools (closed book)."""
    return qa_suite(agent, args, [])


def qa_fewshot(agent, args):
    """The no-search floor of a base model: 5-shot "Q: ...\\nA: ..." prompts, greedy, the answer up to the line end."""
    prefix = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FEWSHOT)
    results, episodes, samples = {}, [], []
    for name in QA_SETS:
        items = web_items(name, args)
        eps = agent.run([Episode(agent.tok.encode_docs([f"{prefix}Q: {item['question']}\nA:"])[0], None, 0, 32)
                         for item in items])
        answers = [ep.reply.strip().split("\n")[0] for ep in eps]
        results.update(qa_scores(name, items, answers))
        episodes += eps
        samples += [sample(ep, task=item["question"], gold=item["answers"][:5], ok=scoring.exact_match(a, item["answers"]))
                    for item, ep, a in zip(items, eps, answers)]
    results = dict(em=float(np.mean([v for k, v in results.items() if k.endswith("/em")])), **results)
    return results, episodes, samples


ALL_TOOLS = ["bash", "python", "search", "fetch", "weather"]
TEMPERATURE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?\b|C\b)")


def weather_suite(agent, args):
    """Weather requests, lookups and no-tool requests on cities held out from training (build_evals.weather_items),
    with all tools listed. choice: the first call is to the expected tool (weather / search) or there is none
    (choice/none is the "no tool needed" rate). Weather requests: location: a weather call is for the city (its first
    comma-separated part); value: the answer states one temperature, the simulator's for the city (current, or the
    day's high / low; ±0.5 °C), after such a call."""
    items = web_items("weather", args)
    episodes = run_web(agent, items, lambda item: ALL_TOOLS, args)
    choice = [(ep.calls[0]["tool"] if ep.calls else "none") == item["kind"] for item, ep in zip(items, episodes)]
    location, value = [], []
    for item, ep in zip(items, episodes):
        if item["kind"] == "weather":
            location.append(any(weather.parse(c["code"])[0].split(",")[0].strip().casefold() == item["city"].casefold()
                                for c in ep.calls if c["tool"] == "weather"))
            s = weather.summary(weather.simulate(item["city"]))
            truth = s["current"]["temperature"] if item["field"] == "current" else s["daily"][item["day"]][item["field"]]
            stated = [float(x) for x in TEMPERATURE.findall(ep.answer)] if ep.finish == "answer" else []
            value.append(location[-1] and len(stated) == 1 and abs(stated[0] - truth) <= 0.5)  # (no hedging)
    results = dict(value=float(np.mean(value)), location=float(np.mean(location)), choice=float(np.mean(choice)),
                   **by_group("choice", [item["kind"] for item in items], choice))
    ok = iter(value)
    samples = [sample(ep, task=item["question"], kind=item["kind"], ok=next(ok) if item["kind"] == "weather" else c)
               for item, ep, c in zip(items, episodes, choice)]
    return results, episodes, samples


def terminal_new(agent, args):
    """The held-out task types (terminal_tasks.HELDOUT_GENERATORS, 8 x 12), never trained on: transfer."""
    return terminal(agent, args, terminal_tasks.HELDOUT_GENERATORS)


SUITES = {"terminal": terminal, "terminal_new": terminal_new, "nl2sh": nl2sh, "gsm8k": gsm8k, "arc": arc, "chat": chat_suite, "bfcl": bfcl, "qa": qa,
          "qa_closed": qa_closed, "qa_fewshot": qa_fewshot, "weather": weather_suite}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--suites", default="terminal,nl2sh,gsm8k,arc", help=f"of {','.join(SUITES)}")
    parser.add_argument("--limit", type=int, default=0, help="items per suite (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="of the terminal tasks")
    parser.add_argument("--tiers", default="", help="terminal task tiers to run, e.g. 1,2 (default: all)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="chat suite only (the CLI's)")
    parser.add_argument("--n_samples", type=int, default=1, help="attempts per item, for pass@k")
    parser.add_argument("--max_calls", type=int, default=8, help="tool calls per episode")
    parser.add_argument("--max_tokens", type=int, default=512, help="per generation between tool calls")
    parser.add_argument("--n_slots", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    sampling = Sampling(temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty)
    assert not sampling.penalized or args.suites == "chat", "repetition penalties are for the chat suite only"

    path = checkpoint.model_path(args.run_name)
    params, cfg, _ = checkpoint.load_model(path, dtype=args.dtype)
    engine = Engine(params, cfg, n_slots=args.n_slots, max_len=min(args.max_len, cfg.seq_len))
    agent = Agent(engine, Tokenizer.load(cfg.tokenizer), sampling)
    report, all_samples = {}, []
    for name in args.suites.split(","):
        t0 = time.time()
        results, episodes, samples = SUITES[name](agent, args)
        if args.n_samples > 1 and "ok" in samples[0]:
            results.update(pass_at_k([s["ok"] for s in samples], args.n_samples))
        results.update(format=health(episodes), n=len(episodes), seconds=round(time.time() - t0, 1))
        report[name] = results
        all_samples += [dict(suite=name, **s) for s in samples[:3] + [s for s in samples[3:] if s.get("ok")][:3]]
        print(f"{name}: {json.dumps(results)}", flush=True)

    out_dir = os.path.dirname(path)
    suffix = f"_limit{args.limit}" if args.limit else ""
    suffix += f"_tiers{args.tiers.replace(',', '')}" if args.tiers else ""
    suffix += "" if args.suites == parser.get_default("suites") else "_" + args.suites.replace(",", "-")
    suffix += f"_T{args.temperature:g}_n{args.n_samples}" if args.temperature or args.n_samples > 1 else ""
    suffix += f"_p{args.top_p:g}" if args.top_p < 1 else ""
    suffix += f"_rp{args.repetition_penalty:g}" if args.repetition_penalty != 1 else ""
    with open(os.path.join(out_dir, f"agent_eval{suffix}.json"), "w") as f:
        json.dump(dict(args=vars(args), results=report), f, indent=1)
    with open(os.path.join(out_dir, f"agent_eval{suffix}_samples.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in all_samples)
    print(f"\n{'suite':<10} {'score':<48} {'tool rate':>9} {'calls':>6} {'malformed':>9} {'runaway':>8} {'n':>5} {'time':>7}")
    for name, r in report.items():
        scores = ", ".join(f"{k} {v:.3f}" for k, v in r.items() if isinstance(v, float) and "/" not in k and k != "seconds")
        h = r["format"]
        print(f"{name:<10} {scores[:48]:<48} {h['tool_rate']:>9.2f} {h['calls']:>6.2f} {h['malformed_calls']:>9.3f} "
              f"{h['runaway']:>8.3f} {r['n']:>5} {r['seconds']:>6.0f}s")


if __name__ == "__main__":
    main()
