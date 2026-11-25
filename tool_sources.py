"""
Tool-use sources (registered in prepare.SOURCES), rendered in the chat format of chat.py with the available tools in
the system prompt (chat.tools_prompt). No new LLM-generated data: existing datasets, or programmatic traces whose tool
outputs are real (search / fetch over the local Wikipedia index, the weather simulator, python).
    fc_apigen   SmolTalk's apigen-80k (xLAM function-calling-60k + Synth-APIGen + irrelevance): single-turn, calls only
    fc_glaive   glaiveai/glaive-function-calling-v2: multi-turn, with the recorded function responses
    fc_toolace  Team-ACE/ToolACE: mostly single-turn calls, some multi-turn with responses
    fc_hermes   NousResearch/hermes-function-calling-v1 (func-calling.json): calls, responses, final answers
    search_traces   NQ-open / TriviaQA / HotpotQA train questions answered with `search` (and `fetch`) over the local
                index; ~10% of the documents are arithmetic done with `python` instead (no search needed)
    weather_traces  templated weather requests -> `weather` calls (1-2 cities) -> simulator output -> templated answers
    notool_chat SmolTalk everyday conversations, rewriting and summarization, with tools listed but none used
    fc_irrelevant   declared functions that don't fit the request (no shared content word): search / weather traces
                and no-tool chat answered as before, apigen requests refused (apigen's irrelevance replies)
Function calls are `<name>\\n<JSON arguments>`, their outputs the JSON result (tools.function_output). Calls without a
recorded response (apigen, most of ToolACE) get the evals' mock output (MOCK, {"status": "success"}) and the turn ends
with a short templated confirmation naming the calls (no invented result values): otherwise the model never sees a turn
end after such an output and keeps calling. Irrelevance cases (no suitable function -> answer in text) are kept.
Search traces never state an answer that is not in the context: the final answer quotes a gold answer only if it appears
in a search / fetch output; otherwise the trace tries a second step (fetch the top page, search a page named in the
results, or another query) and, if that fails too, says it couldn't find it.
Questions, function-calling conversations and cities with rl_split(text) are left to RL (envs.py); the eval sets
(NQ-open / TriviaQA / HotpotQA validation, PopQA, BFCL; weather cities with scoring.eval_city) are dropped too.

    python tool_sources.py      # prints a few documents of each kind
"""

import ast
import datetime
import functools
import glob
import itertools
import json
import os
import random
import re
import string
import zlib
from multiprocessing import Pool

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import chat
import tools
import weather
from build_evals import WEATHER_CITIES
from chat_sources import rng_for
from config import BASE_DIR
from scoring import eval_city

BUILTINS = ["bash", "python", "search", "fetch", "weather"]
FC_MAX_CHARS = 12_000  # ~3.5k tokens
MOCK = {"status": "success"}  # the output of a call without a recorded response (as agent_eval.BFCL_MOCK)


def rl_split(text):
    """True for the ~20% of questions / conversations / cities reserved for RL (agreed with rl.py)."""
    return zlib.crc32(" ".join(text.lower().split()).encode()) % 10 < 2


@functools.cache
def held_out():
    """The questions of the web evals (normalized: the QA sets' and BFCL's) and BFCL's functions as (name, normalized
    description), from build_evals.build_web's files. Function names alone are not held out: BFCL's are generic
    (calculate_tip, get_stock_price, ...) and shared with ~25% of glaive's rows; 13-gram decontamination catches its
    longer descriptions anyway."""
    web = os.path.join(BASE_DIR, "evals", "web")
    if not os.path.exists(os.path.join(web, "bfcl.jsonl")):
        print("warning: no web eval sets found, can't drop the questions and functions they use (build_evals.py)")
        return frozenset(), frozenset()
    questions, functions = set(), set()
    for name in ("nq_open", "triviaqa", "popqa", "hotpotqa", "bfcl"):
        with open(os.path.join(web, f"{name}.jsonl")) as f:
            for item in map(json.loads, f):
                questions.add(normalize(item["question"]))
                functions.update((fn["name"], normalize(fn.get("description", ""))) for fn in item.get("functions", []))
    return frozenset(questions), frozenset(functions)


def system_prompt(rng, used, functions=()):
    """The tools prompt: all built-ins (70%) or a random subset containing the used ones, then the functions. With
    functions, half of the prompts list the functions only."""
    if functions and rng.random() < 0.5:
        return chat.system(chat.tools_prompt(list(functions)))
    names = BUILTINS if rng.random() < 0.7 else [t for t in BUILTINS if t in used or rng.random() < 0.5]
    return chat.system(chat.tools_prompt(names + list(functions)))


# ---------------------------------------------------------------------------------------------------------------------
# function-calling datasets

def schema(f):
    """A function schema as {name, description, parameters}, unwrapped from {"type": "function", "function": ...} and
    without null fields; the parameter style of each dataset is kept."""
    f = f.get("function", f)
    return {k: v for k, v in f.items() if v is not None and k in ("name", "description", "parameters")}


CONFIRMATIONS = ["Done: {calls}.", "I called {calls}.", "I've sent the request: {calls}.", "Called {calls}; it succeeded."]


def confirmation(calls):
    """A short reply after calls whose output is the mock: the calls as name(arg=value, ...), no result values."""
    def value(v):
        text = json.dumps(v, ensure_ascii=False)
        return text if len(text) <= 60 else text[:57] + "..."
    text = " and ".join(f"{name}({', '.join(f'{k}={value(v)}' for k, v in args.items())})" for name, args in calls)
    return rng_for(text).choice(CONFIRMATIONS).format(calls=text)


def calls_and_outputs(calls, outputs=None):
    """Turns for calls [(name, args)] and their outputs [result]. Without outputs (no recorded response), every call
    gets the mock output and a short confirmation that names the calls ends the turn."""
    turns = []
    for i, (name, args) in enumerate(calls):
        turns.append(chat.tool_call(name, json.dumps(args, ensure_ascii=False)))
        turns.append(chat.output(tools.function_output(MOCK if outputs is None else outputs[i])))
    return turns + ([chat.say(confirmation(calls))] if outputs is None else [])


def fc_doc(rng, functions, turns):
    """A function-calling conversation with its system prompt."""
    if not functions or not turns or turns[0][0] != "user" or rl_split(turns[0][1]):
        return None
    questions, bfcl = held_out()
    if normalize(turns[0][1]) in questions or any((f["name"], normalize(f.get("description", ""))) in bfcl
                                                  for f in functions):
        return None
    doc = chat.render([system_prompt(rng, set(), functions)] + turns)
    return doc if len(doc) <= FC_MAX_CHARS else None


def known(calls, functions):
    names = {f["name"] for f in functions}
    return all(name in names and isinstance(args, dict) for name, args in calls)


def apigen_docs(row):
    system, user, reply = (m["content"] for m in row["messages"])
    functions = [schema(f) for f in json.loads(system.split("<tools>", 1)[1].split("</tools>", 1)[0])]
    match = re.fullmatch(r"<tool_call>(.*)</tool_call>", reply.strip(), re.S)
    if match is None:  # irrelevance or missing parameters: an answer in text
        turns = [chat.user(user), chat.say(reply.strip())]
    else:
        calls = [(c.get("name"), c.get("arguments")) for c in json.loads(match.group(1))]
        if not calls or not known(calls, functions):
            return
        turns = [chat.user(user)] + calls_and_outputs(calls)
    if doc := fc_doc(rng_for(user), functions, turns):
        yield doc


GLAIVE_TURN = re.compile(r"(?:^|\n\n\n)(USER|ASSISTANT|FUNCTION RESPONSE): ")
GLAIVE_CALL = re.compile(r"<functioncall>\s*\{\"name\":\s*\"([^\"]+)\",\s*\"arguments\":\s*'?(\{.*\})'?\s*\}\s*$", re.S)


def glaive_docs(row):
    """Conversations with at least one function; a call is the end of an assistant message, its response the next
    FUNCTION RESPONSE. Rows calling an undeclared function or with unparseable JSON are dropped."""
    system = row["system"].removeprefix("SYSTEM: ")
    if "{" not in system:
        return
    functions, decoder, text = [], json.JSONDecoder(), system[system.index("{"):]
    try:
        while text.strip():
            f, end = decoder.raw_decode(text.strip())
            functions.append(schema(f))
            text = text.strip()[end:]
    except json.JSONDecodeError:
        return
    parts = GLAIVE_TURN.split(row["chat"].strip())[1:]
    turns = []
    for role, text in zip(parts[::2], parts[1::2]):
        text = text.replace("<|endoftext|>", "").strip()
        if role == "USER":
            turns.append(chat.user(text))
        elif role == "ASSISTANT":
            before, _, call = text.partition("<functioncall>")
            if before.strip():
                turns.append(chat.say(before.strip()))
            if call:
                match = GLAIVE_CALL.match("<functioncall>" + call)
                try:
                    calls = [(match.group(1), json.loads(match.group(2)))]
                except (AttributeError, json.JSONDecodeError):
                    return
                if not known(calls, functions):
                    return
                turns += calls_and_outputs(calls)
        else:
            if not turns or turns[-1][0] != "tool":
                return
            try:
                turns.append(chat.output(tools.function_output(json.loads(text))))
            except json.JSONDecodeError:
                return
    if turns and turns[-1][0] == "output":
        return  # cut off after a response
    if doc := fc_doc(rng_for(row["chat"]), functions, turns):
        yield doc


def toolace_calls(text):
    """'[Get Weather(city="Paris", days=3), Other()]' -> [(name, args)], or None."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith(")]")):
        return None
    calls, i = [], 1
    while i < len(text) - 1:
        j = text.find("(", i)
        if j < 0:
            return None
        depth, quote, k = 0, None, j
        for k in range(j, len(text)):  # the matching parenthesis, outside of string literals
            c = text[k]
            if quote:
                quote = None if c == quote and text[k - 1] != "\\" else quote
            elif c in "\"'":
                quote = c
            elif c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    break
        try:
            call = ast.parse(f"f({text[j + 1:k]})", mode="eval").body
            args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        except (SyntaxError, ValueError):
            return None
        if call.args:
            return None
        calls.append((text[i:j].strip(" ,"), args))
        i = k + 1
        while i < len(text) - 1 and text[i] in " ,":
            i += 1
    return calls


def toolace_docs(row):
    marker = "can invoke:\n"
    if marker not in row["system"]:
        return  # (~5% list their tools in other formats)
    try:
        functions = json.JSONDecoder().raw_decode(row["system"].split(marker, 1)[1])[0]
    except json.JSONDecodeError:
        return
    functions = [schema(f) for f in functions]
    turns, pending = [], []
    for m in row["conversations"]:
        if m["from"] == "user":
            turns.append(chat.user(m["value"]))
        elif m["from"] == "tool":
            try:
                results = json.loads(m["value"])
            except json.JSONDecodeError:
                return
            if not pending or len(results) != len(pending):
                return
            turns += calls_and_outputs(pending, [r.get("results", r) for r in results])
            pending = []
        elif (calls := toolace_calls(m["value"])) is not None:
            if pending or not calls or not known(calls, functions):
                return
            pending = calls
        else:
            if pending:
                return
            turns.append(chat.say(m["value"].strip()))
    turns += calls_and_outputs(pending) if pending else []
    if doc := fc_doc(rng_for(row["conversations"][0]["value"]), functions, turns):
        yield doc


TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def hermes_docs(row):
    functions = [schema(f) for f in json.loads(row["tools"])] if row.get("tools") else []
    turns, pending = [], []
    try:
        for m in row["conversations"]:
            if m["from"] == "human":
                turns.append(chat.user(m["value"].strip()))
            elif m["from"] == "gpt":
                text = TOOL_CALL.sub("", m["value"]).strip()
                calls = [json.loads(c) for c in TOOL_CALL.findall(m["value"])]
                if pending or (text and calls):
                    return
                if text:
                    turns.append(chat.say(text))
                pending = [(c["name"], c["arguments"]) for c in calls]
                if not known(pending, functions):
                    return
            elif m["from"] == "tool":
                responses = [json.loads(r) for r in TOOL_RESPONSE.findall(m["value"])]
                if len(responses) != len(pending) or [r.get("name") for r in responses] != [c[0] for c in pending]:
                    return
                turns += calls_and_outputs(pending, [r.get("content") for r in responses])
                pending = []
    except (json.JSONDecodeError, KeyError, TypeError):
        return
    turns += calls_and_outputs(pending) if pending else []
    if doc := fc_doc(rng_for(row["id"]), functions, turns):
        yield doc


# ---------------------------------------------------------------------------------------------------------------------
# search traces

def normalize(text):
    """SQuAD-style answer normalization: lowercase, no punctuation or articles, single spaces."""
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def found(answers, text):
    """The first gold answer that occurs in text (normalized, whole words), as written there if it can be located
    (e.g. 'Grantchester' for the alias 'GRANTCHESTER'), or None."""
    normalized = f" {normalize(text)} "
    for answer in answers:
        if (a := normalize(answer)) and f" {a} " in normalized:
            words = [re.escape(w) for w in a.split()]
            span = (re.search(r"(?<!\w)" + re.escape(answer.strip()) + r"(?!\w)", text, re.I) or
                    re.search(r"(?<!\w)" + r"[\W_]+".join(words) + r"(?!\w)", text, re.I))
            return span.group() if span else answer
    return None


def qa_rows():
    """{question, answers, titles} of the train questions of NQ-open, TriviaQA and HotpotQA (minus rl_split ones and
    yes/no answers, and the questions of the evals), interleaved, each dataset shuffled."""
    def rows(repo, path):
        return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()
    nq = [dict(question=r["question"], answers=r["answer"]) for r in
          rows("google-research-datasets/nq_open", "nq_open/train-00000-of-00001.parquet")]
    trivia = [dict(question=r["question"], answers=[r["answer"]["value"]] + r["answer"]["aliases"]) for r in
              rows("mandarjoshi/trivia_qa", "rc.nocontext/train-00000-of-00001.parquet")]
    hotpot = [dict(question=r["question"], answers=[r["answer"]], titles=list(dict.fromkeys(r["supporting_facts"]["title"])))
              for i in range(2) for r in rows("hotpotqa/hotpot_qa", f"distractor/train-0000{i}-of-00002.parquet")]
    out = []
    for i, data in enumerate([nq, trivia, hotpot]):
        data = [r for r in {r["question"].strip(): r for r in data}.values()
                if not rl_split(r["question"]) and normalize(r["answers"][0]) not in ("yes", "no")
                and normalize(r["question"]) not in held_out()[0]]
        random.Random(i).shuffle(data)
        out.append(data)
    return [r for rs in itertools.zip_longest(*out) for r in rs if r]


STOPWORDS = set("""a an the of in on at to for by with from and or is are was were be been being do does did what which
who whom whose when where why how that this these those it its as name named called there their his her than""".split())


def keywords(question):
    """A keyword query: the question's words without stopwords, e.g. 'who wrote the hobbit' -> 'wrote hobbit'."""
    words = re.findall(r"[\w'-]+", question)
    return " ".join(w for w in words if w.lower() not in STOPWORDS) or question


def first_query(rng, question):
    q = question.strip()
    q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?") if rng.random() < 0.5 else q
    return keywords(question) if rng.random() < 0.4 else q


ANSWERS = ["{a}.", "The answer is {a}.", "It's {a}.", "{a}, according to the article on {t}.",
           "According to Wikipedia ({t}), the answer is {a}.", "Based on the search results: {a}."]
NOT_FOUND = ["I couldn't find that in the search results.", "I couldn't find the answer, sorry.",
             "The search results don't answer this, so I don't know.",
             "I couldn't find it: none of the results mention the answer."]


def step(searcher, tool, arg, answers):
    """A search or fetch call and its output: (turns, the first gold answer in the output or None, the title of the
    page it is on, the [(title, passage)] shown)."""
    if tool == "fetch":
        out = tools.fetch_output(searcher.page(arg))
        pages = [(arg, out)]
    else:
        pages = searcher.search(arg, 3)
        out = tools.search_output(pages)
    answer = found(answers, out)
    title = next((t for t, p in pages if found([answer], f"{t}\n{p}")), pages[0][0]) if answer else None
    return [chat.tool_call(tool, arg), chat.output(out)], answer, title, pages


def search_trace(searcher, row, not_found_frac=0.3):
    """(tools used, turns) of one question: a search; if its results don't contain a gold answer, one more step (the
    first of these, in random order, that finds one: fetch the top result's page; for HotpotQA, search a supporting
    page named in the results; search with another query); then a short answer containing it. If no step finds it, the
    trace (kept with probability not_found_frac, else None) makes one of the steps and says it couldn't find it."""
    rng, q, answers = rng_for(row["question"]), row["question"].strip(), row["answers"]
    query = first_query(rng, q)
    turns, answer, title, first = step(searcher, "search", query, answers)
    used = {"search"}
    if not answer:
        steps = [("fetch", first[0][0])] if first else []
        steps += [("search", t) for t in row.get("titles", []) if t in turns[-1][1] and t not in dict(first)][:1]
        steps.append(("search", keywords(q) if query != keywords(q) else q.rstrip("?")))
        rng.shuffle(steps)
        for tool, arg in steps:
            more, answer, title, _ = step(searcher, tool, arg, answers)
            if answer:
                turns, used = turns + more, used | {tool}
                break
        if not answer:
            if rng.random() >= not_found_frac:
                return None
            return used | {steps[0][0]}, turns + more + [chat.say(rng.choice(NOT_FOUND))]
    reply = rng.choice(ANSWERS).format(a=answer, t=title)
    return used, turns + [chat.say(reply[0].upper() + reply[1:])]


def arithmetic_turns(rng):
    """A calculation done with python (no search), its output really computed."""
    a, b, c = rng.randint(12, 9999), rng.randint(2, 999), rng.randint(2, 99)
    expr, question = rng.choice([
        (f"{a} * {b}", f"What is {a} times {b}?"), (f"{a} + {b} * {c}", f"Compute {a} + {b} * {c}."),
        (f"{a} ** 2", f"What's {a} squared?"), (f"round({a} / {b}, 3)", f"What is {a} divided by {b}?"),
        (f"round({a} * {c} / 100, 2)", f"What is {c}% of {a}?"), (f"{a} % {b}", f"What's the remainder of {a} / {b}?"),
        (f"round({a} ** 0.5, 3)", f"What's the square root of {a}?"), (f"({a} - {b}) * {c}", f"Calculate ({a} - {b}) x {c}."),
    ])
    value = eval(expr)  # (our own expression; print shows str(value))
    return [chat.user(question), chat.tool_call("python", f"print({expr})"), chat.tool_output(f"{value}\n", 0),
            chat.say(rng.choice(["The answer is {v}.", "{v}.", "It's {v}."]).format(v=value))]


def search_doc(item):
    seed, row = item
    rng = random.Random(seed)
    if row is None:
        return chat.render([system_prompt(rng, {"python"})] + arithmetic_turns(rng))
    if (trace := search_trace(_searcher, row)) is None:
        return None
    used, turns = trace
    return chat.render([system_prompt(rng, used), chat.user(row["question"].strip())] + turns)


def qa_items(rows, arithmetic_frac):
    rng, rows = random.Random(0), iter(rows)
    for seed in itertools.count(10**6):
        row = None if rng.random() < arithmetic_frac else next(rows, False)
        if row is False:
            return
        yield seed, row


def search_docs(n_procs=None, arithmetic_frac=0.1, rows=None):
    """Search traces of the QA train questions, with ~arithmetic_frac python arithmetic documents in between. The
    index is read into memory once, before the worker processes fork (they share it)."""
    global _searcher
    import search
    _searcher = search.Searcher.load(search.INDEX_DIR, in_memory=True)
    n_procs = n_procs or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
    with Pool(n_procs) as pool:
        for doc in pool.imap(search_doc, qa_items(rows or qa_rows(), arithmetic_frac), chunksize=16):
            if doc:
                yield doc


# ---------------------------------------------------------------------------------------------------------------------
# weather traces

def c_to_f(c):
    return round(c * 9 / 5 + 32)


def weather_call(city, date=None):
    raw = weather.simulate(city, date)
    arg = f"{city}, {date}" if date else city
    return [chat.tool_call("weather", arg), chat.output(tools.weather_output(raw))], weather.summary(raw)


def describe_day(day, units):
    t = units["temperature"]
    rain = f", {day['precipitation_chance']}% chance of rain" if day["precipitation_chance"] else ""
    return f"{day['conditions']}, {day['low']} to {day['high']}{t}{rain}"


def weather_turns(rng, cities):
    """(user, tool, answer) turns for one templated request about one or two cities."""
    a, b = rng.sample(cities, 2)
    kind = rng.choice(["now", "now", "tomorrow", "rain", "forecast", "fahrenheit", "compare", "followup", "date"])
    if kind == "compare":
        q = rng.choice(["Is it warmer in {a} or {b} right now?", "Which is warmer today, {a} or {b}?",
                        "Compare the weather in {a} and {b}."]).format(a=a, b=b)
        calls_a, sa = weather_call(a)
        calls_b, sb = weather_call(b)
        ta, tb, u = sa["current"]["temperature"], sb["current"]["temperature"], sa["units"]["temperature"]
        warmer = "They are equally warm" if ta == tb else f"{a if ta > tb else b} is warmer"
        return [chat.user(q)] + calls_a + calls_b + [chat.say(
            f"{warmer}: it's {ta}{u} in {a} ({sa['current']['conditions']}) and {tb}{u} in {b} "
            f"({sb['current']['conditions']}).")]
    if kind == "date":
        day = rng.randint(1, 6)
        date = (datetime.date.fromisoformat(weather.TODAY) + datetime.timedelta(days=day)).isoformat()
        q = rng.choice(["What will the weather be in {a} on {d}?", "Weather for {a} on {d}, please."]).format(a=a, d=date)
        calls, s = weather_call(a, date)
        d = next((x for x in s["daily"] if x["date"] == date), s["daily"][0])
        return [chat.user(q)] + calls + [chat.say(f"On {d['date']} in {a}: {describe_day(d, s['units'])}.")]
    calls, s = weather_call(a)
    cur, u = s["current"], s["units"]
    t = u["temperature"]
    if kind == "now":
        q = rng.choice(["What's the weather like in {a}?", "How's the weather in {a} right now?",
                        "weather {a}", "Do I need a jacket in {a} today?", "Tell me the weather in {a}."]).format(a=a)
        return [chat.user(q)] + calls + [chat.say(
            f"It's {cur['temperature']}{t} in {a} right now ({cur['conditions']}), with {cur['humidity']}% humidity "
            f"and wind at {cur['wind']} {u['wind']}.")]
    if kind == "tomorrow":
        q = rng.choice(["What's the weather tomorrow in {a}?", "How will the weather be in {a} tomorrow?"]).format(a=a)
        return [chat.user(q)] + calls + [chat.say(f"Tomorrow in {a}: {describe_day(s['daily'][1], u)}.")]
    if kind == "rain":
        d = s["daily"][1]
        q = rng.choice(["Will it rain in {a} tomorrow?", "Do I need an umbrella in {a} tomorrow?"]).format(a=a)
        p = d["precipitation_chance"]
        verdict = "Yes, probably" if p >= 60 else "Maybe" if p >= 30 else "Probably not"
        return [chat.user(q)] + calls + [chat.say(
            f"{verdict}: there's a {p}% chance of rain in {a} tomorrow ({d['precipitation']} {u['precipitation']} "
            f"expected; {d['conditions']}).")]
    if kind == "forecast":
        q = rng.choice(["What's the forecast for {a} for the next few days?", "3-day forecast for {a}?"]).format(a=a)
        lines = "\n".join(f"- {d['date']}: {describe_day(d, u)}" for d in s["daily"])
        return [chat.user(q)] + calls + [chat.say(f"The forecast for {a}:\n{lines}")]
    if kind == "fahrenheit":
        q = rng.choice(["What's the temperature in {a} in Fahrenheit?", "What's {a}'s temperature in °F?"]).format(a=a)
        c = cur["temperature"]
        if rng.random() < 0.5:  # convert with python
            calls += [chat.tool_call("python", f"print({c} * 9 / 5 + 32)"), chat.tool_output(f"{c * 9 / 5 + 32}\n", 0)]
        return [chat.user(q)] + calls + [chat.say(f"It's {c}{t} in {a}, which is about {c_to_f(c)}°F.")]
    # followup: a second user turn about another city
    q = rng.choice(["What's the weather like in {a}?", "How's the weather in {a}?"]).format(a=a)
    calls_b, sb = weather_call(b)
    return ([chat.user(q)] + calls + [chat.say(f"It's {cur['temperature']}{t} in {a} ({cur['conditions']}).")]
            + [chat.user(rng.choice(["And in {b}?", "What about {b}?", "How about {b}?"]).format(b=b))] + calls_b +
            [chat.say(f"In {b} it's {sb['current']['temperature']}{t} ({sb['current']['conditions']}).")])


def weather_doc(seed):
    rng = random.Random(seed)
    turns = weather_turns(rng, CITIES)
    used = {"weather"} | ({"python"} if any(r == "tool" and x.startswith("python") for r, x in turns) else set())
    return chat.render([system_prompt(rng, used)] + turns)


CITIES = [c for c in WEATHER_CITIES if not rl_split(c) and not eval_city(c)]  # (the others are for RL and evals)


def weather_docs(n_docs=100_000, start_seed=10**6):
    for seed in range(start_seed, start_seed + n_docs):
        yield weather_doc(seed)


# ---------------------------------------------------------------------------------------------------------------------
# no tool needed

def notool_docs(row):
    """A SmolTalk conversation that needs no tool (chit-chat, rewriting or summarizing given text), with tools listed."""
    turns = [(m["role"], m["content"].strip()) for m in row["messages"] if m["role"] in ("user", "assistant")]
    if not turns or any(not text for _, text in turns) or turns[0][0] != "user" or turns[-1][0] != "assistant":
        return
    doc = chat.render([system_prompt(rng_for(turns[0][1]), set())] + turns)
    if len(doc) <= FC_MAX_CHARS:
        yield doc



# ---------------------------------------------------------------------------------------------------------------------
# irrelevance: declared functions that don't fit the request

REFUSALS = ["The query cannot be answered with the provided tools.", "None of the provided functions can do that.",
            "The provided functions can't help with this request."]  # (apigen's irrelevance replies)
STOPWORDS = frozenset("""about above after again also another because been before being below between both could does
doing down during each few from further have having here into just like more most much must only other over same
should some such than that their them then there these they this those through under until very want were what when
where which while will with within would your yours give tell show please need make find know type description
parameters properties string integer number boolean object array required default optional items enum float dict
list name""".split())  # (+ JSON schema keywords)
TOOLISH = ("searc", "wikip", "weath", "forec", "calcu", "compu", "trivi", "encyc", "looku")  # never declared next to
# requests that the built-ins answer (search / weather / python), whatever the words


def stems(text):
    """Content words of a text (camelCase and snake_case split), as 5-letter prefixes."""
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in STOPWORDS}


@functools.cache
def function_pool():
    """Distinct function schemas of apigen-80k (as declared there), minus BFCL's and the tool-ish ones (TOOLISH)."""
    _, bfcl = held_out()
    path = hf_hub_download("HuggingFaceTB/smoltalk", "data/apigen-80k/train-00000-of-00001.parquet", repo_type="dataset")
    pool = {}
    for messages in pq.read_table(path, columns=["messages"]).column("messages").to_pylist():
        for f in json.loads(messages[0]["content"].split("<tools>", 1)[1].split("</tools>", 1)[0]):
            f = schema(f)
            words = stems(json.dumps(f))
            if (f["name"], normalize(f.get("description", ""))) not in bfcl and not words & set(TOOLISH):
                pool.setdefault(f["name"], (f, words))
    return list(pool.values())


def unrelated(rng, request):
    """1-3 function schemas sharing no content word with the request."""
    words, pool, out = stems(request), function_pool(), []
    for _ in range(50):
        f, f_words = rng.choice(pool)
        if not f_words & words and f not in out:
            out.append(f)
        if len(out) == 3 or (out and rng.random() < 0.5):
            break
    return out


SYSTEM_BLOCK = re.compile(r"<\|system_start\|>.*?<\|system_end\|>", re.S)


def redeclared(rng, doc):
    """A rendered document (search / weather traces, no-tool chat) with unrelated functions declared next to the tools
    it lists; its turns are unchanged (the built-ins it uses stay listed)."""
    user = re.search(r"<\|user_start\|>(.*?)<\|user_end\|>", doc, re.S)
    functions = unrelated(rng, user.group(1)) if user else []
    if not functions or not SYSTEM_BLOCK.search(doc):
        return None
    used = set(re.findall(r"<\|tool_start\|>(\w+)\n", doc))
    names = BUILTINS if rng.random() < 0.7 else [t for t in BUILTINS if t in used or rng.random() < 0.5]
    system = system_prompt(rng, set(), functions) if not used else chat.system(chat.tools_prompt(names + functions))
    doc = SYSTEM_BLOCK.sub(lambda _: chat.render([system]), doc, count=1)
    return doc if len(doc) <= FC_MAX_CHARS else None


def refused(row):
    """An apigen request that called a function, with unrelated functions declared instead: a refusal, no call."""
    system, user, reply = (m["content"] for m in row["messages"])
    rng = rng_for(user)
    if "<tool_call>" not in reply or not (functions := unrelated(rng, user)):
        return None
    return fc_doc(rng, functions, [chat.user(user), chat.say(rng.choice(REFUSALS))])


def irrelevant_docs(n_per_kind=10_000):
    """Requests with declared functions that can't serve them: search / weather traces and
    no-tool chat from their rendered train documents (raw/<source>/train, so those sources come first) answered as
    before, and apigen requests that called a function, refused. Interleaved, n_per_kind of each."""
    def shuffled(items):
        return random.Random(0).sample(items, len(items))

    def traces(source):
        path = sorted(glob.glob(os.path.join(BASE_DIR, "raw", source, "train", "*.parquet")))[0]
        for i, doc in enumerate(shuffled(pq.read_table(path).column("text").to_pylist())):
            yield redeclared(random.Random(i), doc)
    path = hf_hub_download("HuggingFaceTB/smoltalk", "data/apigen-80k/train-00000-of-00001.parquet", repo_type="dataset")
    apigen = map(refused, shuffled(pq.read_table(path, columns=["messages"]).to_pylist()))
    kinds = [traces("search_traces"), traces("weather_traces"), traces("notool_chat"), apigen]
    kinds = [itertools.islice(filter(None, docs), n_per_kind) for docs in kinds]
    for docs in itertools.zip_longest(*kinds):
        yield from filter(None, docs)

if __name__ == "__main__":
    print(next(apigen_docs({"messages": [
        {"role": "system", "content": '... <tools>[{"name": "is_prime", "description": "Checks if a number is prime.", '
                                      '"parameters": {"n": {"description": "The number.", "type": "int"}}}]</tools> ...'},
        {"role": "user", "content": "Are 17 and 21 prime?"},
        {"role": "assistant", "content": '<tool_call>[{"name": "is_prime", "arguments": {"n": 17}}, '
                                         '{"name": "is_prime", "arguments": {"n": 21}}]</tool_call>'}]})), end="\n\n")
    print(toolace_calls('[Get Weather(city="Paris, FR", days=3), Other(x=[1, "a)"])]'), end="\n\n")
    print(weather_doc(0), end="\n\n")
    print(chat.render(arithmetic_turns(random.Random(0))))
