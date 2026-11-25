"""
Chat with a model in the terminal. Its tool calls run in a sandbox (tools.py, a fresh directory per conversation) and
are shown with their outputs as it writes them. Works on CPU (a Mac is fine) and GPU.
    python cli.py [--run_name=femto] [--system="You are a helpful assistant."] [--temperature=0]
    python cli.py [--tools=bash,python,search,fetch,weather] [--live] [--functions=functions.json]
--tools: the tools listed in the system prompt (default: all built-ins; --tools=bash,python: no list, as the terminal-only RL runs were
trained); any built-in can be called in any case. search and fetch use the local Wikipedia index (search.py) if there
is one, else live Wikipedia; weather is the simulator, for today's date. --live: live Wikipedia and Open-Meteo.
--functions: a JSON list of function schemas {name, description, parameters}, each optionally with "result", the JSON
its calls return (default {"status": "success"}).
Samples by default (CLI_SAMPLING: temperature 0.7, top-p 0.9, repetition penalty 1.1); --temperature=0 is greedy
(--repetition_penalty=1: no penalty). Repetition penalties are for this CLI only: evals and RL sample from the model's
own distribution.
The model is a released one (femto, the default; femto-mid, femto-base, ...: downloaded from the Hugging Face Hub on
first use into BASE_DIR/runs/<name>/) or a run of yours (BASE_DIR/runs/<run_name>/ckpt.npz), or --ckpt=<path>. Its
tokenizer is BASE_DIR/tokenizers/<name>/ (BASE_DIR = $FEMTO_DIR, else $SCRATCH/femto, else ~/scratch/femto).
In the chat: /reset starts over (new conversation and sandbox), /quit or Ctrl-D exits, Ctrl-C stops a reply.
"""

import argparse
import codecs
import dataclasses
import datetime
import functools
import json
import os
import sys
import time

import jax

import chat
import checkpoint
import search
import tools
import weather
from agent import Agent, Episode
from config import HF_REPO
from engine import Engine, Sampling, Sequence
from tokenizer import Tokenizer
from tools import Sandbox

if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):  # the compiled steps persist across sessions (slurm/common.sh)
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/femto/jax"))
COLOR = sys.stdout.isatty()
STYLE = {"tool": "\033[36m", "output": "\033[2m", "info": "\033[33m", "end": "\033[0m"}


def style(text, kind):
    return f"{STYLE[kind]}{text}{STYLE['end']}" if COLOR else text


class Printer:
    """Streams a sequence's new tokens to the terminal, showing tool calls and outputs as blocks."""

    def __init__(self, tok, seq):
        self.tok, self.seq, self.done = tok, seq, len(seq.tokens)
        self.special = {tok.enc.encode_single_token(s): s for s in tok.enc.special_tokens_set}
        self.utf8, self.kind = codecs.getincrementaldecoder("utf-8")(errors="replace"), None

    def write(self, text):
        sys.stdout.write(style(text, self.kind) if self.kind and text else text)

    def update(self):
        for token in self.seq.tokens[self.done:]:
            name = self.special.get(token)
            if name is None:
                self.write(self.utf8.decode(self.tok.enc.decode_single_token_bytes(token)))
            elif name in ("<|tool_start|>", "<|output_start|>"):
                self.kind = "tool" if name == "<|tool_start|>" else "output"
                self.write("\n$ " if self.kind == "tool" else "\n")
            elif name in ("<|tool_end|>", "<|output_end|>"):
                self.kind = None
                self.write("\n" if name == "<|output_end|>" else "")
            elif name != "<|assistant_end|>":
                self.write(style(name, "info"))
        self.done = len(self.seq.tokens)
        sys.stdout.flush()


def load(args):
    try:
        path = args.ckpt or checkpoint.model_path(args.run_name)
        params, cfg, _ = checkpoint.load_model(path)
        tok = Tokenizer.load(cfg.tokenizer)
    except FileNotFoundError as e:  # (download errors propagate with the Hub's own message)
        sys.exit(f"{e}\nRun one of the released models (--run_name={'|'.join(checkpoint.RELEASED)}, downloaded from "
                 f"huggingface.co/{HF_REPO}), or pass --ckpt=<path to a ckpt.npz or model.safetensors>.")
    dtype = args.dtype or ("float32" if jax.default_backend() == "cpu" else "bfloat16")
    return params, dataclasses.replace(cfg, dtype=dtype), tok


CLI_SAMPLING = Sampling(temperature=0.7, top_p=0.9, repetition_penalty=1.1)  # the decoding that fixed greedy looping


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", default="femto", help="a released model or a run of BASE_DIR/runs")
    parser.add_argument("--ckpt", default="", help="path to a ckpt.npz or model.safetensors (default: from --run_name)")
    parser.add_argument("--system", default="", help="system prompt")
    for field in dataclasses.fields(Sampling):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=getattr(CLI_SAMPLING, field.name),
                            help="see engine.Sampling (temperature 0 = greedy)" if field.name == "temperature" else None)
    parser.add_argument("--max_tokens", type=int, default=1024, help="per generation between tool calls")
    parser.add_argument("--max_calls", type=int, default=10, help="tool calls per reply")
    parser.add_argument("--max_len", type=int, default=0, help="context length (default: the run's seq_len)")
    parser.add_argument("--dtype", default="", choices=["", "bfloat16", "float32"],
                        help="default: float32 on CPU (faster there than bf16, and exact), bfloat16 on GPU")
    parser.add_argument("--no_isolation", action="store_true", help="run tools without the sandbox's isolation")
    parser.add_argument("--tools", default=",".join(tools.BUILTINS), help="the tools to list, comma-separated")
    parser.add_argument("--functions", default="", help="a JSON file of declared functions")
    parser.add_argument("--live", action="store_true", help="live Wikipedia and Open-Meteo")
    parser.add_argument("--seed", type=int, default=int(time.time()))
    args = parser.parse_args()

    names = [t for t in args.tools.split(",") if t]
    functions = json.load(open(args.functions)) if args.functions else []
    handlers = {f["name"]: tools.mock(f.pop("result", {"status": "success"})) for f in functions}
    system = chat.tools_prompt(names + functions) if set(names) != {"bash", "python"} or functions else ""
    system = "\n\n".join(s for s in (system, args.system) if s)
    tools.WEATHER = weather.live if args.live else functools.partial(weather.simulate, today=datetime.date.today())
    if args.live or not os.path.exists(os.path.join(search.INDEX_DIR, "meta.json")):
        tools.SEARCHER = search.LiveWikipedia()

    params, cfg, tok = load(args)
    # fp32 on CPU: one token per call (streams token by token; XLA:CPU's decode loop is no faster there), otherwise 16
    # (bf16 on XLA:CPU upcasts the weights once per call: ~25 ms, paid once per 16 tokens)
    decode_steps = 1 if cfg.dtype == "float32" and jax.default_backend() == "cpu" else 16
    engine = Engine(params, cfg, n_slots=1, max_len=args.max_len or cfg.seq_len, decode_steps=decode_steps,
                    seed=args.seed)
    agent = Agent(engine, tok, Sampling(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Sampling)}))
    assistant_end = tok.enc.encode_single_token("<|assistant_end|>")
    print(style(f"{args.run_name}: {sum(p.size for p in jax.tree.leaves(params)) / 1e6:.0f}M params, "
                f"context {engine.max_len}, {cfg.dtype} on {jax.default_backend()}. /reset, /quit", "info"))
    if {"search", "fetch"} & set(names):
        print(style(f"search / fetch: {'live Wikipedia' if tools.SEARCHER else search.INDEX_DIR}", "info"))
    sandbox, seq = None, None
    while True:
        if sandbox is None:
            sandbox = Sandbox(isolate=not args.no_isolation, functions=handlers)
            print(style(f"tools run in {sandbox.dir} (seen as /tmp/work; isolation: {sandbox.isolation})", "info"))
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message in ("/quit", "/exit"):
            break
        if message == "/reset" or not message:
            if message:
                engine.release(seq) if seq else None
                sandbox.close()
                sandbox, seq = None, None
            continue

        if seq is None:
            seq = Sequence(agent.prompt(([chat.system(system)] if system else []) + [chat.user(message)]))
        else:
            if seq.tokens[-1] != assistant_end:  # the last reply was cut short
                seq.extend([assistant_end])
            seq.extend(agent.encode(chat.render([chat.user(message)]) + "<|assistant_start|>"))
        episode = Episode(seq, sandbox, max_calls=args.max_calls, max_tokens=args.max_tokens)
        printer = Printer(tok, seq)
        try:
            agent.run([episode], on_step=printer.update, release=False)
        except KeyboardInterrupt:
            engine.running.clear()
            episode.finish = "interrupted"
        printer.update()
        if episode.finish != "answer":
            note = "the context is full, /reset to start over" if seq.n_cached + 1 >= engine.max_len else ""
            print(style(f"\n[{episode.finish}{': ' + note if note else ''}]", "info"))
    if sandbox:
        sandbox.close()


if __name__ == "__main__":
    main()
