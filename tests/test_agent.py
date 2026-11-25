"""An agent-loop transcript of an episode is token for token the training document of the same episode
(exec_traces.py's chat rendering): same tool output rendering, truncation, exit code line and special tokens."""

import random

import chat
import exec_traces
from agent import Agent, Episode
from tokenizer import SPECIAL_TOKENS, Tokenizer
from tools import Sandbox


class ReplayEngine:
    """Stands in for engine.Engine: 'samples' the next token of a reference document (teacher forcing)."""

    def __init__(self, reference):
        self.reference, self.running = reference, []

    def add(self, seq):
        seq.finish_reason = None
        self.running.append(seq)
        return seq

    def has_work(self):
        return bool(self.running)

    def release(self, seq):
        pass

    def step(self, sampling):
        for seq in self.running:
            seq.tokens.append(self.reference[len(seq.tokens)])
            seq.logprobs.append(0.0)
            seq.finish_reason = "stop" if seq.tokens[-1] in seq.stop else None
        finished = [s for s in self.running if s.done]
        self.running = [s for s in self.running if not s.done]
        return finished


def shell_episode(seed):
    """exec_traces.shell_item, keeping the files it starts from."""
    rng = random.Random(seed)
    files, setup, tasks = rng.choice(exec_traces.SHELL_GENERATORS)(rng)
    task, command = rng.choice(tasks)
    steps = [(cmd, *exec_traces.run_shell(files, setup[:i], cmd)) for i, cmd in enumerate(setup)]
    steps += [(f"cat {name}", content.rstrip("\n"), 0) for name, content in files.items()]
    steps.append((command, *exec_traces.run_shell(files, setup, command)))
    return dict(task=task, tool="bash", steps=steps), files


def python_episode(seed):
    rng = random.Random(seed)
    while (item := exec_traces.python_item(rng))["code"] != 0:  # tracebacks differ: exec_traces runs code in-process
        pass
    return item, {}


def test_transcript_tokenizes_as_training_document():
    text = open(exec_traces.__file__).read() + open(chat.__file__).read()
    tok = Tokenizer.train([text] * 3, vocab_size=len(SPECIAL_TOKENS) + 512)
    for seed in range(12):
        item, files = shell_episode(seed) if seed % 3 else python_episode(seed)
        turns = exec_traces.chat_turns(item)
        document = tok.encode_docs([chat.render(turns)], allow_special=True)[0]
        agent = Agent(ReplayEngine(document), tok)
        with Sandbox(files) as box:
            ep = agent.run([Episode(agent.prompt(turns[:1]), box)])[0]
        assert ep.finish == "answer" and ep.seq.tokens == document, (item["task"], ep.reply)
        n_out = sum(len(agent.output_tokens(c["output"], c["exit_code"])) for c in ep.calls)
        assert sum(ep.seq.mask) == len(document) - ep.start - n_out  # tool outputs are not the model's tokens
