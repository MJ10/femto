"""
The agent loop. The model writes its reply in the chat format of chat.py; when it closes a tool call (<|tool_end|>),
the call runs in the episode's sandbox (tools.py) and its output is appended exactly as the training documents render
it (chat.tool_output: head/tail truncation, "[exit code N]"; special-token text defused), then generation continues,
until <|assistant_end|> or a budget runs out. Many episodes run at once on one Engine; tool calls run in a thread pool
while the others keep generating. Given several engines (one per device), an Agent splits the episodes between them,
one thread per engine.

Token in, token out (for RL): an episode's Sequence holds the exact token ids, never decoded and re-encoded: the prompt
and tool outputs (mask 0) and the sampled tokens with their log probabilities (mask 1); see Episode.
"""

import concurrent.futures

import chat
from engine import Sampling, Sequence

TOOLS = ("bash", "python")  # without a sandbox


class Episode:
    """One assistant reply. seq: the conversation so far (a list of tokens, or a Sequence to continue), ending in
    <|assistant_start|>; sandbox: where its tool calls run. After Agent.run:
        finish   "answer" (<|assistant_end|>), "length" (a generation ran past max_tokens or the context is full),
                 "max_calls" (one more call than allowed), "malformed" (another special token, e.g. <|user_start|>)
        calls    [{tool, code, output, exit_code}], tool None for a call that isn't a tool's name + input (code);
                 exit_code None for tools other than bash / python; the call that exceeds max_calls is recorded
                 with output None
        reply    the reply's text with its special tokens; answer: its text after the last tool output
        seq.tokens / seq.logprobs / seq.mask    the token-level transcript for training"""

    def __init__(self, seq, sandbox, max_calls=8, max_tokens=512):
        self.seq = seq if isinstance(seq, Sequence) else Sequence(seq, max_tokens)
        self.seq.max_tokens = max_tokens
        self.sandbox, self.max_calls, self.start = sandbox, max_calls, len(self.seq.tokens)
        self.calls, self.finish, self.reply = [], None, ""
        self.names = sandbox.tool_names() if sandbox else TOOLS

    @property
    def answer(self):
        return self.reply.rsplit("<|output_end|>", 1)[-1].removesuffix("<|assistant_end|>").strip()

    @property
    def last_output(self):
        outputs = [c["output"] for c in self.calls if c["output"] is not None]
        return outputs[-1] if outputs else None


class Agent:
    def __init__(self, engine, tok, sampling=Sampling(), workers=16):
        """engine: an Engine, or a list of them; workers: threads running tool calls (per engine)."""
        self.engines = engine if isinstance(engine, list) else [engine]
        self.engine, self.tok, self.sampling, self.workers = self.engines[0], tok, sampling, workers
        self.special = {name: tok.enc.encode_single_token(name) for name in tok.enc.special_tokens_set}
        stop = {i for name, i in self.special.items() if name != "<|tool_start|>"}  # any other special token stops
        self.stop = frozenset(stop)

    def prompt(self, turns):
        """Tokens of a conversation (chat.py turns) followed by <|assistant_start|>, as the training data has them."""
        return self.tok.encode_docs([chat.render(turns) + "<|assistant_start|>"], allow_special=True)[0]

    def encode(self, text):
        """Rendered chat text (special tokens allowed) -> token ids, e.g. to append the next user turn."""
        return self.tok.enc.encode(text, allowed_special="all")

    def output_tokens(self, output, exit_code):
        text = chat.tool_output(output, exit_code)[1] if exit_code is not None else output  # (others are final)
        return self.encode(f"<|output_start|>{chat.defuse(text)}<|output_end|>")

    def parse_call(self, reply, names=TOOLS):
        """The tool call that the reply's last token (<|tool_end|>) closes: {tool, code}; tool None if malformed or
        not one of names."""
        tokens = reply[:-1]
        starts = [i for i, t in enumerate(tokens) if t == self.special["<|tool_start|>"]]
        ends = [i for i, t in enumerate(tokens) if t == self.special["<|output_end|>"]]
        if not starts or (ends and ends[-1] > starts[-1]):
            return dict(tool=None, code="")
        name, _, code = self.tok.decode(tokens[starts[-1] + 1:]).partition("\n")
        return dict(tool=name if name in names and code.strip() else None, code=code)

    def call(self, ep, call):
        """Runs a call (in a worker thread) and returns the output's tokens."""
        if call["tool"] is None:
            names = f"{', '.join(ep.names[:-1])} and {ep.names[-1]}"
            call.update(output=f"error: not a tool call (the tools are {names}, e.g. <|tool_start|>bash\\nls<|tool_end|>)",
                        exit_code=2)
        else:
            call["output"], call["exit_code"] = ep.sandbox.run(call["tool"], call["code"])
        return self.output_tokens(call["output"], call["exit_code"])

    def run(self, episodes, on_step=None, release=True):
        """Runs the episodes to the end. on_step() is called after every engine step (e.g. to stream text). release:
        free the episodes' cache slots at the end (False: keep them, to continue the conversation; one engine only).
        With several engines, each runs a contiguous part of the episodes (so an RL group mostly shares its prompt's
        prefill)."""
        if len(self.engines) == 1:
            return self._run(self.engine, episodes, on_step, release)
        assert release, "continuing a conversation needs one engine"
        size = -(-len(episodes) // len(self.engines))
        with concurrent.futures.ThreadPoolExecutor(len(self.engines)) as pool:
            for f in [pool.submit(self._run, engine, episodes[i * size:(i + 1) * size], on_step, release)
                      for i, engine in enumerate(self.engines)]:
                f.result()
        return episodes

    def _run(self, engine, episodes, on_step, release):
        by_seq, pending = {id(ep.seq): ep for ep in episodes}, {}

        def finish(ep, reason):
            ep.finish, ep.reply = reason, self.tok.decode(ep.seq.tokens[ep.start:])
            if release:
                engine.release(ep.seq)

        def add(ep):
            ep.seq.stop = self.stop
            if engine.add(ep.seq).done:  # doesn't fit in the cache any more
                finish(ep, "length")

        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            for ep in episodes:
                add(ep)
            while engine.has_work() or pending:
                for seq in engine.step(self.sampling) if engine.has_work() else []:
                    ep, last = by_seq[id(seq)], seq.tokens[-1]
                    if seq.finish_reason == "length":
                        finish(ep, "length")
                    elif last == self.special["<|assistant_end|>"]:
                        finish(ep, "answer")
                    elif last != self.special["<|tool_end|>"]:
                        finish(ep, "malformed")
                    elif len(ep.calls) >= ep.max_calls:  # recorded, not run
                        ep.calls.append(dict(self.parse_call(seq.tokens[ep.start:], ep.names), output=None, exit_code=None))
                        finish(ep, "max_calls")
                    else:
                        ep.calls.append(self.parse_call(seq.tokens[ep.start:], ep.names))
                        pending[pool.submit(self.call, ep, ep.calls[-1])] = ep
                if pending:  # wait for a tool only when there is nothing to generate
                    done, _ = concurrent.futures.wait(pending, timeout=0 if engine.has_work() else None,
                                                      return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        ep = pending.pop(future)
                        ep.seq.extend(future.result())
                        add(ep)
                if on_step:
                    on_step()
        return episodes
