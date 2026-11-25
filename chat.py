"""
The chat / tool-use format: how conversations are rendered into documents with the tokenizer's special tokens.

    <|bos|>[<|system_start|>...<|system_end|>]<|user_start|>task<|user_end|>
    <|assistant_start|>optional short text<|tool_start|>bash
    <command><|tool_end|><|output_start|><output, head/tail truncated>
    [exit code N]<|output_end|>...more text, tool calls and outputs...final answer<|assistant_end|>
    [<|user_start|>...more turns...]

- A tool call has the tool's name on its first line, then its input: `bash` / `python` + a command or a program, `search`
  + a query, `fetch` + a Wikipedia title, `weather` + a location, a declared function's name + a JSON object of arguments.
- Its output follows right away. bash / python outputs are truncated to their head and tail (~300-500 tokens) around a
  "[... N lines truncated ...]" marker, with an "[exit code N]" line when the exit code is known (tool_output); the other
  tools' outputs are bounded by the tool (tools.py) and shown as they are (output).
- The available tools can be listed in the system prompt (tools_prompt).
- <|bos|> is added by the tokenizer (Tokenizer.encode_docs), so rendered documents start at the first turn.
- Documents are kept within ~MAX_DOC_CHARS characters, so they fit in a 2048-token training row.
Rendered documents are tokenized with the special tokens allowed (prepare.CHAT_SOURCES); any special token text inside
a message is defused first, so a document never contains a stray <|bos|>.
"""

import json
import re

MAX_DOC_CHARS = 5600  # ~1800 tokens of terminal text (v2 tokenizer: ~3.2 characters per token on these documents)
SPECIAL = re.compile(r"<\|([a-z_0-9]+)\|>")


def defuse(text):
    return SPECIAL.sub(r"<\\|\1|>", text)


def system(text):
    return ("system", text)


def user(text):
    return ("user", text)


def say(text):
    """Assistant text (reasoning before a tool call, or the final answer)."""
    return ("assistant", text)


def tool_call(tool, code):
    return ("tool", f"{tool}\n{code.strip(chr(10))}")


def tool_output(text, exit_code=None, head=900, tail=500):
    """A tool's output, truncated to its first ~head and last ~tail characters."""
    text = truncate(text.rstrip("\n"), head, tail)
    if exit_code is not None:
        text += ("\n" if text else "") + f"[exit code {exit_code}]"
    return ("output", text)


def output(text):
    """A search / fetch / weather / declared function output, as the tool rendered it."""
    return ("output", text)


TOOLS = {"bash": "run a bash command",
         "python": "run a Python program",
         "search": "search Wikipedia; input: a query; output: the 3 best passages",
         "fetch": "read a Wikipedia page; input: its title, or Title#Section",
         "weather": "current weather and a 3-day forecast; input: a place, optionally followed by \", YYYY-MM-DD\""}


def tools_prompt(tools):
    """System prompt text listing the tools: built-in names (TOOLS) one line each, declared functions (JSON schemas
    {name, description, parameters}) one compact JSON line each."""
    lines = ["You can use these tools:"] + [f"- {t}: {TOOLS[t]}" for t in tools if isinstance(t, str)]
    functions = [json.dumps(f, ensure_ascii=False) for f in tools if not isinstance(f, str)]
    if functions:
        lines += ["Functions (input: a JSON object of arguments):"] + functions
    return "\n".join(lines)


def truncate(text, head, tail):
    """Keeps the first ~head and last ~tail characters (cut at line breaks where possible) around a marker."""
    if len(text) <= head + tail + 100:
        return text
    a = text.rfind("\n", 0, head)
    a = a if a > head // 2 else head
    b = text.find("\n", len(text) - tail) + 1
    b = b if 0 < b < len(text) - tail // 2 else len(text) - tail
    n = text.count("\n", a, b)
    return f"{text[:a].rstrip(chr(10))}\n[... {n} lines truncated ...]\n{text[b:]}"


def render(turns):
    """Renders a list of (role, text) turns (see the helpers above) into one document. Consecutive assistant / tool /
    output turns form one assistant message."""
    out, in_assistant = [], False
    for role, text in turns:
        text = defuse(text)
        if role in ("assistant", "tool", "output"):
            if not in_assistant:
                out.append("<|assistant_start|>")
                in_assistant = True
            out.append(text if role == "assistant" else f"<|{role}_start|>{text}<|{role}_end|>")
        else:
            if in_assistant:
                out.append("<|assistant_end|>")
                in_assistant = False
            out.append(f"<|{role}_start|>{text}<|{role}_end|>")
    if in_assistant:
        out.append("<|assistant_end|>")
    return "".join(out)
