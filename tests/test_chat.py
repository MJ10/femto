"""The chat format renders as documented, and terminal screens split into per-command outputs."""

import chat
from prepare import command_outputs


def test_render():
    doc = chat.render([chat.user("list files <|bos|>"), chat.say("Sure."), chat.tool_call("bash", "ls\n"),
                       chat.tool_output("a.txt\n", exit_code=0), chat.say("One file.")])
    assert doc == ("<|user_start|>list files <\\|bos|><|user_end|><|assistant_start|>Sure.<|tool_start|>bash\nls"
                   "<|tool_end|><|output_start|>a.txt\n[exit code 0]<|output_end|>One file.<|assistant_end|>")


def test_truncate():
    text = "\n".join(f"line {i}" for i in range(1000))
    short = chat.truncate(text, 200, 100)
    assert short.startswith("line 0\n") and short.endswith("line 999") and "lines truncated ..." in short
    assert len(short) < 400


def test_command_outputs():
    screen = ("old outputroot@host:/app# echo hi\nhi\nroot@host:/app# cat > f << 'EOF'\n> a\n> EOF\n"
              "root@host:/app# printf x\nxroot@host:/app# ")
    assert command_outputs(screen, ["echo hi\n", "cat > f << 'EOF'\na\nEOF\n", "printf x\n"]) == ["hi\n", "", "x"]
    assert command_outputs(screen, ["echo bye\n"]) is None
