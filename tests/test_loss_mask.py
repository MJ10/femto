"""The assistant-only loss mask of chat documents (loss_mask="assistant", and the end of "anneal") keeps exactly the
assistant's tokens: its text, tool calls and <|assistant_end|>, not system / user turns or tool outputs. Other documents
keep all their tokens."""

import jax.numpy as jnp
import numpy as np

import chat
from tokenizer import Tokenizer
from train import assistant_mask

TOK = Tokenizer.train(["list the files, one file, sure, you are a helpful assistant"] * 10, vocab_size=320)
IDS = {name: TOK.enc.encode_single_token(name) for name in TOK.enc.special_tokens_set}
CHAT = chat.render([chat.system("You are helpful."), chat.user("list files"), chat.say("Sure."),
                    chat.tool_call("bash", "ls"), chat.tool_output("a.txt", exit_code=0), chat.say("One file."),
                    chat.user("thanks"), chat.say("You're welcome.")])
PLAIN = "plain text, <|user_start|> not special"


def test_assistant_mask():
    docs = TOK.encode_docs([PLAIN], allow_special=False) + TOK.encode_docs([CHAT], allow_special=True)
    tokens = np.array([sum(docs, [])])
    weights = np.asarray(assistant_mask(jnp.asarray(tokens), IDS))[0]
    assert set(weights.tolist()) == {0.0, 1.0}
    assert TOK.decode([t for t, w in zip(tokens[0], weights) if w]) == (
        "<|bos|>" + PLAIN + "Sure.<|tool_start|>bash\nls<|tool_end|>One file.<|assistant_end|>"
        "You're welcome.<|assistant_end|>")
