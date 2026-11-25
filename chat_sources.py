"""
Sources for the chat mid-training stage (registered in prepare.SOURCES), rendered in the chat format of chat.py:
    smoltalk   HuggingFaceTB/smol-smoltalk conversations; some of those without a system prompt get one of SYSTEM_PROMPTS
    gsm8k      GSM8K train problems solved with one python tool call: the calculator annotations of the reference solution
               become the program (its sentences as comments), the output is really computed, then a short answer.
               Items whose steps or output don't match the reference are dropped.
    tinygsm    TinyGSM problems with their python solutions (docstring removed), run the same way. There are no gold
               answers: items whose program fails or doesn't return a number are dropped.
    mmlu_aux / arc   multiple choice questions (MMLU auxiliary_train, ARC-Easy/Challenge train) answered with the letter
    tool_docs  plain text: --help outputs of command line tools that are not in build_evals.MAN_PAGES (whose man pages
               are the bpb/man eval), the Python documentation and a few GNU manuals, in ~2k-token documents
Programs run in a fresh python process (isolated mode, in a temporary directory, 1GB of memory, 5s timeout).
    python chat_sources.py      # prints a few documents of each kind
"""

import io
import os
import random
import re
import resource
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zlib

import chat
from build_evals import MAN_PAGES

CHAT_MAX_CHARS = 24_000  # ~6-8k tokens: longer conversations are dropped
SYSTEM_PROMPT_FRAC = 0.3
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a helpful, concise assistant.",
    "You are a friendly AI assistant. Answer clearly and accurately.",
    "You are a knowledgeable assistant. Keep your answers short and to the point.",
    "Answer the user's questions as accurately as possible. If you are not sure, say so.",
    "You are a patient tutor who explains things step by step.",
    "You are an assistant that helps with writing, programming and everyday questions.",
    "You are a helpful assistant. Today's conversation is with a curious user.",
]


def rng_for(text):
    """A random generator seeded by the text, so that rendering is deterministic."""
    return random.Random(zlib.crc32(text.encode()))


# ---------------------------------------------------------------------------------------------------------------------
# chat

def smoltalk_docs(row):
    messages = row["messages"]
    turns = [(m["role"], m["content"].strip()) for m in messages if m["role"] in ("system", "user", "assistant")]
    if not turns or any(not text for _, text in turns) or turns[-1][0] != "assistant":
        return
    rng = rng_for(turns[0][1])
    if turns[0][0] != "system" and rng.random() < SYSTEM_PROMPT_FRAC:
        turns.insert(0, chat.system(rng.choice(SYSTEM_PROMPTS)))
    doc = chat.render(turns)
    if len(doc) <= CHAT_MAX_CHARS:
        yield doc


# ---------------------------------------------------------------------------------------------------------------------
# math as python tool calls

def limit_memory():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
    except ValueError:  # macOS
        pass


def run_python(program, timeout=5):
    """Runs a program in a fresh python process. Returns (stdout + stderr, exit code), or (None, None) on a timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "-c", program], cwd=tmp, env={"PATH": "/usr/bin:/bin"},
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
                               preexec_fn=limit_memory)
        except subprocess.TimeoutExpired:
            return None, None
    return p.stdout + p.stderr, p.returncode


def to_number(text):
    try:
        x = float(text.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return None
    return x if abs(x) < 1e15 else None


def fmt(x):
    """72.0 -> '72', 1.8181 -> '1.82'"""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}".rstrip("0").rstrip(".")


def math_doc(question, program, output, answer):
    return chat.render([chat.user(question), chat.tool_call("python", program), chat.tool_output(output, exit_code=0),
                        chat.say(f"The answer is {answer}.")])


CALC = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
CALC_WITH_RESULT = re.compile(r"\s*=?\s*\$?<<[^<>]*>>\$?[\d.,]*\d")  # " = $<<12/60=0.2>>0.2" in "earns 12/60 = $<<...>>0.2"


def gsm8k_program(question, answer):
    """(program, checks) from a GSM8K reference solution: each sentence becomes a comment (without its results), each
    calculation a numbered step, and a number in a calculation that is an earlier step's result (and not a number of
    the question) refers to that step. checks assert that every step computes its annotated result."""
    solution = answer.split("####")[0].strip()
    given = set(re.findall(r"\d[\d.]*", question.replace(",", "")))
    lines, checks, results = [], [], {}  # results: '24' -> 'step1'
    for sentence in solution.split("\n"):
        comment = CALC_WITH_RESULT.sub("", sentence).strip()
        if comment:
            lines.append(f"# {comment}")
        for expr, result in CALC.findall(sentence):
            name = f"step{len(checks) + 1}"
            expr = re.sub(r"\d[\d.]*", lambda m: m.group() if m.group() in given else results.get(m.group(), m.group()),
                          expr.strip())
            lines.append(f"{name} = {expr}")
            checks.append(f"assert abs({name} - ({result.strip()})) < 1e-6")
            results[result.strip()] = name
    if not checks:
        return None, None
    return "\n".join(lines + [f"print({name})"]), "\n".join(checks)


def gsm8k_docs(row):
    gold = to_number(row["answer"].split("####")[-1])
    program, checks = gsm8k_program(row["question"], row["answer"])
    if program is None or gold is None:
        return
    output, code = run_python(f"{program}\n{checks}")
    if code == 0 and (value := to_number(output)) is not None and abs(value - gold) < 1e-6 * max(1, abs(gold)):
        yield math_doc(row["question"].strip(), program, output, fmt(gold))


DOCSTRING = re.compile(r'\n(\s*)("""|\'\'\')[\s\S]*?\2[ \t]*(?=\n)')


def tinygsm_docs(row):
    code = DOCSTRING.sub("", row["code"], count=1).strip()
    match = re.match(r"def (\w+)\(\)", code)
    if not match:
        return
    program = f"{code}\n\n\nprint({match.group(1)}())"
    output, exit_code = run_python(program)
    if exit_code == 0 and (value := to_number(output)) is not None:
        yield math_doc(row["question"].strip(), program, output, fmt(value))


# ---------------------------------------------------------------------------------------------------------------------
# multiple choice

def mc_doc(question, choices, answer):
    """A question with lettered choices; half of the time the user asks for the letter only, else the assistant also
    repeats the choice. The assistant's reply always starts with the letter."""
    letters = "ABCDEFGHIJ"
    question, choices = question.strip(), [c.strip() for c in choices]
    letter_only = rng_for(question).random() < 0.5
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    prompt = f"{question}\n\n{options}" + ("\n\nAnswer with the letter of the correct choice." if letter_only else "")
    reply = letters[answer] if letter_only else f"{letters[answer]}. {choices[answer]}"
    return chat.render([chat.user(prompt), chat.say(reply)])


def mmlu_docs(row):
    row = row.get("train", row)  # auxiliary_train nests its fields under "train"
    yield mc_doc(row["question"], row["choices"], row["answer"])


def arc_docs(row):
    labels = row["choices"]["label"]
    if row["answerKey"] in labels:
        yield mc_doc(row["question"], row["choices"]["text"], labels.index(row["answerKey"]))


# ---------------------------------------------------------------------------------------------------------------------
# plain-text tool documentation

HELP_COMMANDS = """
arch b2sum base32 basenc dircolors dir vdir hostid link pathchk pinky ptx sha224sum sha384sum stdbuf sum users egrep fgrep
zgrep zless zdiff zmore bzcat bzgrep xzcat unxz lzma zstd lz4 pigz cpio locate updatedb sdiff findmnt lsns lslocks lsipc
lsmem losetup nsenter unshare setsid getopt look cal col colrm ul uuidgen mcookie namei chrt prlimit wipefs fallocate
setterm hardlink lsfd mountpoint scriptreplay utmpdump sfdisk partx blockdev fstrim mkswap pmap pwdx pidof slabtop sysctl
skill snice ssh-keyscan ssh-copy-id ssh-add pip pip3 ruby node npm java javac go cargo rustc g++ clang cc cpp cmake
autoconf automake pkg-config ldconfig objcopy strip size addr2line c++filt ranlib gprof valgrind ltrace emacs ed ex
sqlite3 psql mysql redis-cli docker podman apptainer singularity kubectl conda uv sbatch squeue scancel sinfo srun salloc
sacct scontrol nvidia-smi htop iotop nmap tcpdump iptables nslookup whois ifconfig route arp ethtool mtr iperf3 socat
telnet ftp lftp aria2c yq xmllint xsltproc pandoc convert ffmpeg gs pdftotext qpdf units parallel ncdu fd rg ag fzf
inotifywait systemd-run systemd-analyze loginctl timedatectl localectl busctl resolvectl networkctl update-alternatives
dpkg dpkg-query apt apt-get apt-cache yum dnf rpm snap flatpak visudo chage userdel groupdel gpasswd getfacl setfacl
chattr lsattr fuser expect dialog whiptail dos2unix unix2dos enscript lp lpr datamash csvcut mlr xmlstarlet aspell shasum
pv rclone gh hg svn zipinfo zipgrep funzip unrar 7z bsdtar mksquashfs gcov tini sponge ts chronic ifne
""".split()
GIT_SUBCOMMANDS = """
bisect blame cherry-pick revert show worktree archive clean describe mv rm restore switch shortlog submodule reflog notes
apply format-patch am cherry ls-files rev-parse rev-list cat-file show-ref for-each-ref gc fsck bundle sparse-checkout
range-diff difftool mergetool maintenance
""".split()
PYTHON_DOCS = "https://docs.python.org/3.13/archives/python-3.13-docs-text.tar.bz2"
PYTHON_DOC_DIRS = ("library", "tutorial", "howto", "reference", "faq", "using", "glossary.txt")
GNU_MANUALS = ["sed/manual/sed.txt", "grep/manual/grep.txt", "make/manual/make.txt", "gawk/manual/gawk.txt",
               "findutils/manual/find.txt", "tar/manual/tar.txt"]
DOC_CHARS = 6000


def split_doc(title, text, max_chars=DOC_CHARS):
    """Splits a long text at blank lines into documents of at most ~max_chars; the later ones start with the title."""
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text.strip()):
        for i in range(0, len(para), max_chars):  # a paragraph longer than max_chars is cut
            piece = para[i:i + max_chars]
            if cur and len(cur) + len(piece) + 2 > max_chars:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n\n{piece}" if cur else piece
    parts.append(cur)
    return [p if i == 0 else f"{title} (continued)\n\n{p}" for i, p in enumerate(parts) if p.strip()]


def command_help(argv, timeout=5):
    """The output of a --help style invocation, run in an empty temporary directory, or None."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {"PATH": os.environ["PATH"], "HOME": tmp, "LC_ALL": "C", "COLUMNS": "80", "TERM": "dumb", "PAGER": "cat"}
        try:
            p = subprocess.run(argv, cwd=tmp, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout, errors="replace")
        except (OSError, subprocess.TimeoutExpired):
            return None
    out = (p.stdout or p.stderr).strip()
    usage = re.search(r"usage", out, re.I) or len(re.findall(r"^\s+-", out, re.M)) >= 3
    return out if usage and out.count("\n") >= 3 else None


def help_docs():
    """'$ <command> --help' and its output, for the installed commands of HELP_COMMANDS and git subcommands."""
    for name in dict.fromkeys(HELP_COMMANDS):
        if name in MAN_PAGES:
            continue
        for flag in ("--help", "-h"):
            if (out := command_help([name, flag])) is not None:
                yield from split_doc(f"$ {name} {flag}", f"$ {name} {flag}\n{out}")
                break
    for sub in GIT_SUBCOMMANDS:
        if f"git-{sub}" not in MAN_PAGES and (out := command_help(["git", sub, "-h"])) is not None:
            yield f"$ git {sub} -h\n{out}"


def fetch(url, attempts=5):
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(10 * 2**i)


def fetch_or_skip(url):
    """fetch(), but a source that stays unreachable is skipped (with a warning) instead of failing the whole download."""
    try:
        return fetch(url)
    except OSError as e:
        print(f"skipping {url}: {e}")
        return None


def python_docs():
    """The Python documentation (library reference, tutorial, howtos, language reference, FAQ) as plain text."""
    if (archive := fetch_or_skip(PYTHON_DOCS)) is None:
        return
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:bz2") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            path = member.name.split("/", 1)[-1]
            if member.isfile() and path.endswith(".txt") and path.startswith(PYTHON_DOC_DIRS):
                text = tar.extractfile(member).read().decode()
                yield from split_doc(f"Python documentation: {path.removesuffix('.txt')}", text)


def gnu_manual_docs():
    for path in GNU_MANUALS:
        if (text := fetch_or_skip(f"https://www.gnu.org/software/{path}")) is None:
            continue
        text = text.decode(errors="replace")
        yield from split_doc(f"GNU {os.path.basename(path).removesuffix('.txt')} manual", text)


def tool_docs():
    yield from help_docs()
    yield from python_docs()
    yield from gnu_manual_docs()


if __name__ == "__main__":
    print(next(smoltalk_docs({"messages": [{"role": "user", "content": "Hi! What's 2+2?"},
                                           {"role": "assistant", "content": "2 + 2 = 4."}]})), end="\n\n")
    gsm = {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                       "How many clips did Natalia sell altogether in April and May?",
           "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips "
                     "altogether in April and May.\n#### 72"}
    print(next(gsm8k_docs(gsm)), end="\n\n")
    tiny = {"question": "Mark has 10 crayons. He gives 2 to his sister and loses 4. How many does he have left?",
            "code": 'def simple_math_problem() -> int:\n    """\n    Mark has 10 crayons.\n    """\n    total = 10\n'
                    '    result = total - 2 - 4\n\n    return result\n'}
    print(next(tinygsm_docs(tiny)), end="\n\n")
    print(next(arc_docs({"question": "Which factor will most likely cause a person to develop a fever?",
                         "choices": {"text": ["a leg muscle relaxing after exercise", "a bacterial population in the "
                                              "bloodstream", "several viral particles on the skin",
                                              "carbohydrates being digested in the stomach"],
                                     "label": ["A", "B", "C", "D"]}, "answerKey": "B"})), end="\n\n")
    print(next(help_docs())[:500])
