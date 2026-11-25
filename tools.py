"""
The tools and a simple sandbox to run them in: bash and python (executed), search and fetch (Wikipedia, search.py),
weather (weather.py) and declared functions (name -> handler(args dict) -> JSON-able result, e.g. mock(result)).
Sandbox.run(tool, code) runs any of them; only bash / python have an exit code. The other tools' outputs are final
(bounded here, ~400 tokens at most) and shown as they are; errors are outputs too.
Backends (module-level, so training, evals and the CLI share them): SEARCHER, the local Wikipedia index loaded on first
use (or search.LiveWikipedia), and WEATHER, the seeded simulator (or weather.live). Search and fetch outputs are
memoized per backend and query (the RL rollouts of one prompt often make the same query, at the same time).
- Each Sandbox is a fresh temporary working directory (one per episode or chat session), seen by the tools as
  /tmp/work (the path in exec_traces.py's outputs); files persist across its tool calls.
- Every call is a new process with a timeout, a minimal environment (exec_traces.py's: PATH, LC_ALL=C, HOME = the work
  dir), no stdin, and ulimits on CPU time, file size and memory. The process group is killed afterwards.
- Isolation, best effort: on Linux, a user + mount + pid + network namespace (unshare, no root needed) in which the
  user-writable trees (/home, /network, /tmp, ...) are hidden under empty tmpfs mounts, so nothing outside the work dir
  can be changed, other processes can't be seen, and there is no network. On macOS, sandbox-exec denies writes outside
  the work dir and the network. Otherwise (or with isolate=False) commands just run in the work dir.
- bash: stdout and stderr interleaved (as exec_traces.py); python: `python3 -c`, stdout then stderr (as chat_sources.py).
Outputs are rendered for the model by chat.tool_output (head/tail truncation, "[exit code N]"), as in the training data.
    python tools.py      # checks the isolation
"""

import collections
import concurrent.futures
import functools
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

import chat
import weather

BUILTINS = ("bash", "python", "search", "fetch", "weather")
# the python tool's interpreter. On macOS, the one running this: /usr/bin/python3 is Xcode's shim, which under the
# sandbox prints xcrun cache errors and leaves a Library/ directory in the work dir
PYTHON = sys.executable if platform.system() == "Darwin" else "python3"
SEARCHER = None  # a search.Searcher (default: search.INDEX_DIR, loaded on first use) or a search.LiveWikipedia
WEATHER = weather.simulate  # (location, date) -> Open-Meteo forecast JSON
_lock = threading.Lock()
WORKDIR = "/tmp/work"
TIMEOUT = 10  # seconds per call
LIMITS = "ulimit -t {cpu} -f 65536 -c 0; ulimit -v 4194304 2>/dev/null; "  # CPU s, 64MB files, no core, 4GB memory
HIDDEN = ["/home", "/network", "/tmp", "/var/tmp", "/dev/shm", "/run/user", "/scratch", "/localscratch", "/project",
          "/projects", "/checkpoint", "/lustre", "/nfs", "/Tmp", "/mnt", "/media"]


def hidden_dirs():
    """Trees to hide in the namespace: the usual user-writable ones plus the top directories of $HOME, $SCRATCH, ..."""
    tops = {"/" + os.path.realpath(p).split("/")[1] for p in (os.path.expanduser("~"), tempfile.gettempdir(),
                                                              os.environ.get("SCRATCH", "/"), os.environ.get("SLURM_TMPDIR", "/"))}
    return [d for d in dict.fromkeys(HIDDEN + sorted(tops)) if d not in ("/", "/usr", "/etc", "/proc", "/dev")
            and os.path.isdir(d) and not os.path.islink(d)]


@functools.cache
@functools.cache
def isolation():
    """'namespace' (Linux), 'sandbox-exec' (macOS) or None, whichever works here (checked once)."""
    if platform.system() == "Linux" and shutil.which("unshare"):
        ok = subprocess.run(["unshare", "-rmnpf", "--mount-proc", "bash", "-c", "mount -t tmpfs none /tmp"],
                            capture_output=True).returncode == 0
        if ok:
            return "namespace"
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    return None


def macos_profile(workdir):
    return f"""(version 1) (allow default) (deny network*) (deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/zero") (regex #"^/dev/tty") (regex #"^/dev/fd/"))"""


def searcher():
    global SEARCHER
    with _lock:
        if SEARCHER is None:
            import search
            SEARCHER = search.Searcher.load(search.INDEX_DIR)
    return SEARCHER


def search_output(results):
    """[(title, passage)] -> "[1] Title\npassage" blocks separated by blank lines."""
    return "\n\n".join(f"[{i}] {title}\n{passage}" for i, (title, passage) in enumerate(results, 1)) or "no results"


def fetch_output(page):
    return chat.truncate(page, 1400, 200) if page else "no such page"


def weather_output(raw):
    return json.dumps(weather.summary(raw), ensure_ascii=False)


def function_output(result):
    return chat.truncate(json.dumps(result, ensure_ascii=False), 1100, 300)


def mock(result):
    """A declared function's handler that always returns result."""
    return lambda args: result


MEMO_SIZE = 10000
_memo, _memo_lock = collections.OrderedDict(), threading.Lock()


def memo(key, fn):
    """fn() memoized under key (LRU); threads asking for a key being computed wait for that result."""
    with _memo_lock:
        future = _memo.get(key)
        owner = future is None
        if owner:
            future = _memo[key] = concurrent.futures.Future()
            if len(_memo) > MEMO_SIZE:
                _memo.popitem(last=False)
        else:
            _memo.move_to_end(key)
    if owner:
        try:
            future.set_result(fn())
        except Exception as e:
            future.set_exception(e)
    return future.result()


def call(tool, code, functions=None):
    """Output of a search / fetch / weather / declared function call."""
    try:
        if tool in ("search", "fetch"):
            s, query = searcher(), code.strip()
            return memo((tool, s, query), lambda: search_output(s.search(query, 3)) if tool == "search" else
                        fetch_output(s.page(query)))
        if tool == "weather":
            return weather_output(WEATHER(*weather.parse(code)))
        args = json.loads(code)
        if not isinstance(args, dict):
            raise ValueError("the arguments must be a JSON object")
        return function_output(functions[tool](args))
    except Exception as e:  # shown to the model
        error = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
        return f"error: {error}" if tool in ("search", "fetch") else json.dumps({"error": error})


def wait(p, timeout):
    """p's exit code, or 124 once it has run for timeout seconds (its process group is then killed). A blocking wait,
    so it returns as soon as p exits: Popen.wait(timeout) polls with sleeps that double up to 50 ms (a 40 ms process
    took 63 ms; a sandboxed `ls` 65 ms, ~115 ms under the rollouts' load)."""
    expired = threading.Event()

    def kill():
        expired.set()
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    timer = threading.Timer(timeout, kill)
    timer.start()
    code = p.wait()
    timer.cancel()
    return 124 if expired.is_set() else code


class Sandbox:
    def __init__(self, files=None, setup="", timeout=TIMEOUT, isolate=True, functions=None):
        """files: {relative path: text, bytes, or None for a directory} to start with; setup: a bash command run first;
        functions: the declared functions {name: handler(args dict) -> JSON-able result}."""
        self.dir = os.path.realpath(tempfile.mkdtemp(prefix="sandbox_"))
        self.timeout, self.isolation = timeout, isolation() if isolate else None
        self.functions = functions or {}
        for path, content in (files or {}).items():
            self.write(path, content)
        if setup:
            self.run("bash", setup)

    @property
    def workdir(self):
        """The work dir's path as the tools see it."""
        return WORKDIR if self.isolation == "namespace" else self.dir

    def write(self, path, content):
        path = os.path.join(self.dir, path)
        if content is None:
            return os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb" if isinstance(content, bytes) else "w") as f:
            f.write(content)

    def read(self, path):
        """Text of a (regular) file in the work dir, or None. Opening a FIFO the commands made would block forever."""
        path = os.path.join(self.dir, path)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, errors="replace") as f:
                return f.read()
        except (OSError, ValueError):
            return None

    def files(self):
        """{relative path: text} of everything under the work dir (None for directories), to check its final state."""
        return {path: text for path, (_, text) in self.state().items()}

    def state(self):
        """{relative path: (permission bits, text or None for a directory or symlink target)}."""
        out = {}
        for root, dirs, names in os.walk(self.dir):
            for name in dirs + names:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, self.dir)
                if os.path.islink(path):
                    out[rel] = (0, "-> " + os.readlink(path))
                else:
                    out[rel] = (os.stat(path).st_mode & 0o7777, None if os.path.isdir(path) else self.read(rel))
        return out

    def command(self, argv):
        """argv wrapped in the limits and the isolation."""
        limits = LIMITS.format(cpu=self.timeout + 1)
        if self.isolation == "namespace":
            mounts = "".join(f"mount -t tmpfs none {d}; " for d in hidden_dirs())
            setup = f"set -e; {mounts}mkdir -p {WORKDIR}; mount --bind /proc/self/cwd {WORKDIR}; cd {WORKDIR}; set +e; "
            return ["unshare", "-rmnpf", "--mount-proc", "bash", "-c", setup + limits + 'exec "$@"', "sandbox", *argv]
        if self.isolation == "sandbox-exec":
            return ["sandbox-exec", "-p", macos_profile(self.dir), "bash", "-c", limits + 'exec "$@"', "sandbox", *argv]
        return ["bash", "-c", limits + 'exec "$@"', "sandbox", *argv]

    def tool_names(self):
        return list(BUILTINS) + [f for f in self.functions if f not in BUILTINS]

    def run(self, tool, code):
        """Runs a bash command or a python program: (output, exit code), 124 on a timeout (as `timeout`). Any other
        tool: (output, None)."""
        if tool not in ("bash", "python"):
            return call(tool, code, self.functions), None
        if "\0" in code:  # can't go into argv (the model can sample a null byte)
            return f"{tool}: the code contains a null byte", 2
        argv = {"bash": ["bash", "-c", code], "python": [PYTHON, "-c", code]}[tool]
        env = dict(PATH="/usr/local/bin:/usr/bin:/bin", LC_ALL="C", HOME=self.workdir)
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            p = subprocess.Popen(self.command(argv), cwd=self.dir, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=out if tool == "bash" else err, start_new_session=True)
            code = wait(p, self.timeout)
            try:
                os.killpg(p.pid, signal.SIGKILL)  # also whatever it left running in the background
            except ProcessLookupError:
                pass
            p.wait()
            text = ""
            for f in (out, err):
                f.seek(0)
                text += f.read().decode(errors="replace")
        code = 128 - code if code < 0 else code  # killed by a signal
        return text.replace(self.dir, WORKDIR), code

    def close(self):
        # the commands may have removed permissions: top-down, so every directory (not a symlink) is made accessible
        # before the walk enters it; a leftover temp dir is better than a crashed RL run (a command that nests a
        # directory in itself makes paths too long to chmod: ENAMETOOLONG killed vfg4_grpo at step 54)
        def chmod(path):
            try:
                if not os.path.islink(path):
                    os.chmod(path, 0o700)
            except OSError:
                pass

        chmod(self.dir)
        for root, dirs, _ in os.walk(self.dir):  # top-down: the chmod comes before the walk enters the directory
            for d in dirs:
                chmod(os.path.join(root, d))
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    print(f"isolation: {isolation()}, hidden: {hidden_dirs() if isolation() == 'namespace' else '-'}")
    with Sandbox({"a.txt": "hello\n"}) as box:
        for tool, code in [("bash", "pwd; ls; cat a.txt; echo $HOME; id -un"), ("bash", f"touch {os.path.expanduser('~')}/x"),
                           ("bash", "curl -sS -m 3 https://example.com | head -c 80"), ("bash", "sleep 30"),
                           ("bash", "ps aux | wc -l"), ("python", "import os; print(os.getcwd()); 1/0"),
                           ("bash", "yes | head -c 100000000 > big; ls -l big")]:
            print(f"$ {code}\n{box.run(tool, code)}")
