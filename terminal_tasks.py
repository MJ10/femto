"""
Terminal task suite for agent_eval.py: small shell tasks, each with starting files, an instruction and a programmatic
check of the final answer and / or the final state of the sandbox. Four tiers:
    1  one command (count, find, copy, mkdir, delete)
    2  a pipeline (unique IPs in an access log, the busiest path, totals of a CSV column, ...)
    3  read, then act (move the largest log, delete the files a list names, act on a config value, ...)
    4  a 2-3 step fix-up (fix a failing python script, make a script executable and run it, clean a CSV, ...)
HELDOUT_GENERATORS: 8 more types that RL never trains on (agent_eval's terminal_new suite), to measure transfer.
Generation is deterministic (tasks(seed)). Words, file names and number ranges are this file's own: disjoint from the
training generators of exec_traces.py and from build_evals.py's eval items.
    python terminal_tasks.py     # prints a few tasks
"""

import dataclasses
import os
import random
import re
from typing import Callable

NAMES = "ada boris celia dmitri esme felix gwen hugo iris jonas kira luca nadia otto pia rufus selma tomas".split()
CITIES = "lyon porto kyoto nairobi lagos zurich denver krakow".split()
MINERALS = "quartz garnet basalt granite obsidian jade opal topaz onyx slate marble pumice".split()
SERVICES = "billing search mailer scheduler gateway storage".split()
URLS = "/index.html /login /api/items /api/cart /static/app.js /about /search /checkout".split()
TEAMS = "infra design research support".split()


@dataclasses.dataclass
class Task:
    name: str
    tier: int
    instruction: str
    files: dict  # {relative path: text}
    check: Callable  # (agent.Episode) -> bool, looking at its answer and its sandbox
    solution: str  # a bash command that solves it (tests/test_agent_eval.py checks the checks with it)
    parts: Callable = None  # (agent.Episode) -> fraction of the task's sub-goals reached, for partial credit in RL


def fraction(checks, gate=lambda ep: True):
    """parts for a Task: the fraction of checks that pass, 0 if gate fails (e.g. a file that must stay untouched)."""
    return lambda ep: sum(bool(c(ep)) for c in checks) / len(checks) if gate(ep) else 0.0


def lines(rows):
    return "".join(f"{row}\n" for row in rows)


def answer_text(ep):
    """The final answer, or else a short last tool output (exec_traces.py documents may end with an output of a few
    lines); nothing if the episode didn't end with <|assistant_end|>."""
    if getattr(ep, "finish", "answer") != "answer":
        return ""
    output = ep.last_output or ""
    return ep.answer or (output if output.strip().count("\n") < 3 else "")


def numbers(text):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]


def mentions(text, word):
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I) is not None


def says_number(value, others=()):
    """The answer contains value (and none of the other candidates)."""
    def close(x, v):
        return abs(x - v) < (1e-9 if float(v).is_integer() else 0.01)

    def check(ep):
        found = numbers(answer_text(ep))
        return any(close(x, value) for x in found) and not any(close(x, o) for x in found for o in others)
    return check


def says_words(words, others=()):
    """The answer mentions every word of words and none of others."""
    return lambda ep: (all(mentions(answer_text(ep), w) for w in words)
                       and not any(mentions(answer_text(ep), o) for o in others))


def is_file(box, path, text):
    return box.read(path) is not None and box.read(path).rstrip("\n") == text.rstrip("\n")


def access_log(rng, n_ips, n_rows):
    ips = [f"{rng.choice([10, 172, 192])}.{rng.randint(0, 40)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
           for _ in range(n_ips)]
    ips = list(dict.fromkeys(ips))
    rows = [(ip, rng.choice(URLS), rng.choice([200, 200, 200, 301, 404, 500])) for ip in ips]
    rows += [(rng.choice(ips), rng.choice(URLS), rng.choice([200, 200, 200, 301, 404, 500]))
             for _ in range(n_rows - len(rows))]
    rng.shuffle(rows)
    text = lines(f'{ip} - - [14/Jun/2025:{9 + i // 60:02d}:{i % 60:02d}:{rng.randint(0, 59):02d} +0000] '
                 f'"GET {url} HTTP/1.1" {status} {rng.randint(120, 9000)}' for i, (ip, url, status) in enumerate(rows))
    return rows, text


def distinct_max(rng, keys, total):
    """Counts per key (summing to ~total) with a unique largest one."""
    while True:
        counts = {k: rng.randint(1, max(2, 2 * total // len(keys))) for k in keys}
        top = sorted(counts.values())
        if top[-1] > top[-2]:
            return counts


# ---------------------------------------------------------------------------------------------------------------------
# tier 1: one command

def count_lines(rng):
    f = rng.choice(["inventory.txt", "catalog.txt", "stock.txt"])
    rows = [f"{rng.choice(MINERALS)} {rng.randint(1, 99)}" for _ in range(rng.randint(8, 40))]
    return Task("count_lines", 1, f"How many lines are in {f}?", {f: lines(rows)}, says_number(len(rows)),
                f"wc -l < {f}")


def count_extension(rng):
    ext, k = rng.choice(["md", "cfg", "yaml"]), rng.randint(2, 7)
    files = {f"notes/{w}.{ext}": f"{w}\n" for w in rng.sample(MINERALS, k)}
    files.update({f"notes/{w}.{rng.choice(['txt', 'bak'])}": f"{w}\n" for w in rng.sample(CITIES, rng.randint(1, 4))})
    return Task("count_extension", 1, f"How many .{ext} files are in the notes directory?", files, says_number(k),
                f"ls notes/*.{ext} | wc -l")


def make_dir(rng):
    d = f"backups/{rng.choice(CITIES)}/{rng.randint(2019, 2026)}"
    return Task("make_dir", 1, f"Create the directory {d}, including any missing parent directories.", {},
                lambda ep: os.path.isdir(os.path.join(ep.sandbox.dir, d)), f"mkdir -p {d}")


def copy_file(rng):
    f = rng.choice(["settings.conf", "app.ini", "server.cfg"])
    text = lines(f"{k} = {rng.randint(1, 500)}" for k in rng.sample(["workers", "retries", "port", "timeout", "depth"], 3))
    return Task("copy_file", 1, f"Make a copy of {f} named {f}.orig in the same directory.", {f: text},
                lambda ep: is_file(ep.sandbox, f + ".orig", text) and is_file(ep.sandbox, f, text), f"cp {f} {f}.orig")


def which_file(rng):
    code = f"{rng.choice(['REF', 'TKT', 'INV'])}-{rng.randint(1000, 9999)}"
    names = rng.sample(MINERALS, 5)
    target = rng.choice(names)
    files = {f"docs/{n}.txt": lines(f"{rng.choice(NAMES)} visited {rng.choice(CITIES)}" for _ in range(4)) for n in names}
    files[f"docs/{target}.txt"] += f"see {code} for details\n"
    return Task("which_file", 1, f"Which file in the docs directory mentions {code}?", files,
                says_words([target], [n for n in names if n != target]), f"grep -l {code} docs/*")


def max_value(rng):
    values = rng.sample(range(-50, 900), rng.randint(10, 25))
    return Task("max_value", 1, "What is the largest value in readings.dat?", {"readings.dat": lines(values)},
                says_number(max(values)), "sort -n readings.dat | tail -n 1")


def count_matches(rng):
    rows = [f"{rng.choice(MINERALS[:5])} {rng.choice(CITIES)}" for _ in range(rng.randint(10, 30))]
    target = rows[rng.randrange(len(rows))].split()[0]
    n = sum(r.split()[0] == target for r in rows)
    return Task("count_matches", 1, f"Each line of samples.txt is a mineral sample and the city it was found in. How "
                f"many {target} samples are there?", {"samples.txt": lines(rows)}, says_number(n),
                f"grep -c '^{target} ' samples.txt")


def delete_tmp(rng):
    tmp = [f"cache/{w}.tmp" for w in rng.sample(MINERALS, rng.randint(2, 6))]
    keep = [f"cache/{w}.{rng.choice(['dat', 'json', 'txt'])}" for w in rng.sample(CITIES, rng.randint(1, 4))]
    files = {p: "data\n" for p in tmp + keep}
    return Task("delete_tmp", 1, "Delete all the .tmp files in the cache directory.", files,
                lambda ep: all(ep.sandbox.read(p) is None for p in tmp) and all(ep.sandbox.read(p) for p in keep),
                "rm cache/*.tmp")


# ---------------------------------------------------------------------------------------------------------------------
# tier 2: a pipeline

def unique_ips(rng):
    rows, text = access_log(rng, rng.randint(3, 12), rng.randint(15, 40))
    return Task("unique_ips", 2, "How many unique IP addresses appear in access.log?", {"access.log": text},
                says_number(len({r[0] for r in rows})), "cut -d' ' -f1 access.log | sort -u | wc -l")


def top_url(rng):
    counts = distinct_max(rng, rng.sample(URLS, rng.randint(3, 6)), 30)
    rows, _ = access_log(rng, 6, 6)
    urls = [u for u, c in counts.items() for _ in range(c)]
    rng.shuffle(urls)
    text = lines(f'{rows[i % 6][0]} - - [14/Jun/2025:10:{i % 60:02d}:00 +0000] "GET {u} HTTP/1.1" 200 {rng.randint(100, 999)}'
                 for i, u in enumerate(urls))
    top = max(counts, key=counts.get)
    return Task("top_url", 2, "Which path was requested most often in access.log?", {"access.log": text},
                says_words([top], [u for u in counts if u != top]),
                "awk '{print $7}' access.log | sort | uniq -c | sort -rn | head -n 1")


def status_count(rng):
    rows, text = access_log(rng, 8, rng.randint(15, 40))
    status = rng.choice([404, 500, 301])
    return Task("status_count", 2, f"How many requests in access.log got the status code {status}?",
                {"access.log": text}, says_number(sum(r[2] == status for r in rows)),
                f"awk '$9 == {status}' access.log | wc -l")


def top_error_service(rng):
    services = rng.sample(SERVICES, 4)
    errors = distinct_max(rng, services, 16)
    rows = [("ERROR", s) for s, c in errors.items() for _ in range(c)]
    rows += [(rng.choice(["INFO", "WARN"]), rng.choice(services)) for _ in range(rng.randint(10, 30))]
    rng.shuffle(rows)
    text = lines(f"2025-04-{1 + i // 10:02d} {8 + i % 10:02d}:{rng.randint(0, 59):02d} {lvl} {s}: "
                 f"{rng.choice(['request done', 'retrying', 'disk check', 'queue drained', 'lost connection'])}"
                 for i, (lvl, s) in enumerate(rows))
    top = max(errors, key=errors.get)
    return Task("top_error_service", 2, "Which service logged the most ERROR lines in system.log?",
                {"system.log": text}, says_words([top], [s for s in services if s != top]),
                "grep ' ERROR ' system.log | awk '{print $4}' | sort | uniq -c | sort -rn | head -n 1")


def csv_total(rng):
    rows = [(f"2025-0{rng.randint(1, 9)}-{rng.randint(10, 28)}", rng.choice(CITIES[:4]), rng.randint(5, 950))
            for _ in range(rng.randint(8, 25))]
    city = rng.choice(rows)[1]
    return Task("csv_total", 2, f"What is the total amount of the {city} sales in sales.csv?",
                {"sales.csv": lines(["date,city,amount"] + [",".join(map(str, r)) for r in rows])},
                says_number(sum(r[2] for r in rows if r[1] == city)),
                f"awk -F, '$2 == \"{city}\" {{s += $3}} END {{print s}}' sales.csv")


def top_three(rng):
    people = rng.sample(NAMES, rng.randint(6, 10))
    salaries = rng.sample(range(41000, 99000, 500), len(people))
    rows = [f"{p},{rng.choice(TEAMS)},{s}" for p, s in zip(people, salaries)]
    top = [p for _, p in sorted(zip(salaries, people), reverse=True)[:3]]
    return Task("top_three", 2, "Who are the three highest-paid people in payroll.csv?",
                {"payroll.csv": lines(["name,team,salary"] + rows)}, says_words(top, [p for p in people if p not in top]),
                "tail -n +2 payroll.csv | sort -t, -k3 -rn | head -n 3 | cut -d, -f1",
                fraction([says_words([p]) for p in top], says_words([], [p for p in people if p not in top])))


def distinct_values(rng):
    rows = [f"{rng.choice(MINERALS)},{rng.choice(CITIES)}" for _ in range(rng.randint(8, 30))]
    return Task("distinct_values", 2, "finds.csv lists mineral finds as mineral,city. How many different minerals "
                "does it list?", {"finds.csv": lines(rows)}, says_number(len({r.split(',')[0] for r in rows})),
                "cut -d, -f1 finds.csv | sort -u | wc -l")


# ---------------------------------------------------------------------------------------------------------------------
# tier 3: read, then act

def move_largest(rng):
    names = rng.sample(SERVICES, rng.randint(3, 5))
    sizes = rng.sample(range(3, 60), len(names))
    files = {f"logs/{n}.log": lines(f"{n} event {i}" for i in range(s)) for n, s in zip(names, sizes)}
    big = max(files, key=lambda p: len(files[p]))
    files[big] += "the end\n"  # the largest in bytes, uniquely
    big = os.path.basename(big)
    files["logs/readme.txt"] = lines("x" * 50 for _ in range(80))  # larger, but not a .log
    others = [p for p in files if p != f"logs/{big}"]
    parts = [lambda ep: is_file(ep.sandbox, f"archive/{big}", files[f"logs/{big}"]),
             lambda ep: ep.sandbox.read(f"logs/{big}") is None]
    untouched = lambda ep: all(is_file(ep.sandbox, p, files[p]) for p in others)
    return Task("move_largest", 3, "Move the largest .log file in logs/ into a new directory named archive/.", files,
                lambda ep: all(c(ep) for c in parts) and untouched(ep),
                "mkdir -p archive && mv $(ls -S logs/*.log | head -n 1) archive/", fraction(parts, untouched))


def delete_listed(rng):
    names = [f"junk/{w}.{rng.choice(['tmp', 'old', 'bak'])}" for w in rng.sample(MINERALS, rng.randint(4, 8))]
    listed = rng.sample(names, rng.randint(1, len(names) - 2))
    files = {p: "old data\n" for p in names}
    files["cleanup.txt"] = lines(listed)
    kept = lambda ep: all(ep.sandbox.read(p) for p in names if p not in listed)
    return Task("delete_listed", 3, "cleanup.txt lists files to delete, one path per line. Delete those files.", files,
                lambda ep: all(ep.sandbox.read(p) is None for p in listed) and kept(ep), "xargs rm < cleanup.txt",
                fraction([lambda ep, p=p: ep.sandbox.read(p) is None for p in listed], kept))


def version_dir(rng):
    version = f"{rng.randint(1, 4)}.{rng.randint(0, 20)}.{rng.randint(0, 30)}"
    return Task("version_dir", 3, "Create a directory named release-<version>, where <version> is the version number "
                "written in the VERSION file.", {"VERSION": version + "\n", "README": "project notes\n"},
                lambda ep: os.path.isdir(os.path.join(ep.sandbox.dir, f"release-{version}")),
                'mkdir "release-$(cat VERSION)"')


def copy_urgent(rng):
    msgs = {f"inbox/msg_{i:02d}.txt": lines([f"from: {rng.choice(NAMES)}", f"about: {rng.choice(MINERALS)} order"])
            for i in rng.sample(range(1, 60), rng.randint(4, 8))}
    urgent = rng.sample(sorted(msgs), rng.randint(1, len(msgs) - 1))
    for p in urgent:
        msgs[p] += "URGENT: reply today\n"

    def check(ep):
        copied = {p for p, t in ep.sandbox.files().items() if p.startswith("urgent/") and t is not None}
        return (copied == {"urgent/" + os.path.basename(p) for p in urgent}
                and all(is_file(ep.sandbox, "urgent/" + os.path.basename(p), msgs[p]) for p in urgent)
                and all(is_file(ep.sandbox, p, t) for p, t in msgs.items()))
    def only_urgent(ep):  # nothing else copied, the inbox untouched
        copied = {p for p, t in ep.sandbox.files().items() if p.startswith("urgent/") and t is not None}
        return (copied <= {"urgent/" + os.path.basename(p) for p in urgent}
                and all(is_file(ep.sandbox, p, t) for p, t in msgs.items()))
    return Task("copy_urgent", 3, "Copy every file in inbox/ that contains the word URGENT into a new directory named "
                "urgent/.", msgs, check, "mkdir urgent && cp $(grep -l URGENT inbox/*) urgent/",
                fraction([lambda ep, p=p: is_file(ep.sandbox, "urgent/" + os.path.basename(p), msgs[p]) for p in urgent],
                         only_urgent))


def rename_extension(rng):
    names = rng.sample(MINERALS, rng.randint(2, 6))
    files = {f"drafts/{n}.txt": f"draft about {n}\n" for n in names}
    files["drafts/index.md"] = "contents\n"
    renamed = [lambda ep, n=n: is_file(ep.sandbox, f"drafts/{n}.md", f"draft about {n}") and
               ep.sandbox.read(f"drafts/{n}.txt") is None for n in names]
    return Task("rename_extension", 3, "Rename every .txt file in drafts/ so that it ends in .md instead of .txt.",
                files, lambda ep: all(c(ep) for c in renamed), 'for f in drafts/*.txt; do mv "$f" "${f%.txt}.md"; done',
                fraction(renamed))


def config_value(rng):
    ports = rng.sample(range(1024, 9999), 3)
    sections = ["server", "database", "cache"]
    rng.shuffle(sections)
    text = "".join(f"[{s}]\nhost = {rng.choice(CITIES)}.internal\nport = {p}\n\n" for s, p in zip(sections, ports))
    target = rng.choice(sections)
    value = ports[sections.index(target)]
    return Task("config_value", 3, f"What port does the [{target}] section of config.ini set?", {"config.ini": text},
                says_number(value, [p for p in ports if p != value]),
                f"awk '/^\\[{target}\\]/ {{s = 1}} s && /^port/ {{print $3; exit}}' config.ini")


def total_to_file(rng):
    counts = [rng.randint(3, 30) for _ in range(rng.randint(2, 5))]
    files = {f"data/{c}.csv": lines(f"{rng.randint(1, 99)},{rng.choice(MINERALS)}" for _ in range(n))
             for c, n in zip(rng.sample(CITIES, len(counts)), counts)}
    files["data/notes.txt"] = lines(["not a csv"] * 7)

    def check(ep):
        found = numbers(ep.sandbox.read("total.txt") or "")
        return bool(found) and found[-1] == sum(counts)
    return Task("total_to_file", 3, "Count the lines of all the .csv files in data/ together and write the total "
                "(just the number) to total.txt.", files, check, "cat data/*.csv | wc -l > total.txt")


# ---------------------------------------------------------------------------------------------------------------------
# tier 4: a 2-3 step fix-up

def fix_script(rng):
    values = [rng.randint(1, 500) for _ in range(rng.randint(4, 12))]
    good = "with open('values.txt') as f:\n    values = [int(line) for line in f]\ntotal = sum(values)\nprint(f'total: {total}')\n"
    bug = rng.choice([
        ("print(f'total: {total}')", "print(f'total: {totl}')"),  # NameError
        ("open('values.txt')", "open('value.txt')"),  # FileNotFoundError
        ("[int(line) for line in f]", "[line.strip() for line in f]"),  # TypeError
        ("total = sum(values)", "total = sum(values"),  # SyntaxError
    ])
    files = {"values.txt": lines(values), "compute.py": good.replace(*bug)}

    def check(ep):
        out, code = ep.sandbox.run("bash", "python3 compute.py")
        return code == 0 and sum(values) in numbers(out) and is_file(ep.sandbox, "values.txt", files["values.txt"])

    def runs(ep):  # partial credit: the script runs (without printing the total)
        return ep.sandbox.run("bash", "python3 compute.py")[1] == 0
    return Task("fix_script", 4, "Running `python3 compute.py` fails. Fix the script so that it prints the total of "
                "the numbers in values.txt.", files, check,
                f"cat > compute.py << 'EOF'\n{good}EOF\npython3 compute.py",
                fraction([runs, check], lambda ep: is_file(ep.sandbox, "values.txt", files["values.txt"])))


def run_script(rng):
    n = rng.randint(3, 40)
    word = rng.choice(MINERALS)
    files = {"report.sh": f'#!/bin/bash\necho "{word} report: $(wc -l < data.txt) rows"\n',
             "data.txt": lines(f"{rng.choice(CITIES)} {rng.randint(1, 9)}" for _ in range(n))}

    parts = [lambda ep: os.access(os.path.join(ep.sandbox.dir, "report.sh"), os.X_OK), says_number(n)]
    return Task("run_script", 4, "Make report.sh executable, run it, and tell me what it prints.", files,
                lambda ep: all(c(ep) for c in parts), "chmod +x report.sh && ./report.sh", fraction(parts))


def clean_csv(rng):
    rows = [(i, rng.choice(MINERALS), rng.randint(1, 300)) for i in range(1, rng.randint(6, 14))]
    bad = rng.randrange(len(rows))
    text_rows = [f"{i},{m},{a}" for i, m, a in rows]
    good_text = lines(["id,item,amount"] + [r for j, r in enumerate(text_rows) if j != bad])
    text_rows[bad] = f"{rows[bad][0]},{rows[bad][1]},{rng.choice(['N/A', 'twelve', '?', 'pending'])}"
    total = sum(a for j, (_, _, a) in enumerate(rows) if j != bad)
    return Task("clean_csv", 4, "orders.csv has one row whose amount is not a number. Delete that row from the file, "
                "then tell me the sum of the amount column.", {"orders.csv": lines(["id,item,amount"] + text_rows)},
                lambda ep: is_file(ep.sandbox, "orders.csv", good_text) and says_number(total)(ep),
                "grep -E '^id,|,[0-9]+$' orders.csv > tmp.csv && mv tmp.csv orders.csv && "
                "awk -F, 'NR > 1 {s += $3} END {print s}' orders.csv",
                fraction([lambda ep: is_file(ep.sandbox, "orders.csv", good_text), says_number(total)]))


def dedupe_sort(rng):
    names = rng.sample(NAMES, rng.randint(4, 10))
    rows = names + rng.choices(names, k=rng.randint(2, 8))
    rng.shuffle(rows)
    return Task("dedupe_sort", 4, "Remove the duplicate lines from guests.txt, sort it alphabetically, and tell me "
                "how many guests are left.", {"guests.txt": lines(rows)},
                lambda ep: is_file(ep.sandbox, "guests.txt", lines(sorted(names))) and says_number(len(names))(ep),
                "sort -u guests.txt -o guests.txt && wc -l < guests.txt",
                fraction([lambda ep: is_file(ep.sandbox, "guests.txt", lines(sorted(names))), says_number(len(names))]))


# ---------------------------------------------------------------------------------------------------------------------
# held-out types: never trained on (rl.py draws only from GENERATORS), to measure transfer to new kinds of tasks. Each
# recombines skills of the training types (wc, sort, cut, find, cp, sed / tr) in a way none of them asks for.

def count_words(rng):
    rows = [" ".join(rng.choice(MINERALS + CITIES) for _ in range(rng.randint(2, 7))) for _ in range(rng.randint(3, 12))]
    return Task("count_words", 1, "How many words are in notes.txt?", {"notes.txt": lines(rows)},
                says_number(sum(len(r.split()) for r in rows), [len(rows)]), "wc -w < notes.txt")


def min_value(rng):
    values = rng.sample(range(-50, 900), rng.randint(10, 25))
    return Task("min_value", 1, "What is the smallest value in levels.dat?", {"levels.dat": lines(values)},
                says_number(min(values), [max(values)]), "sort -n levels.dat | head -n 1")


def delete_empty(rng):
    empty = [f"tmp/{w}.log" for w in rng.sample(MINERALS, rng.randint(2, 5))]
    full = [f"tmp/{w}.log" for w in rng.sample(CITIES, rng.randint(2, 5))]
    files = {**{p: "" for p in empty}, **{p: f"{rng.randint(1, 99)} entries\n" for p in full}}
    return Task("delete_empty", 1, "Delete the empty files in the tmp directory.", files,
                lambda ep: all(ep.sandbox.read(p) is None for p in empty) and all(ep.sandbox.read(p) for p in full),
                "find tmp -type f -empty -delete")


def city_count(rng):
    rows = [f"{rng.choice(MINERALS)},{rng.choice(CITIES)}" for _ in range(rng.randint(8, 30))]
    cities = {r.split(",")[1] for r in rows}
    return Task("city_count", 2, "finds.csv lists mineral finds as mineral,city. In how many different cities were "
                "minerals found?", {"finds.csv": lines(rows)}, says_number(len(cities), [len(rows)]),
                "cut -d, -f2 finds.csv | sort -u | wc -l")


def max_item(rng):
    items = rng.sample(MINERALS, rng.randint(4, 9))
    amounts = rng.sample(range(5, 900), len(items))
    rows = [f"{i + 1},{m},{a}" for i, (m, a) in enumerate(zip(items, amounts))]
    top = items[amounts.index(max(amounts))]
    return Task("max_item", 2, "Which item has the largest amount in orders.csv?",
                {"orders.csv": lines(["id,item,amount"] + rows)}, says_words([top], [m for m in items if m != top]),
                "tail -n +2 orders.csv | sort -t, -k3 -n | tail -n 1 | cut -d, -f2")


def backup_csv(rng):
    csvs = {f"data/{c}.csv": lines(f"{rng.randint(1, 99)},{rng.choice(MINERALS)}" for _ in range(rng.randint(2, 6)))
            for c in rng.sample(CITIES, rng.randint(2, 5))}
    files = {**csvs, **{f"data/{w}.txt": "notes\n" for w in rng.sample(MINERALS, rng.randint(1, 3))}}

    def check(ep):
        backed = {p for p, t in ep.sandbox.files().items() if p.startswith("backup/") and t is not None}
        return (backed == {"backup/" + os.path.basename(p) for p in csvs}
                and all(is_file(ep.sandbox, "backup/" + os.path.basename(p), t) for p, t in csvs.items())
                and all(is_file(ep.sandbox, p, t) for p, t in files.items()))
    return Task("backup_csv", 3, "Copy every .csv file in data/ into a new directory named backup/.", files, check,
                "mkdir backup && cp data/*.csv backup/")


def set_config(rng):
    keys = rng.sample(["workers", "retries", "port", "timeout", "depth"], 3)
    values = [rng.randint(1, 500) for _ in keys]
    key, new = rng.choice(keys), rng.randint(501, 999)
    text = lines(f"{k} = {v}" for k, v in zip(keys, values))
    want = lines(f"{k} = {new if k == key else v}" for k, v in zip(keys, values))
    return Task("set_config", 3, f"In settings.conf, change the value of {key} to {new}.", {"settings.conf": text},
                lambda ep: is_file(ep.sandbox, "settings.conf", want),
                f"sed 's/^{key} = .*/{key} = {new}/' settings.conf > tmp.conf && mv tmp.conf settings.conf")


def lowercase_count(rng):
    names = rng.sample(NAMES, rng.randint(4, 9))
    rows = [n.upper() if rng.random() < 0.5 else n.capitalize() for n in names]
    return Task("lowercase_count", 4, "Convert every line of roster.txt to lowercase, in place, and tell me how many "
                "lines it has.", {"roster.txt": lines(rows)},
                lambda ep: is_file(ep.sandbox, "roster.txt", lines(names)) and says_number(len(names))(ep),
                "tr A-Z a-z < roster.txt > tmp.txt && mv tmp.txt roster.txt && wc -l < roster.txt")


HELDOUT_GENERATORS = [count_words, min_value, delete_empty, city_count, max_item, backup_csv, set_config,
                      lowercase_count]


GENERATORS = [count_lines, count_extension, make_dir, copy_file, which_file, max_value, count_matches, delete_tmp,
              unique_ips, top_url, status_count, top_error_service, csv_total, top_three, distinct_values,
              move_largest, delete_listed, version_dir, copy_urgent, rename_extension, config_value, total_to_file,
              fix_script, run_script, clean_csv, dedupe_sort]


def tasks(seed=0, per_generator=12, generators=GENERATORS):
    """per_generator tasks from each generator (26 x 12 = 312 by default), the same for the same seed."""
    return [gen(random.Random(f"{seed}/{gen.__name__}/{i}")) for gen in generators for i in range(per_generator)]


if __name__ == "__main__":
    for task in tasks()[::40]:
        print(f"--- tier {task.tier} {task.name}: {task.instruction}")
        for path, text in task.files.items():
            print(f"{path}:\n{text[:200]}")
