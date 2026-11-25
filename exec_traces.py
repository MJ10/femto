"""
Synthetic execution traces (prepare.py source `exec_traces`): many small shell and python tasks, actually run in a
throwaway directory, rendered either in the chat / tool format (task -> tool call -> real output -> short answer) or as
a plain terminal / REPL session. They target predicting what commands and code print, which TOOL's predict-the-output
tasks measure (at chance so far).
The generators use their own words, file names and number ranges (not build_evals.py's), and items whose eval-style
query coincides with a TOOL eval item are dropped anyway.
    python exec_traces.py      # prints a few documents
"""

import contextlib
import glob
import io
import itertools
import json
import linecache
import os
import random
import signal
import subprocess
import tempfile
import traceback
from multiprocessing import Pool

import chat
from config import BASE_DIR

ANIMALS = "otter zebra falcon badger lynx heron moose gecko koala panda raven shark tiger walrus yak bison".split()
TOOLS = "hammer wrench saw drill chisel pliers ladder clamp shovel rake trowel sander".split()
PEOPLE = "amir bea chen dana eli fatima gus hana igor jun kofi lena mia noah omar priya".split()
PLACES = "austin bergen dakar quito sapporo turin perth hanoi".split()
DEPTS = "sales ops eng hr legal".split()
SERVICES = "auth db api cache queue web".split()
LEVELS = ["INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]


def to_file(rows):
    return "".join(f"{row}\n" for row in rows)


# ---------------------------------------------------------------------------------------------------------------------
# shell: each generator returns (files, setup commands, [(task, command), ...]) and one of the tasks is used

def sh_words(rng):
    f, words = rng.choice(["animals.txt", "tools.txt", "names.txt"]), rng.sample(ANIMALS + TOOLS, rng.randint(4, 9))
    k = rng.randint(2, 4)
    p = rng.choice(sorted({w[i:i + 2] for w in words for i in range(len(w) - 1)}))
    c = rng.choice(words)[0]
    return {f: to_file(words)}, [], [
        (f"Show the first {k} lines of {f}.", f"head -n {k} {f}"),
        (f"Show the last {k} lines of {f}.", f"tail -n {k} {f}"),
        (f"Sort {f} alphabetically.", f"sort {f}"),
        (f"Sort {f} in reverse alphabetical order.", f"sort -r {f}"),
        (f"Print {f} with the order of its lines reversed.", f"tac {f}"),
        (f"How many lines does {f} have?", f"wc -l < {f}"),
        (f"Print line {k} of {f}.", f"sed -n '{k}p' {f}"),
        (f"Print lines {k} to {k + 2} of {f}.", f"sed -n '{k},{k + 2}p' {f}"),
        (f"Number the lines of {f}.", f"nl {f}"),
        (f"Which lines of {f} contain '{p}'?", f"grep {p} {f}"),
        (f"Which lines of {f} do not contain '{p}'?", f"grep -v {p} {f}"),
        (f"How many lines of {f} contain '{p}'?", f"grep -c {p} {f}"),
        (f"Find '{p}' in {f}, with line numbers.", f"grep -n {p} {f}"),
        (f"Which words in {f} start with '{c}'?", f"grep '^{c}' {f}"),
        (f"Print the length of each word in {f}.", f"awk '{{print length($0)}}' {f}"),
        (f"Print the words in {f} longer than {k + 3} letters.", f"awk 'length($0) > {k + 3}' {f}"),
        (f"Print {f} in uppercase.", f"tr a-z A-Z < {f}"),
        (f"Print the first {k} letters of each line of {f}.", f"cut -c1-{k} {f}"),
        (f"Join the lines of {f} with commas.", f"paste -sd, {f}"),
        (f"Spell each word in {f} backwards.", f"rev {f}"),
        (f"Print every other line of {f}, starting with the first.", f"awk 'NR % 2 == 1' {f}"),
        (f"Print the last line of {f} after sorting it.", f"sort {f} | tail -n 1"),
        (f"How many characters are in {f}?", f"wc -c < {f}"),
        (f"Print the longest word in {f}.", f"awk '{{print length($0), $0}}' {f} | sort -rn | head -n 1 | cut -d' ' -f2"),
        (f"Delete the lines containing '{p}' from {f} and print the rest.", f"sed '/{p}/d' {f}"),
        (f"Replace '{p}' with '__' in {f}.", f"sed 's/{p}/__/g' {f}"),
    ]


def sh_counts(rng):
    f = rng.choice(["votes.txt", "visits.txt", "pets.txt"])
    pool = rng.sample(ANIMALS, rng.randint(2, 4))
    rows = rng.choices(pool, k=rng.randint(5, 10))
    return {f: to_file(rows)}, [], [
        (f"Count how many times each line appears in {f}.", f"sort {f} | uniq -c"),
        (f"List the distinct lines of {f}.", f"sort -u {f}"),
        (f"Which lines appear more than once in {f}?", f"sort {f} | uniq -d"),
        (f"Which lines appear exactly once in {f}?", f"sort {f} | uniq -u"),
        (f"What is the most common line in {f}, with its count?", f"sort {f} | uniq -c | sort -rn | head -n 1"),
        (f"How many distinct lines are in {f}?", f"sort -u {f} | wc -l"),
        (f"Remove adjacent duplicate lines from {f}.", f"uniq {f}"),
        (f"How many times does '{pool[0]}' appear in {f}?", f"grep -c '^{pool[0]}$' {f}"),
    ]


def sh_numbers(rng):
    f, nums = rng.choice(["scores.txt", "sizes.txt", "temps.txt"]), [rng.randint(-20, 400) for _ in range(rng.randint(4, 9))]
    t, k = rng.choice(nums), rng.randint(2, 3)
    return {f: to_file(nums)}, [], [
        (f"Sort the numbers in {f} in ascending order.", f"sort -n {f}"),
        (f"Sort the numbers in {f} from largest to smallest.", f"sort -rn {f}"),
        (f"What is the sum of the numbers in {f}?", f"awk '{{s += $1}} END {{print s}}' {f}"),
        (f"What is the largest number in {f}?", f"sort -n {f} | tail -n 1"),
        (f"What is the smallest number in {f}?", f"sort -n {f} | head -n 1"),
        (f"What is the average of the numbers in {f}?", f"awk '{{s += $1}} END {{print s / NR}}' {f}"),
        (f"How many numbers in {f} are greater than {t}?", f"awk '$1 > {t}' {f} | wc -l"),
        (f"Print the numbers in {f} that are at most {t}.", f"awk '$1 <= {t}' {f}"),
        (f"Show the {k} largest numbers in {f}.", f"sort -rn {f} | head -n {k}"),
        (f"Double every number in {f}.", f"awk '{{print $1 * 2}}' {f}"),
        (f"Print the negative numbers in {f}.", f"grep '^-' {f}"),
        (f"Add up the numbers in {f} with paste and bc.", f"paste -sd+ {f} | bc"),
    ]


def sh_table(rng):
    f, n = rng.choice(["staff.csv", "team.csv", "orders.csv"]), rng.randint(3, 6)
    rows = [f"{p},{rng.choice(DEPTS)},{rng.randint(10, 99)}" for p in rng.sample(PEOPLE, n)]
    dept = rows[rng.randrange(n)].split(",")[1]
    t = sorted(int(r.split(",")[2]) for r in rows)[n // 2]
    return {f: to_file(["name,dept,score"] + rows)}, [], [
        (f"Print the name column of {f}.", f"cut -d, -f1 {f}"),
        (f"Print the dept column of {f} without the header.", f"tail -n +2 {f} | cut -d, -f2"),
        (f"Who is in {dept} according to {f}?", f"grep ',{dept},' {f} | cut -d, -f1"),
        (f"Who has a score of at least {t} in {f}?", f"awk -F, 'NR > 1 && $3 >= {t} {{print $1}}' {f}"),
        (f"Sort the rows of {f} (without its header) by score, highest first.", f"tail -n +2 {f} | sort -t, -k3 -rn"),
        (f"What is the total score in {f}?", f"awk -F, 'NR > 1 {{s += $3}} END {{print s}}' {f}"),
        (f"How many people are in each dept in {f}?", f"tail -n +2 {f} | cut -d, -f2 | sort | uniq -c"),
        (f"How many data rows does {f} have?", f"tail -n +2 {f} | wc -l"),
        (f"Who has the highest score in {f}?", f"tail -n +2 {f} | sort -t, -k3 -n | tail -n 1 | cut -d, -f1"),
        (f"Print name and score from {f}, separated by a tab.", f"awk -F, -v OFS='\\t' '{{print $1, $3}}' {f}"),
        (f"How many columns does {f} have?", f"head -n 1 {f} | tr ',' '\\n' | wc -l"),
    ]


def sh_log(rng):
    f = rng.choice(["app.log", "server.log", "events.log"])
    rows = [f"2025-0{rng.randint(1, 9)}-1{rng.randint(0, 9)} {rng.choice(LEVELS)} {rng.choice(SERVICES)} "
            f"{rng.choice(['started', 'timeout', 'retry', 'ok', 'failed', 'slow'])}" for _ in range(rng.randint(4, 8))]
    rows.sort()
    svc = rows[rng.randrange(len(rows))].split()[2]
    return {f: to_file(rows)}, [], [
        (f"How many ERROR lines are in {f}?", f"grep -c ERROR {f}"),
        (f"Show the ERROR and WARN lines of {f}.", f"grep -E 'ERROR|WARN' {f}"),
        (f"Show {f} without the INFO lines.", f"grep -v INFO {f}"),
        (f"Count the lines of {f} per level.", f"awk '{{print $2}}' {f} | sort | uniq -c"),
        (f"Which services appear in {f}?", f"awk '{{print $3}}' {f} | sort -u"),
        (f"Show the {svc} lines of {f}.", f"grep ' {svc} ' {f}"),
        (f"Print the date of the last entry in {f}.", f"tail -n 1 {f} | cut -d' ' -f1"),
        (f"Print just the messages of {f}.", f"cut -d' ' -f4- {f}"),
    ]


def sh_echo(rng):
    a, b = rng.randint(100, 999), rng.randint(11, 60)
    w, ws = rng.choice(ANIMALS + TOOLS), rng.sample(ANIMALS + PEOPLE, rng.randint(3, 5))
    text, path = " ".join(ws), f"/srv/{rng.choice(PEOPLE)}/{rng.choice(['logs', 'data', 'bin'])}/{rng.choice(TOOLS)}.{rng.choice(['txt', 'csv', 'sh'])}"
    lo, step = rng.randint(1, 20), rng.randint(2, 5)
    return {}, [], [
        (f"Compute {a} + {b} in the shell.", f"echo $(({a} + {b}))"),
        (f"Compute {a} - {b} in the shell.", f"echo $(({a} - {b}))"),
        (f"Compute {a} * {b} in the shell.", f"echo $(({a} * {b}))"),
        (f"Integer-divide {a} by {b} in the shell.", f"echo $(({a} / {b}))"),
        (f"What is {a} mod {b}?", f"echo $(({a} % {b}))"),
        (f"Compute {a} / {b} with two decimals.", f"echo 'scale=2; {a} / {b}' | bc"),
        (f"Spell '{w}' backwards.", f"echo {w} | rev"),
        (f"Uppercase '{w}'.", f"echo {w} | tr a-z A-Z"),
        (f"How many characters does '{w}' have?", f"w={w}; echo ${{#w}}"),
        (f"How many words are in '{text}'?", f"echo {text} | wc -w"),
        (f"Put each word of '{text}' on its own line.", f"echo {text} | tr ' ' '\\n'"),
        (f"Sort the words '{text}'.", f"printf '%s\\n' {text} | sort"),
        (f"Print the second word of '{text}'.", f"echo {text} | cut -d' ' -f2"),
        (f"Print the last word of '{text}'.", f"echo {text} | awk '{{print $NF}}'"),
        (f"Get the file name from {path}.", f"basename {path}"),
        (f"Get the directory of {path}.", f"dirname {path}"),
        (f"Get the extension of {path}.", f"p={path}; echo ${{p##*.}}"),
        (f"Count from {lo} to {lo + 3 * step} in steps of {step}.", f"seq {lo} {step} {lo + 3 * step}"),
        (f"Print the numbers {lo} to {lo + 4} on one line.", f"seq -s ' ' {lo} {lo + 4}"),
        (f"Pad {b} with zeros to 5 digits.", f"printf '%05d\\n' {b}"),
        (f"Factor {a}.", f"factor {a}"),
        (f"Print '{w}' in groups of 2 letters.", f"echo {w} | fold -w2"),
        (f"Replace the letter '{w[1]}' with '*' in '{w}'.", f"echo {w} | sed 's/{w[1]}/*/g'"),
        (f"Is {a} greater than {b * 10}?", f"[ {a} -gt {b * 10} ] && echo yes || echo no"),
        (f"Show {a} in hexadecimal.", f"printf '%x\\n' {a}"),
    ]


def sh_files(rng):
    names = rng.sample(["a.txt", "b.txt", "notes.md", "setup.py", "util.py", "run.sh", "data.json", "out.log"],
                       rng.randint(3, 6))
    gone, other = rng.sample(names, 2)
    d = rng.choice(["build", "tmp", "src", "docs"])
    return {}, [f"touch {' '.join(names)}"], [
        ("List the files here.", "ls"),
        ("How many files are here?", "ls | wc -l"),
        ("List the python files here.", "ls *.py"),
        ("List the text files here.", "ls *.txt"),
        (f"Delete {gone} and list what is left.", f"rm {gone} && ls"),
        (f"Rename {gone} to {d}.bak and list the files.", f"mv {gone} {d}.bak && ls"),
        (f"Create the directory {d}/x and list all directories.", f"mkdir -p {d}/x && find . -type d | sort"),
        (f"Move {other} into a new directory {d} and list {d}.", f"mkdir {d} && mv {other} {d}/ && ls {d}"),
        (f"Show the contents of missing_{other}.", f"cat missing_{other}"),
        (f"Write 'hi' to {other} and count its bytes.", f"echo hi > {other} && wc -c < {other}"),
        (f"Append two lines to {other} and show it.", f"echo one >> {other}; echo two >> {other}; cat {other}"),
        ("Find all files ending in .py, sorted.", "find . -name '*.py' | sort"),
        ("Which files here are empty?", "find . -type f -empty | sort"),
    ]


def sh_two_files(rng):
    a, b = rng.sample(ANIMALS, 5), rng.sample(ANIMALS, 5)
    return {"left.txt": to_file(sorted(a)), "right.txt": to_file(sorted(b))}, [], [
        ("Which lines are in both left.txt and right.txt?", "comm -12 left.txt right.txt"),
        ("Which lines are only in left.txt?", "comm -23 left.txt right.txt"),
        ("Concatenate left.txt and right.txt and sort the result.", "cat left.txt right.txt | sort"),
        ("Count the lines of left.txt and right.txt.", "wc -l left.txt right.txt"),
        ("Show the differences between left.txt and right.txt.", "diff left.txt right.txt"),
        ("Show left.txt and right.txt side by side.", "paste left.txt right.txt"),
        ("How many distinct lines are in left.txt and right.txt together?", "sort -u left.txt right.txt | wc -l"),
    ]


SHELL_GENERATORS = [sh_words, sh_counts, sh_numbers, sh_table, sh_log, sh_echo, sh_files, sh_two_files]


def run_shell(files, setup, command):
    """Runs the setup, then command, in a fresh directory. Returns (combined stdout + stderr, exit code)."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(content)
        script = "".join(f"{cmd} > /dev/null 2>&1\n" for cmd in setup) + command
        env = dict(PATH="/usr/local/bin:/usr/bin:/bin", LC_ALL="C", HOME=tmp)
        try:
            r = subprocess.run(["bash", "-c", script], cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=env, timeout=5)
        except subprocess.TimeoutExpired:
            return None, None
        return r.stdout.replace(tmp, "/tmp/work").rstrip("\n"), r.returncode


def shell_item(rng):
    files, setup, tasks = rng.choice(SHELL_GENERATORS)(rng)
    task, command = rng.choice(tasks)
    steps = [(cmd, *run_shell(files, setup[:i], cmd)) for i, cmd in enumerate(setup)]
    steps += [(f"cat {name}", content.rstrip("\n"), 0) for name, content in files.items()]
    out, code = run_shell(files, setup, command)
    if out is None or "command not found" in out:
        return None
    eval_query = "".join(f"$ {c}\n" + (o + "\n" if o else "") for c, o, _ in steps) + f"$ {command}"
    return dict(task=task, tool="bash", steps=steps + [(command, out, code)], eval_query=eval_query)


# ---------------------------------------------------------------------------------------------------------------------
# python: generators return (setup statements, [expressions]); scripts are whole programs

def py_list(rng):
    x = [rng.randint(-9, 60) for _ in range(rng.randint(4, 8))]
    y, t, i = rng.sample(range(0, 40), 3), rng.choice(x), rng.randrange(len(x))
    return [f"nums = {x}"], [
        "sorted(nums)", "sorted(nums, reverse=True)", "nums[::-1]", "len(nums)", "sum(nums)", "max(nums)", "min(nums)",
        f"nums[{i}]", f"nums[{-1 - i % 3}]", f"nums[{i // 2}:{i // 2 + 3}]", "nums[::2]", "nums[1::2]",
        "[n * 3 for n in nums]", f"[n for n in nums if n > {t}]", f"nums.index({t})", f"nums.count({t})",
        "sum(nums) / len(nums)", "max(nums) - min(nums)", "sorted(nums)[len(nums) // 2]", "[n % 5 for n in nums]",
        f"any(n > {t} for n in nums)", "all(n >= 0 for n in nums)", f"nums + {y}", f"sorted(set(nums) | set({y}))",
        "list(enumerate(nums))[:3]", "[abs(n) for n in nums]", "sum(n for n in nums if n % 2 == 0)",
        f"nums[{len(x) + 2}]", "sorted(nums, key=abs)", "[nums[j + 1] - nums[j] for j in range(len(nums) - 1)]",
    ]


def py_str(rng):
    s = rng.choice(ANIMALS + TOOLS + PEOPLE + PLACES)
    c, k = rng.choice(s), rng.randint(1, 3)
    return [f"word = '{s}'"], [
        "word.upper()", "word.title()", "word[::-1]", "len(word)", f"word.count('{c}')", f"word.find('{c}')",
        f"word.find('z{c}')", f"word.replace('{c}', '-')", f"word[{k}:]", f"word[:{k}]", f"word[-{k}:]", "word[0]",
        "sorted(word)", "''.join(sorted(word))", "len(set(word))", "word * 2", "word.center(10, '.')",
        f"word.startswith('{s[:2]}')", f"word.endswith('{c}')", "list(word)[:3]", "word.swapcase()",
        f"word.split('{c}')", "word + '!' * 3", "word.isalpha()", "ord(word[0])", "[ch for ch in word if ch in 'aeiou']",
        f"word.index('{c}')", f"word[{len(s) + 1}]", "word.zfill(8)", "f'{word:>9}|'",
    ]


def py_text(rng):
    words = rng.sample(ANIMALS + PLACES, rng.randint(3, 6))
    sep = rng.choice([",", ";", "-", " "])
    return [f"line = '{sep.join(words)}'"], [
        f"line.split('{sep}')", f"len(line.split('{sep}'))", f"line.split('{sep}')[-1]", f"line.split('{sep}')[1]",
        f"sorted(line.split('{sep}'))", f"' '.join(line.split('{sep}')).title()", "line.upper()", "len(line)",
        f"[w[0] for w in line.split('{sep}')]", f"max(line.split('{sep}'), key=len)", f"line.replace('{sep}', '/')",
        f"line.split('{sep}', 1)", f"line.rsplit('{sep}', 1)[0]", f"{{w: len(w) for w in line.split('{sep}')[:3]}}",
        f"line.partition('{sep}')",
    ]


def py_dict(rng):
    d = {p: rng.randint(1, 50) for p in rng.sample(PEOPLE, rng.randint(2, 5))}
    k, missing, t = rng.choice(list(d)), rng.choice([p for p in PEOPLE if p not in d]), rng.randint(5, 45)
    return [f"ages = {d}"], [
        f"ages['{k}']", f"ages.get('{missing}', -1)", f"ages.get('{missing}')", "len(ages)", "sorted(ages)",
        "list(ages.values())", "max(ages, key=ages.get)", "min(ages.values())", "sum(ages.values())",
        f"{{n: a for n, a in ages.items() if a > {t}}}", f"'{k}' in ages", f"'{missing}' in ages",
        "sorted(ages.items(), key=lambda kv: kv[1])", "list(ages.items())[0]", f"ages['{missing}']",
        f"ages | {{'{missing}': {t}}}", "{a: n for n, a in ages.items()}", "sorted(ages.values(), reverse=True)[:2]",
    ]


def py_math(rng):
    a, b, n = rng.randint(100, 999), rng.randint(11, 40), rng.randint(3, 9)
    return [], [
        f"{a} + {b} * {n}", f"({a} + {b}) * {n}", f"{a} // {b}", f"{a} % {b}", f"{a} / {b}", f"round({a} / {b}, 2)",
        f"divmod({a}, {b})", f"{n} ** {n % 4 + 2}", f"2 ** {n + 5}", f"abs({b} - {a})", f"-{a} // {b}", f"-{a} % {b}",
        f"bin({a})", f"hex({a})", f"int('{bin(a)[2:]}', 2)", f"len(str({a} ** {n}))", f"sum(range({b}))",
        f"list(range({n}, {n + 4 * b}, {b}))", f"max({a}, {b}, {n})", f"str({b}) + str({n})", f"int('{a}') + {b}",
        f"float({a})", f"{a} == {a}.0", f"{a} > {b} and {b} > {n}", f"pow({n}, {b}, {a})", f"{a} / 0",
        f"round({a} / {n})", f"min([{a}, {b}, {n}])", f"int({a} / {b})", f"f'{{{a * 1000}:,}}'",
    ]


PY_EXPR_GENERATORS = [py_list, py_str, py_text, py_dict, py_math]


def py_script(rng):
    n, k, a, b = rng.randint(3, 7), rng.randint(2, 6), rng.randint(1, 9), rng.randint(10, 25)
    xs = [rng.randint(1, 60) for _ in range(rng.randint(4, 7))]
    w, words = rng.choice(ANIMALS + PLACES), rng.sample(ANIMALS + TOOLS, 5)
    d = {p: rng.randint(1, 9) for p in rng.sample(PEOPLE, 3)}
    return rng.choice([
        f"total = 0\nfor i in range({a}, {b}, {k}):\n    total += i\nprint(total)",
        f"for i in range(1, {n + 1}):\n    print(i, i ** 2)",
        f"for i in range({n}, 0, -1):\n    print('*' * i)",
        f"def area(w, h):\n    return w * h\n\nprint(area({a}, {k}) + area({n}, {n}))",
        f"def g(x):\n    if x % {k} == 0:\n        return 'div'\n    return x\n\nprint([g(i) for i in range(1, {b // 2})])",
        f"n = {b * 7}\nsteps = 0\nwhile n != 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    steps += 1\nprint(steps)",
        f"a, b = 0, 1\nfor _ in range({n + 3}):\n    a, b = b, a + b\nprint(a)",
        f"s = '{w}'\nout = ''\nfor i, c in enumerate(s):\n    out += c.upper() if i % 2 == 0 else c\nprint(out)",
        f"words = {words}\nfor w in sorted(words, key=len):\n    print(len(w), w)",
        f"xs = {xs}\nevens = [x for x in xs if x % 2 == 0]\nprint(len(evens), sum(evens))",
        f"xs = {xs}\nprint(sorted(xs)[-{min(k, 3)}:])\nprint(xs.pop(), len(xs))",
        f"counts = {{}}\nfor c in '{w}{w[::-1]}':\n    counts[c] = counts.get(c, 0) + 1\nprint(sorted(counts.items()))",
        f"stock = {d}\nfor name in sorted(stock):\n    print(f'{{name}}: {{stock[name] * {k}}}')",
        f"for i in range(1, {b}):\n    if i % 3 == 0 and i % 5 == 0:\n        print('fizzbuzz')\n    elif i % 3 == 0:\n"
        f"        print('fizz')\n    elif i % 5 == 0:\n        print('buzz')\n    else:\n        print(i)",
        f"for i in range(1, {min(n, 4) + 1}):\n    print(' '.join(str(i * j) for j in range(1, {min(n, 4) + 1})))",
        f"class Counter:\n    def __init__(self):\n        self.n = {a}\n\n    def step(self):\n        self.n += {k}\n"
        f"        return self.n\n\nc = Counter()\nfor _ in range({n}):\n    c.step()\nprint(c.n)",
        f"try:\n    print({xs}[{len(xs) + k}])\nexcept IndexError:\n    print('out of range')\nfinally:\n    print('done')",
        f"values = {xs}\nprint(values[{len(xs) + 1}])",
        f"x = int('{w}')\nprint(x + 1)",
        f"def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n\nfor i in range({n - 2}, {n + 1}):\n    print(i, fact(i))",
        f"price = {b}.5\nqty = {k}\nprint(f'total: {{price * qty:.2f}}')",
        f"pairs = list(zip({words[:3]}, {xs[:3]}))\nprint(pairs)\nprint(dict(pairs))",
        f"seen = set()\nfor x in {xs + xs[:2]}:\n    if x in seen:\n        print('dup', x)\n    seen.add(x)",
    ])


def run_python(code, expr=None):
    """Runs code (as main.py), then evaluates expr if given. Returns (stdout or repr(expr) or the traceback, exit code)."""
    out, scope = io.StringIO(), {"__name__": "__main__"}
    linecache.cache["main.py"] = (len(code), None, code.splitlines(True), "main.py")
    signal.alarm(2)
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(code, "main.py", "exec"), scope)
            value = eval(compile(expr, "<stdin>", "eval"), scope) if expr else None
        return (repr(value) if expr else out.getvalue().rstrip("\n")), 0
    except Exception as e:
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename in ("main.py", "<stdin>")]
        tb = "Traceback (most recent call last):\n" + "".join(traceback.format_list(frames))
        return (out.getvalue() + tb + "".join(traceback.format_exception_only(e))).rstrip("\n"), 1
    finally:
        signal.alarm(0)


def python_item(rng):
    if rng.random() < 0.5:
        setup, exprs = rng.choice(PY_EXPR_GENERATORS)(rng)
        expr = rng.choice(exprs)
        value, _ = run_python("\n".join(setup), expr)
        printed, code = run_python("\n".join(setup + [f"print({expr})"]))
        given = f"with {' and '.join(setup)}, " if setup else ""
        task = rng.choice([f"In Python, {given}what is `{expr}`?", f"{given}evaluate `{expr}` in Python.",
                           f"What does `{expr}` give in Python{', ' + given[:-2] if given else ''}?"])
        task = task[0].upper() + task[1:]
        eval_query = "".join(f">>> {line}\n" for line in setup) + f">>> {expr}"
        return dict(task=task, tool="python", program="\n".join(setup + [f"print({expr})"]), output=printed, code=code,
                    repl=(setup, expr, value), eval_query=eval_query)
    program = py_script(rng)
    output, code = run_python(program)
    task = rng.choice(["What does this program print?", "Run this script and tell me its output.",
                       "What is the output of the following Python code?"])
    return dict(task=f"{task}\n\n{program}", tool="python", program=program, output=output, code=code,
                eval_query=f"$ cat main.py\n{program}\n$ python3 main.py")


# ---------------------------------------------------------------------------------------------------------------------
# documents

def answer(output):
    lines = output.strip("\n").split("\n") if output.strip() else []
    if not lines:
        return "There is no output."
    return f"The result is {lines[0].strip()}." if len(lines) == 1 and len(lines[0]) <= 80 else None


def chat_turns(item):
    turns = [chat.user(item["task"])]
    if item["tool"] == "bash":
        for cmd, out, code in item["steps"]:
            turns += [chat.tool_call("bash", cmd), chat.tool_output(out, code)]
        final = answer(item["steps"][-1][1])
    else:
        turns += [chat.tool_call("python", item["program"]), chat.tool_output(item["output"], item["code"])]
        final = answer(item["output"])
    return turns + [chat.say(final)] * bool(final)


def plain_text(item):
    """The item as a terminal session (shell, python scripts) or at the python REPL."""
    if item["tool"] == "bash":
        return "".join(f"$ {cmd}\n" + (out + "\n" if out else "") for cmd, out, _ in item["steps"])
    if "repl" in item:
        setup, expr, value = item["repl"]
        return "".join(f">>> {line}\n" for line in setup) + f">>> {expr}\n{value}\n"
    return f"{item['eval_query']}\n{item['output']}\n"


def eval_queries():
    queries = set()
    for path in glob.glob(os.path.join(BASE_DIR, "evals", "tool", "*.jsonl")):
        with open(path) as f:
            queries.update(json.loads(line)["query"] for line in f)
    return queries


def make_doc(seed, held_out=frozenset()):
    """One document: a few items, half the time as a chat (one user task each), else as a plain session."""
    rng = random.Random(seed)
    as_chat, n_items, items = rng.random() < 0.5, rng.randint(2, 8), []
    while len(items) < n_items:
        item = shell_item(rng) if rng.random() < 0.5 else python_item(rng)
        if item and item["eval_query"] not in held_out:
            items.append(item)
    if as_chat:
        doc = chat.render([turn for item in items for turn in chat_turns(item)])
    else:
        doc = "\n".join(plain_text(item) for item in items)
    return doc if len(doc) <= chat.MAX_DOC_CHARS else None


def _make_doc(seed):
    return make_doc(seed, _held_out)


def _init(held_out):
    global _held_out
    _held_out = held_out


def docs(start_seed=10**6, n_procs=None):
    """An endless stream of documents (download it with --max_chars)."""
    held_out = frozenset(eval_queries())
    if not held_out:
        print("warning: no TOOL eval sets found, can't drop items that coincide with them (run build_evals.py first)")
    n_procs = n_procs or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
    with Pool(n_procs, initializer=_init, initargs=(held_out,)) as pool:
        for doc in pool.imap(_make_doc, itertools.count(start_seed), chunksize=64):
            if doc:
                yield doc


if __name__ == "__main__":
    for seed in range(4):
        print(make_doc(seed), end="\n\n" + "=" * 100 + "\n\n")
