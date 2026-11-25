"""
Answer and call scoring, shared by the evals (agent_eval.py) and the RL rewards (rl.py), so both score the same way.
- Answers (open-domain QA): SQuAD's normalization, exact match and token F1 (max over the gold answers), and
  has_answer: a gold answer occurs as whole words in a text (DPR's answer recall, e.g. in a search output).
- Function calls: BFCL's AST checker (gorilla bfcl_eval/eval_checker/ast_eval/ast_checker.py), for Python, applied to
  calls in our format (NAME + a JSON object of arguments). A gold call is {name: {param: [allowed values]}}; "" among
  the allowed values means the parameter may be left out. Strings match after BFCL's standardization (no spaces or
  ,./-_*^, lowercase), an int is accepted for a float, and parallel calls match in any order, one to one.
- eval_city: the weather cities held out for eval (training data and RL use the others).
"""

import collections
import hashlib
import json
import re
import string

# ---------------------------------------------------------------------------------------------------------------------
# answers

def normalize_answer(s):
    """SQuAD: lowercase, no punctuation, no articles, single spaces."""
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def exact_match(pred, golds):
    return any(normalize_answer(pred) == normalize_answer(g) for g in golds)


def token_f1(pred, golds):
    def f1(p, g):
        p, g = normalize_answer(p).split(), normalize_answer(g).split()
        common = sum((collections.Counter(p) & collections.Counter(g)).values())
        return 0.0 if common == 0 else 2 * common / (len(p) + len(g))
    return max((f1(pred, g) for g in golds), default=0.0)


def has_answer(text, golds):
    """A gold answer occurs in the text as whole words (after normalization)."""
    text = f" {normalize_answer(text)} "
    return any(n and f" {n} " in text for n in map(normalize_answer, golds))


# ---------------------------------------------------------------------------------------------------------------------
# function calls

TYPES = {"string": str, "integer": int, "float": float, "boolean": bool, "array": list, "tuple": list, "dict": dict}


def parse_args(text):
    """The arguments of a declared-function call (a JSON object), or None."""
    try:
        args = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return args if isinstance(args, dict) else None


def standardize(value):
    if isinstance(value, str):
        return re.sub(r"[ ,./\-_*^]", "", value).lower().replace("'", '"')
    if isinstance(value, list):
        return [standardize(v) for v in value]
    return value


def type_ok(value, schema):
    kind = schema.get("type", "any")
    if kind not in TYPES:
        return True
    if kind == "float" and type(value) is int:
        return True
    if type(value) is not TYPES[kind]:  # (bool is not int)
        return False
    return kind not in ("array", "tuple") or all(type_ok(v, schema.get("items", {})) for v in value)


def dict_ok(value, allowed):
    """A dict argument against BFCL's allowed dicts ({key: [allowed values]} each)."""
    for option in allowed:
        if isinstance(option, dict) and all(k in option and standardize(v) in standardize(option[k])
                                            for k, v in value.items()) \
                and all(k in value or "" in vs for k, vs in option.items()):
            return True
    return False


def call_ok(name, args, gold, function):
    """One call against one gold call {name: {param: [allowed values]}} and the function's schema."""
    gold_name, allowed = next(iter(gold.items()))
    params = function["parameters"]
    if name != gold_name or not isinstance(args, dict) or any(p not in args for p in params.get("required", [])):
        return False
    for param, value in args.items():
        if param not in params["properties"] or param not in allowed or not type_ok(value, params["properties"][param]):
            return False
        if params["properties"][param].get("type") == "float" and type(value) is int:
            value = float(value)
        options = allowed[param]
        if isinstance(value, dict):
            if not dict_ok(value, options):
                return False
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            if not any(isinstance(o, list) and len(o) == len(value) and all(dict_ok(v, [d]) for v, d in zip(value, o))
                       for o in options):
                return False
        elif standardize(value) not in standardize(options):
            return False
    return all(p in args or "" in vs for p, vs in allowed.items())


def call_matches(calls, gold, functions):
    """calls: [(name, args dict or None)] as the model made them; gold: BFCL's ground_truth list; functions: the
    declared schemas. True if the calls match the gold calls one to one, in any order (BFCL's parallel no-order check;
    simple and multiple are the one-call case). Irrelevance (no gold call) is: no calls."""
    by_name = {f["name"]: f for f in functions}
    if len(calls) != len(gold):
        return False
    unmatched = list(range(len(calls)))
    for g in gold:
        name = next(iter(g))
        hit = next((i for i in unmatched if name in by_name and call_ok(*calls[i], g, by_name[name])), None)
        if hit is None:
            return False
        unmatched.remove(hit)
    return True


# ---------------------------------------------------------------------------------------------------------------------
# held-out weather cities

def eval_city(city):
    """Cities held out for the weather eval: 1 in 8, by a hash of the name."""
    return int(hashlib.md5(city.strip().lower().encode()).hexdigest(), 16) % 8 == 0
