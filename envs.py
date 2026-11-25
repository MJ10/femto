"""
RL environments for the web / function tools, used by rl.py's task mix next to terminal and
gsm8k. Every item is an rl.Prompt: a system prompt listing the tools, a user question, the episode's declared functions
and a reward function (agent.Episode) -> [0, 1]. Answers are scored with tool-evals' scoring.py, so RL and the evals
agree. Train items only, disjoint from the evals: QA questions in tool_sources.rl_split (the slice tool-data's traces
skip; tool-evals uses the validation splits), weather cities in rl_split and not scoring.eval_city, mock functions of
our own (not BFCL's), weather / funcall template seeds >= 1000.
- search   tasks search_nq / search_triviaqa / search_hotpotqa: NQ-open, TriviaQA (rc.nocontext) and HotpotQA
           (distractor, easy + medium) train questions, to look up with search / fetch. Task search_context (1 in 4):
           SQuAD v1.1 train, the paragraph in the prompt, answerable without searching.
- funcall  programmatic mock APIs declared in the system prompt (the target plus 0-2 distractors); a templated request,
           a gold call {name: {param: [allowed values]}} (BFCL format) and a value of the mock's response that the
           answer must state; actions (send_email, set_alarm, ...) answer only {"status": "success"}, and the reply
           confirms with the request's value. Tasks funcall_<api>, and funcall_none (30%): a request no declared
           function serves (simple arithmetic: answer it, with python if wanted, without calling a declared function;
           a refusal gets half the reward, so "full" rates (reward 1) count the right answers); funcall_unserved (15%):
           a request for one of the APIs that isn't declared (BFCL's irrelevance): a reply without a declared call;
           half the prompts list only the functions (funcall_unserved_only, as BFCL: no call at all), half the
           built-ins too (funcall_unserved: a built-in call is fine).
- nl2sh    a shell task in words, one bash command (scored by running it, as agent_eval.nl2sh).
- files    create a file with a given line of text (the CLI's "notes.txt" request), and tell its size in bytes.
- chat     SmolTalk test-split requests that need no tool (small talk, creative writing, advice, rewriting, ...), with
           the tools listed.
- weather  templated questions about the simulator's (weather.py) data: the current temperature, rain on a day of the
           forecast, which of two cities is warmer; the gold value comes from the same simulator the tool calls.
Rewards. The answer is the text after the last tool output, and only counts if the episode ended with it, it has at
most MAX_ANSWER_WORDS words and no "{" (no pasted passage or JSON output); otherwise it is "" (rl.py's --require_answer
also gives 0 to an episode that doesn't end with an answer).
    answer_score   1 if the answer is (after SQuAD normalization) a gold answer, else 0.5 x [a gold answer occurs in
                   it] + 0.5 x token F1. So a short sentence with the answer gets ~0.6-0.8, the bare answer 1, a list of
                   guesses less than the right one alone.
    search         answer_score x (1 - CALL_PENALTY x calls beyond FREE_CALLS); search_context: x NO_SEARCH_FACTOR if
                   it called a tool at all.
    funcall        0.5 x [the declared-function calls match the gold call(s), scoring.call_matches] + 0.5 x [that, and
                   the answer states the response's value and none of the other candidates (restaurants, showtimes,
                   statuses)]; any other call (a built-in, a retry) fails the call match; funcall_none: [no
                   declared function called] x [a finished reply] x (0.5 + 0.5 x [the answer is right]).
    chat           [no tool call] x [a finished reply] x min(1, 0.5 x content-word F1 with the reference / CHAT_F1_PASS).
    weather        [the stated value is right] x [every city of the question was looked up with the weather tool]
                   x (1 - CALL_PENALTY x calls beyond FREE_CALLS); temperatures within 1 degree, among at most two
                   numbers; rain: yes / no (half the questions are rainy days); warmer: the warmer city mentioned
                   first, no "colder" / "cooler". (Without the lookup condition, guessing pays: "warmer" half the time.)
"pass" (rl.py's pass1 / passG, the gate) = reward >= 0.5: a correct answer (search), the right call (funcall).
    python envs.py                                       # prints a few items of each env
    python envs.py --gate --run_name=rl_web7            # pass@1 / pass@16 per env and tier (T=1), one GPU
"""

import argparse
import collections
import dataclasses
import datetime
import functools
import json
import os
import random
import re

import chat
import scoring
from terminal_tasks import is_file, mentions, numbers

MAX_ANSWER_WORDS = 50  # (a retrieved passage is ~100)
FREE_CALLS = 3  # tool calls per episode without a penalty
CALL_PENALTY = 0.1  # per call beyond FREE_CALLS (multiplicative)
NO_SEARCH_FACTOR = 0.5  # context tier: the paragraph was given, searching was not needed
REPEAT_FACTOR = 0.5  # search / weather: per repeat of an identical call
NONE_FRAC = 0.3  # funcall items that no declared function serves (arithmetic)
UNSERVED_FRAC = 0.15  # funcall items asking for one of our APIs that isn't declared (BFCL's irrelevance)
CHAT_F1_PASS = 0.1  # chat: content_f1 that counts as a pass (reward 0.5); mt_tool20c: pass@1 .46, pass@16 .68
SEED0 = 1000  # template seeds of the train items (evals use seeds < 1000)
BUILTINS = ["bash", "python", "search", "fetch", "weather"]


def system_prompt(functions=()):
    return chat.tools_prompt(BUILTINS + list(functions))


def load_search_index():
    """The Wikipedia index in RAM (~5 GB): no page faults on network storage while hundreds of episodes search."""
    import search
    import tools
    tools.SEARCHER = search.Searcher.load(search.INDEX_DIR, in_memory=True)


def rl_split(text):
    import tool_sources
    return tool_sources.rl_split(text)


def answer_score(text, golds):
    if not text:
        return 0.0
    if scoring.exact_match(text, golds):
        return 1.0
    return 0.5 * scoring.has_answer(text, golds) + 0.5 * scoring.token_f1(text, golds)


def call_factor(ep):
    """x (1 - CALL_PENALTY) per call beyond FREE_CALLS, and x REPEAT_FACTOR per repeat of an identical call."""
    repeats = len(ep.calls) - len({(c["tool"], c["code"].strip()) for c in ep.calls})
    return max(0.0, 1 - CALL_PENALTY * max(0, len(ep.calls) - FREE_CALLS)) * REPEAT_FACTOR ** repeats


def plain(ep):
    """The final answer, if the episode ended with one that isn't pasted JSON and has at most MAX_ANSWER_WORDS words."""
    text = ep.answer if ep.finish == "answer" else ""
    return text if "{" not in text and len(text.split()) <= MAX_ANSWER_WORDS else ""


# ---------------------------------------------------------------------------------------------------------------------
# search QA

def qa_rows():
    """(source, question, gold answers) of the RL slice of NQ-open, TriviaQA and HotpotQA train."""
    import pyarrow.parquet as pq
    from build_evals import hf_file
    nq = pq.read_table(hf_file("google-research-datasets/nq_open", "nq_open/train-00000-of-00001.parquet")).to_pylist()
    tqa = pq.read_table(hf_file("mandarjoshi/trivia_qa", "rc.nocontext/train-00000-of-00001.parquet"),
                        columns=["question", "answer"]).to_pylist()
    hotpot = pq.read_table(hf_file("hotpotqa/hotpot_qa", "distractor/train-00000-of-00002.parquet"),
                           columns=["question", "answer", "level"]).to_pylist()
    rows = [("nq", r["question"], list(r["answer"])) for r in nq]
    rows += [("triviaqa", r["question"], [r["answer"]["value"]] + list(r["answer"]["aliases"])) for r in tqa]
    rows += [("hotpotqa", r["question"], [r["answer"]]) for r in hotpot if r["level"] != "hard"]
    held_out = eval_questions()
    return [(s, q.strip(), golds) for s, q, golds in rows
            if rl_split(q) and scoring.normalize_answer(q) not in held_out]


def eval_questions():
    """Normalized questions of tool-evals' QA eval sets (BASE_DIR/evals/web/*.jsonl), to drop from RL."""
    import glob
    from config import BASE_DIR
    out = set()
    for path in glob.glob(os.path.join(BASE_DIR, "evals", "web", "*.jsonl")):
        with open(path) as f:
            out |= {scoring.normalize_answer(json.loads(line).get("question", "")) for line in f}
    return out


def squad_rows():
    import pyarrow.parquet as pq
    from build_evals import hf_file
    rows = pq.read_table(hf_file("rajpurkar/squad", "plain_text/train-00000-of-00001.parquet")).to_pylist()
    return [(r["context"].strip(), r["question"].strip(), list(r["answers"]["text"])) for r in rows
            if len(r["context"]) < 1500]


def search_pool():
    from rl import Prompt
    system = system_prompt()

    def qa_check(golds, handlers=()):  # (a call to a declared function: 0)
        return lambda ep: 0.0 if any(c["tool"] in handlers for c in ep.calls) else \
            answer_score(plain(ep), golds) * call_factor(ep)

    def with_functions(i, s, q, golds):  # 1 in 4: 1-2 of our (unrelated) mock APIs declared next to the built-ins
        names = random.Random(f"search/{i}").sample(sorted(APIS), 1 + i % 2)
        handlers = {n: APIS[n].respond for n in names}
        return Prompt(f"search/{s}/{i}", "search_with_functions", q, {}, qa_check(golds, handlers),
                      system_prompt([APIS[n].schema for n in names]), handlers)

    def context_check(golds):
        return lambda ep: answer_score(plain(ep), golds) * (NO_SEARCH_FACTOR if ep.calls else 1.0)
    pool = [with_functions(i, s, q, golds) if i % 4 == 3 else Prompt(f"search/{s}/{i}", f"search_{s}", q, {},
                                                                      qa_check(golds), system)
            for i, (s, q, golds) in enumerate(qa_rows())]
    squad = [Prompt(f"search/squad/{i}", "search_context", f"{ctx}\n\n{q}", {}, context_check(golds), system)
             for i, (ctx, q, golds) in enumerate(squad_rows())]
    return pool + squad[:len(pool) // 3]  # ~1 in 4 of the search items needs no search


# ---------------------------------------------------------------------------------------------------------------------
# function calling: mock APIs

CURRENCIES = {"USD": "US dollars", "EUR": "euros", "GBP": "British pounds", "JPY": "Japanese yen",
              "CAD": "Canadian dollars", "CHF": "Swiss francs", "INR": "Indian rupees", "MXN": "Mexican pesos",
              "AUD": "Australian dollars"}
USD_VALUE = {"USD": 1, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0064, "CAD": 0.73, "CHF": 1.12, "INR": 0.012, "MXN": 0.055,
             "AUD": 0.66}
TICKERS = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "NVDA": "Nvidia", "TSLA": "Tesla", "NFLX": "Netflix",
           "INTC": "Intel", "DIS": "Disney", "NKE": "Nike", "KO": "Coca-Cola"}
AIRLINES = "AC LH AF BA KL UA DL QR EK SK".split()
TOWNS = "Lyon Porto Kyoto Nairobi Lagos Zurich Denver Krakow Lisbon Oslo Austin Seville Dublin Montreal".split()
CUISINES = "Italian Thai Mexican Indian Japanese Ethiopian Lebanese Greek".split()
WORDS = "Lupo Verde Sol Luna Rosa Azul Nido Olivo Fuego Mar Brisa Cedro".split()
MOVIES = ["The Silent Harbor", "Night Train to Oslo", "Paper Moons", "The Last Glacier", "Copper Sky", "Small Hours"]
UNITS = {"km": "kilometers", "mi": "miles", "kg": "kilograms", "lb": "pounds", "m": "meters", "ft": "feet",
         "l": "liters", "gal": "US gallons"}
UNIT_FACTORS = {("km", "mi"): 0.621371, ("mi", "km"): 1.609344, ("kg", "lb"): 2.204623, ("lb", "kg"): 0.453592,
                ("m", "ft"): 3.280840, ("ft", "m"): 0.3048, ("l", "gal"): 0.264172, ("gal", "l"): 3.785412}


def fn(name, description, required, **params):
    """A declared function's schema in BFCL's typing (string / integer / float / boolean)."""
    props = {p: dict(type=t, description=d) for p, (t, d) in params.items()}
    return dict(name=name, description=description, parameters=dict(type="dict", properties=props, required=required))


def seeded(*key):
    return random.Random("/".join(map(str, key)))


def says_value(text, value):
    """The answer states a number (within 0.5%, or exactly for integers) or a string (as whole words)."""
    if isinstance(value, str):
        return mentions(text, value)
    tol = 1e-9 if float(value).is_integer() else max(0.01, abs(value) * 0.005)
    return any(abs(x - value) <= tol for x in numbers(text))


@dataclasses.dataclass
class Api:
    schema: dict
    make: object  # rng -> (request, gold args {param: [allowed values]}, key of the response the answer states; for an
    #               action: the request's value the confirmation states)
    respond: object  # args dict -> response dict (deterministic in the args)
    action: bool = False  # an action whose response is only STATUS (send an email, set an alarm, ...)


def fx_make(rng):
    a, b = rng.sample(sorted(CURRENCIES), 2)
    amount = rng.choice([rng.randint(5, 2000), round(rng.uniform(1, 500), 2)])
    request = rng.choice([f"How much is {amount} {CURRENCIES[a]} in {CURRENCIES[b]}?",
                          f"Convert {amount} {a} to {b}.", f"What do I get for {amount} {a} in {CURRENCIES[b]}?"])
    return request, dict(amount=[amount], source=[a], target=[b]), "converted"


def fx_respond(args):
    a, b = str(args.get("source")).upper(), str(args.get("target")).upper()
    if a not in USD_VALUE or b not in USD_VALUE:
        return dict(error=f"unknown currency; supported: {', '.join(CURRENCIES)}")
    rate = round(USD_VALUE[a] / USD_VALUE[b] * seeded("fx", a, b).uniform(0.97, 1.03), 4)
    return dict(source=a, target=b, rate=rate, converted=round(float(args.get("amount") or 0) * rate, 2))


def quote_make(rng):
    t = rng.choice(sorted(TICKERS))
    request = rng.choice([f"What is {TICKERS[t]}'s stock trading at right now?", f"Get me the current price of {t}.",
                          f"How much is one share of {TICKERS[t]} ({t}) today?"])
    return request, dict(symbol=[t]), "price"


def quote_respond(args):
    rng = seeded("quote", str(args.get("symbol")).upper())
    return dict(symbol=args.get("symbol"), price=round(rng.uniform(8, 900), 2),
                change_percent=round(rng.uniform(-4, 4), 2), currency="USD")


def flight_make(rng):
    code, number = rng.choice(AIRLINES), rng.randint(10, 999)
    flight, key = f"{code}{number}", rng.choice(["status", "gate", "departure_time"])
    request = {"status": f"Is flight {code} {number} on time?",
               "gate": f"Which gate does flight {code}{number} leave from?",
               "departure_time": f"When does {code} {number} depart?"}[key]
    return request, dict(flight=[flight, f"{code} {number}"]), key


def flight_respond(args):
    rng = seeded("flight", re.sub(r"\s", "", str(args.get("flight"))).upper())
    return dict(flight=args.get("flight"), status=rng.choice(["on time", "on time", "delayed", "cancelled"]),
                departure_time=f"{rng.randint(5, 22):02d}:{rng.choice(range(0, 60, 5)):02d}",
                gate=f"{rng.choice('ABCDEF')}{rng.randint(1, 40)}")


def parcel_make(rng):
    tracking = f"{rng.choice(['1Z', 'PK', 'TR'])}{rng.randint(10**7, 10**8 - 1)}"
    key = rng.choice(["location", "estimated_delivery"])
    request = {"location": f"Where is my package {tracking} right now?",
               "estimated_delivery": f"When will parcel {tracking} arrive?"}[key]
    return request, dict(tracking_number=[tracking]), key


def parcel_respond(args):
    rng = seeded("parcel", str(args.get("tracking_number")).upper())
    return dict(tracking_number=args.get("tracking_number"), status="in transit", location=rng.choice(TOWNS),
                estimated_delivery=f"2025-06-{rng.randint(15, 28)}")


def time_make(rng):
    city = rng.choice(TOWNS)
    return rng.choice([f"What time is it in {city} right now?", f"What's the local time in {city}?"]), \
        dict(city=[city]), "local_time"


def time_respond(args):
    rng = seeded("time", str(args.get("city")).lower())
    return dict(city=args.get("city"), local_time=f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                utc_offset=rng.choice(["-05:00", "+00:00", "+01:00", "+02:00", "+09:00", "+03:00"]))


def restaurants_make(rng):
    city, cuisine = rng.choice(TOWNS), rng.choice(CUISINES)
    request = rng.choice([f"Find the best-rated {cuisine} restaurant in {city}.",
                          f"Which {cuisine} place in {city} has the highest rating?"])
    return request, dict(city=[city], cuisine=[cuisine]), "best"


def restaurants_respond(args):
    rng = seeded("restaurants", str(args.get("city")).lower(), str(args.get("cuisine")).lower())
    names = [f"{w} {rng.choice(['Kitchen', 'House', 'Table', 'Garden', 'Corner'])}" for w in rng.sample(WORDS, 3)]
    ratings = rng.sample([3.6, 3.9, 4.1, 4.3, 4.5, 4.7, 4.8], 3)
    return dict(results=[dict(name=n, rating=r, price_level=rng.randint(1, 4)) for n, r in zip(names, ratings)])


def reserve_make(rng):
    place = f"{rng.choice(WORDS)} {rng.choice(['Kitchen', 'House', 'Table'])}"
    date, time = f"2025-07-{rng.randint(1, 28):02d}", f"{rng.randint(17, 21)}:{rng.choice(['00', '15', '30', '45'])}"
    people = rng.randint(1, 8)
    request = (f"Book a table for {people} at {place} on {date} at {time}, and tell me the confirmation code."
               if rng.random() < 0.5 else
               f"Reserve {place} for {people} {'person' if people == 1 else 'people'} at {time} on {date}. "
               f"What's the confirmation code?")
    return request, dict(restaurant=[place], date=[date], time=[time], party_size=[people]), "confirmation"


def reserve_respond(args):
    rng = seeded("reserve", json.dumps(args, sort_keys=True))
    return dict(status="confirmed", confirmation=f"R{rng.randint(10000, 99999)}")


def units_make(rng):
    a, b = rng.choice(sorted(UNIT_FACTORS))
    value = rng.choice([rng.randint(1, 500), round(rng.uniform(0.5, 100), 1)])
    return rng.choice([f"How many {UNITS[b]} is {value} {UNITS[a]}?", f"Convert {value} {a} to {b}."]), \
        dict(value=[value], from_unit=[a], to_unit=[b]), "result"


def units_respond(args):
    factor = UNIT_FACTORS.get((args.get("from_unit"), args.get("to_unit")))
    if factor is None:
        return dict(error=f"unknown conversion; units: {', '.join(UNITS)}")
    return dict(result=round(float(args.get("value") or 0) * factor, 3), unit=args.get("to_unit"))


def order_make(rng):
    order = str(rng.randint(100000, 999999))
    key = rng.choice(["total", "status", "items"])
    request = {"total": f"What was the total of my order #{order}?", "status": f"What's the status of order {order}?",
               "items": f"How many items are in order {order}?"}[key]
    return request, dict(order_id=[order]), key


def order_respond(args):
    rng = seeded("order", str(args.get("order_id")).lstrip("#"))
    return dict(order_id=args.get("order_id"), status=rng.choice(["shipped", "processing", "delivered", "cancelled"]),
                items=rng.randint(1, 9), total=round(rng.uniform(5, 400), 2))


def showtimes_make(rng):
    movie, city = rng.choice(MOVIES), rng.choice(TOWNS)
    return f"When is the first showing of {movie} in {city} today?", dict(movie=[movie], city=[city]), "first"


def showtimes_respond(args):
    rng = seeded("showtimes", str(args.get("movie")).lower(), str(args.get("city")).lower())
    times = sorted(f"{h:02d}:{rng.choice(['00', '15', '30', '45'])}" for h in rng.sample(range(11, 23), 4))
    return dict(movie=args.get("movie"), city=args.get("city"), times=times)


APIS = {
    "fx_convert": Api(fn("fx_convert", "Convert an amount of money between currencies at today's rate.",
                         ["amount", "source", "target"], amount=("float", "the amount to convert"),
                         source=("string", "ISO 4217 code of the currency to convert from, e.g. USD"),
                         target=("string", "ISO 4217 code of the currency to convert to")), fx_make, fx_respond),
    "stock_quote": Api(fn("stock_quote", "The latest price of a stock.", ["symbol"],
                          symbol=("string", "the ticker symbol, e.g. AAPL")), quote_make, quote_respond),
    "flight_status": Api(fn("flight_status", "Status, departure time and gate of a flight today.", ["flight"],
                            flight=("string", "airline code and flight number, e.g. LH431")), flight_make,
                         flight_respond),
    "track_parcel": Api(fn("track_parcel", "Where a parcel is and when it will be delivered.", ["tracking_number"],
                           tracking_number=("string", "the parcel's tracking number")), parcel_make, parcel_respond),
    "local_time": Api(fn("local_time", "The current local time in a city.", ["city"],
                         city=("string", "the city's name")), time_make, time_respond),
    "dining_search": Api(fn("dining_search", "Restaurants of a cuisine in a city, with ratings.",
                                ["city", "cuisine"], city=("string", "the city"),
                                cuisine=("string", "the kind of food, e.g. Thai")), restaurants_make,
                             restaurants_respond),
    "reserve_table": Api(fn("reserve_table", "Book a table at a restaurant.",
                            ["restaurant", "date", "time", "party_size"],
                            restaurant=("string", "the restaurant's name"), date=("string", "YYYY-MM-DD"),
                            time=("string", "24-hour time, HH:MM"), party_size=("integer", "number of people")),
                         reserve_make, reserve_respond),
    "unit_convert": Api(fn("unit_convert", "Convert a quantity between units.", ["value", "from_unit", "to_unit"],
                           value=("float", "the quantity"), from_unit=("string", f"one of {', '.join(UNITS)}"),
                           to_unit=("string", f"one of {', '.join(UNITS)}")), units_make, units_respond),
    "order_lookup": Api(fn("order_lookup", "Status, number of items and total of an order in the shop.", ["order_id"],
                           order_id=("string", "the order number")), order_make, order_respond),
    "cinema_times": Api(fn("cinema_times", "Today's showtimes of a movie in a city.", ["movie", "city"],
                              movie=("string", "the movie's title"), city=("string", "the city")), showtimes_make,
                           showtimes_respond),
}
STATUS = {"status": "success"}  # an action's response (as agent_eval.BFCL_MOCK and tool_sources.MOCK)
NAMES = "Ana Ben Chloe David Emma Felix Grace Hugo Iris Jonas".split()
EVENTS = ["dentist", "team meeting", "lunch with Sam", "yoga class", "budget review", "piano lesson"]


def alarm_make(rng):
    time = f"{rng.randint(5, 9):02d}:{rng.choice(['00', '15', '30', '45'])}"
    request = rng.choice([f"Set an alarm for {time}.", f"Wake me up at {time} tomorrow.", f"Alarm at {time}, please."])
    return request, dict(time=[time]), time


def email_make(rng):
    name, subject = rng.choice(NAMES), rng.choice(["Invoice", "Meeting notes", "Trip plans", "Weekend", "Report"])
    to = f"{name.lower()}@example.com"
    request = rng.choice([f'Email {to} with the subject "{subject}".', f'Send an email to {to}, subject: {subject}.'])
    return request, dict(to=[to], subject=[subject]), to


def message_make(rng):
    name, text = rng.choice(NAMES), rng.choice(["I'm running late", "See you at 7", "Call me back", "Happy birthday!"])
    return rng.choice([f'Text {name}: "{text}"', f'Send {name} a message saying "{text}".']), \
        dict(contact=[name], text=[text]), name


def event_make(rng):
    title, date = rng.choice(EVENTS), f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    time = f"{rng.randint(8, 18):02d}:{rng.choice(['00', '30'])}"
    request = rng.choice([f'Put "{title}" in my calendar on {date} at {time}.', f"Add {title} on {date} at {time}."])
    return request, dict(title=[title], date=[date], time=[time]), title


def thermostat_make(rng):
    t = rng.randint(17, 24)
    return rng.choice([f"Set the heating to {t} degrees.", f"Make it {t} °C in here."]), dict(temperature=[t]), t


def status(args):
    return dict(STATUS)


ACTIONS = {
    "set_alarm": Api(fn("set_alarm", "Set an alarm on the phone.", ["time"], time=("string", "24-hour time, HH:MM")),
                     alarm_make, status, action=True),
    "send_email": Api(fn("send_email", "Send an email.", ["to", "subject"], to=("string", "the recipient's address"),
                         subject=("string", "the subject line")), email_make, status, action=True),
    "send_message": Api(fn("send_message", "Send a text message to a contact.", ["contact", "text"],
                           contact=("string", "the contact's name"), text=("string", "the message")), message_make,
                        status, action=True),
    "add_event": Api(fn("add_event", "Add an event to the calendar.", ["title", "date", "time"],
                        title=("string", "what the event is"), date=("string", "YYYY-MM-DD"),
                        time=("string", "24-hour time, HH:MM")), event_make, status, action=True),
    "set_thermostat": Api(fn("set_thermostat", "Set the home thermostat.", ["temperature"],
                             temperature=("integer", "the target temperature in °C")), thermostat_make, status,
                          action=True),
}
APIS |= ACTIONS
NO_CALL = [  # requests no declared function serves: answered directly (tier "none")
    lambda rng: (lambda a, b: (f"What is {a} plus {b}?", f"{a} + {b}"))(rng.randint(2, 99), rng.randint(2, 99)),
    lambda rng: (lambda a, b: (f"What is {a} times {b}?", f"{a} * {b}"))(rng.randint(2, 12), rng.randint(2, 12)),
    lambda rng: (lambda n: (f"How many days are in {n} weeks?", f"7 * {n}"))(rng.randint(2, 12)),
    lambda rng: (lambda h: (f"How many minutes are in {h} hours?", f"60 * {h}"))(rng.randint(2, 10)),
]  # (request, expression)
REFUSAL = "The query cannot be answered with the provided tools."  # (apigen's irrelevance reply)


def stated(result, key):
    """(the value of the response that the answer must state, values it must not state: the other candidates)."""
    if key == "best":
        names = [r["name"] for r in sorted(result["results"], key=lambda r: -r["rating"])]
        return names[0], names[1:]
    if key == "first":
        return result["times"][0], result["times"][1:]
    if key == "status":
        statuses = ["on time", "delayed", "cancelled", "shipped", "processing", "delivered"]
        return result[key], [x for x in statuses if x != result[key]]
    return result[key], []


@dataclasses.dataclass
class FuncallItem:
    request: str
    schemas: list  # the declared functions
    handlers: dict
    gold: list  # BFCL ground truth [{name: {param: [allowed values]}}], [] for no call
    value: object  # what the answer must state
    others: list  # what it must not
    reference: tuple = ("", "", "")  # (tool, code, answer) for rl.py --guide: none / unserved items
    builtins: bool = True  # the prompt lists the built-ins too (else only the declared functions, as BFCL)

    def check(self, ep):
        if not self.gold:  # only listed built-ins may be called (bash / python may compute the answer)
            text = plain(ep)
            allowed = BUILTINS if self.builtins else ()
            if any(c["tool"] not in allowed for c in ep.calls) or not text:  # (also an invented or malformed call)
                return 0.0
            return 1.0 if self.value is None else 0.5 + 0.5 * says_value(text, self.value)
        calls = [(c["tool"], scoring.parse_args(c["code"])) for c in ep.calls if c["tool"] in self.handlers]
        call_ok = scoring.call_matches(calls, self.gold, self.schemas) and len(calls) == len(ep.calls)
        text = plain(ep)
        said = says_value(text, self.value) and not any(says_value(text, o) for o in self.others)
        return 0.5 * call_ok + 0.5 * (call_ok and said)


def funcall_item(seed, i):
    """The i-th item of a seed; its task is the target function, "none" (arithmetic) or "unserved" (a request for an
    API that isn't declared)."""
    rng = random.Random(f"funcall/{seed}/{i}")
    declared = rng.sample(sorted(APIS), rng.randint(1, 3))
    handlers = {n: APIS[n].respond for n in declared}
    schemas = [APIS[n].schema for n in declared]
    rng.shuffle(schemas)
    r = rng.random()
    if r < NONE_FRAC:
        request, expr = rng.choice(NO_CALL)(rng)
        value = eval(expr)  # (our own expression)
        return "none", FuncallItem(request, schemas, handlers, [], value, [],
                                   ("python", f"print({expr})", f"The answer is {value}."))
    if r < NONE_FRAC + UNSERVED_FRAC:  # half with the functions only (as BFCL: no call at all), half with built-ins too
        request = APIS[rng.choice(sorted(set(APIS) - set(declared)))].make(rng)[0]
        builtins = rng.random() < 0.5
        return "unserved" if builtins else "unserved_only", \
            FuncallItem(request, schemas, handlers, [], None, [], ("", "", REFUSAL), builtins)
    api = APIS[declared[0]]
    request, allowed, key = api.make(rng)
    value, others = (key, []) if api.action else stated(api.respond({p: v[0] for p, v in allowed.items()}), key)
    return declared[0], FuncallItem(request, schemas, handlers, [{declared[0]: allowed}], value, others)


def funcall_pool(n=4000):
    from rl import Prompt
    pool = []
    for i in range(n):
        task, item = funcall_item(SEED0, i)
        tool, code, answer = item.reference
        system = system_prompt(item.schemas) if item.builtins else chat.tools_prompt(item.schemas)
        pool.append(Prompt(f"funcall/{i}", f"funcall_{task}", item.request, {}, item.check, system, item.handlers,
                           solution=code, solution_tool=tool or "bash", solution_answer=answer))
    return pool


# ---------------------------------------------------------------------------------------------------------------------
# chat: requests that need no tool

CHAT_SOURCES = [  # SmolTalk test splits: (subset, magpie categories, items per task)
    ("everyday-conversations", None, 200), ("smol-rewrite", None, 400), ("smol-summarize", None, 400),
    ("smol-magpie-ultra", ("creative-writing", "role-playing", "editing", "brainstorming", "planning",
                           "advice-seeking"), 400)]
CHAT_STOPWORDS = frozenset("""i me my we our you your he she it its they them their this that these those is are was
were be been am do does did have has had will would can could should may might and or but if so as of to in on at by
for with from about into than then there here what which who how when where why not no yes all any some just also very
more most much many one get got let lets im youre its thats dont""".split())


def content_features(text):
    """Content words (SQuAD normalization, CHAT_STOPWORDS dropped) and the bigrams with at least one content word."""
    w = scoring.normalize_answer(text).split()
    return [x for x in w if x not in CHAT_STOPWORDS] + \
        [f"{a} {b}" for a, b in zip(w, w[1:]) if a not in CHAT_STOPWORDS or b not in CHAT_STOPWORDS]


def content_f1(pred, reference):
    """F1 of content_features with a reference reply. A fixed reply can't pass across prompts: SmolTalk's most common
    80 content words get >= 0.1 on 1% of the pool, another prompt's reference (same category) on 12% (unigrams alone:
    67% / 40%)."""
    p, r = content_features(pred), content_features(reference)
    common = sum((collections.Counter(p) & collections.Counter(r)).values())
    return 0.0 if common == 0 else 2 * common / (len(p) + len(r))


def chat_rows():
    """(task, prompt, reference reply) of the CHAT_SOURCES test splits: the first user turn and the reply to it, minus
    agent_eval's chat prompts (smol-smoltalk's test split overlaps everyday-conversations')."""
    import pyarrow.parquet as pq
    from agent_eval import chat_prompts
    from build_evals import hf_file
    held_out = {" ".join(turns[-1][1].split()) for turns, _ in chat_prompts()}
    for subset, categories, n in CHAT_SOURCES:
        rows = pq.read_table(hf_file("HuggingFaceTB/smoltalk", f"data/{subset}/test-00000-of-00001.parquet")).to_pylist()
        by_task = {}
        for row in rows:
            msgs = [(m["role"], m["content"].strip()) for m in row["messages"] if m["role"] != "system"]
            if len(msgs) < 2 or msgs[0][0] != "user" or msgs[1][0] != "assistant" or len(msgs[0][1]) > 1500 or \
                    len(msgs[1][1]) > 3000 or " ".join(msgs[0][1].split()) in held_out:
                continue
            task = row.get("category") or subset.split("-")[-1]
            if categories is None or task in categories:
                by_task.setdefault(task, []).append((f"chat_{task}", msgs[0][1], msgs[1][1]))
        for items in by_task.values():
            yield from shuffled(items)[:n]


def chat_pool():
    """Requests that need no tool (greetings and small talk, creative writing, role play, editing, brainstorming,
    planning, advice, rewriting, summarizing), all tools listed. Reward: 0 with any tool call or without a finished
    reply, else content_f1 with SmolTalk's reference reply, scaled so that CHAT_F1_PASS is a pass (0.5)."""
    from rl import Prompt
    system = system_prompt()

    def check(reference):
        return lambda ep: 0.0 if ep.calls or ep.finish != "answer" else \
            min(1.0, 0.5 * content_f1(ep.answer, reference) / CHAT_F1_PASS)
    return [Prompt(f"chat/{i}", task, prompt, {}, check(reference), system)
            for i, (task, prompt, reference) in enumerate(chat_rows())]


# ---------------------------------------------------------------------------------------------------------------------
# weather

def weather_cities():
    import weather
    return [c for c in weather.CITIES if rl_split(c) and not scoring.eval_city(c)]


@functools.cache
def show(city):
    """What the weather tool shows for a city (weather.TODAY's)."""
    import weather
    return weather.summary(weather.simulate(city))


@dataclasses.dataclass
class WeatherItem:
    question: str
    kind: str
    places: list  # the cities the answer needs looked up
    ok: object  # answer text -> right?
    oracle: str  # a right answer

    def check(self, ep):
        """The answer is right and comes from weather calls for every city of the question (binary questions can be
        guessed: "warmer" half the time)."""
        looked_up = all(any(c["tool"] == "weather" and p.lower() in c["code"].lower() for c in ep.calls)
                        for p in self.places)
        return float(looked_up and self.ok(plain(ep))) * call_factor(ep)


def weather_item(cities, seed, i):
    """Templated questions on what the weather tool shows (weather.summary) for weather.TODAY: current temperature, a
    day's high, rain on a day, which of two cities is warmer now."""
    rng = random.Random(f"weather/{seed}/{i}")
    kind = rng.choice(["temperature", "high", "rain", "warmer"])
    a = rng.choice(cities)
    wa = show(a)
    k = rng.randint(0, 2)
    day = wa["daily"][k]
    when = ["today", "tomorrow", "on " + datetime.date.fromisoformat(day["date"]).strftime("%A")][k]
    if kind in ("temperature", "high"):
        value = wa["current"]["temperature"] if kind == "temperature" else day["high"]
        question = rng.choice([f"What's the temperature in {a} right now?", f"How warm is it in {a} at the moment?"]
                              if kind == "temperature" else
                              [f"What will the high be in {a} {when}?", f"How hot will it get in {a} {when}?"])
        ok, oracle = (lambda text: says_temperature(text, value)), f"It is {value}°C."
    elif kind == "rain":  # half of them rainy (otherwise always answering "no" pays)
        rain = rng.random() < 0.5
        a, k = rng.choice([(c, j) for c in cities for j in range(3)
                           if (show(c)["daily"][j]["precipitation"] > 0) == rain])
        day = show(a)["daily"][k]
        when = ["today", "tomorrow", "on " + datetime.date.fromisoformat(day["date"]).strftime("%A")][k]
        question = rng.choice([f"Will it rain in {a} {when}?", f"Do I need an umbrella in {a} {when}?"])
        ok, oracle = (lambda text: yes_no(text) == rain), "Yes, rain is expected." if rain else "No, it will be dry."
    else:
        temp = lambda city: show(city)["current"]["temperature"]
        b = rng.choice([c for c in cities if c != a and abs(temp(c) - temp(a)) >= 1])
        warmer, colder = (a, b) if temp(a) > temp(b) else (b, a)
        question = rng.choice([f"Is it warmer in {a} or in {b} right now?", f"Which is warmer right now, {a} or {b}?"])
        ok = lambda text: (first_mentioned(text, [warmer, colder]) == warmer
                           and not re.search(r"\b(colder|cooler)\b", text))
        oracle = f"It is warmer in {warmer}."
    return WeatherItem(question, kind, [a, b] if kind == "warmer" else [a], ok, oracle)


TEMPERATURE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?\b|C\b)")  # (as agent_eval.TEMPERATURE)


def says_temperature(text, value):
    """Exactly one temperature stated, within 1 degree of value: no hedging with two numbers or a range."""
    found = [float(x) for x in TEMPERATURE.findall(text)]
    return len(found) == 1 and abs(found[0] - value) <= 1


def yes_no(text):
    """True / False for a yes-or-no answer about rain (its first sentence), None if it says neither."""
    words = set(re.findall(r"[a-z']+", re.split(r"[.!?\n]", text.lower())[0]))
    if words & {"no", "not", "won't", "don't", "doesn't", "isn't", "dry", "unlikely", "sunny", "clear"}:
        return False
    return True if words & {"yes", "rain", "rainy", "raining", "umbrella", "showers", "drizzle", "likely"} else None


def first_mentioned(text, names):
    found = [(m.start(), n) for n in names for m in [re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.I)] if m]
    return min(found)[1] if found else None


def weather_pool(n=3000):
    """The distinct questions of the first n items (~1k: 21 cities x the templates)."""
    from rl import Prompt
    cities, system, pool = weather_cities(), system_prompt(), {}
    for i in range(n):
        item = weather_item(cities, SEED0, i)
        pool.setdefault(item.question, Prompt(f"weather/{i}", f"weather_{item.kind}", item.question, {}, item.check,
                                              system))
    return list(pool.values())


# ---------------------------------------------------------------------------------------------------------------------
# files: write a file with the given text (and say its size)

FILE_NAMES = ["todo", "shopping", "hello", "memo", "draft", "list", "reminder", "greeting", "plan", "ideas"]
FILE_LINES = ["see you soon", "buy milk", "call the dentist", "meeting at 3pm", "remember the keys", "water the plants",
              "the quick brown fox", "pack for the trip", "pay the rent", "good morning"]


def files_item(i):
    """(task, request, check): create a file with a line of text; files_bytes also asks for its size in bytes (with or
    without the trailing newline). The work dir starts empty. (The CLI checks use notes.txt / "hello world": not here.)"""
    rng = random.Random(f"files/{SEED0}/{i}")
    name = rng.choice(FILE_NAMES) + rng.choice([".txt", ".txt", ".md", ""])
    line = rng.choice(FILE_LINES)
    if rng.random() < 0.5:
        request = rng.choice([f"Create a file called {name} containing the line {line}.",
                              f"Write '{line}' to a new file named {name}.", f"Make a file {name} that says: {line}"])
        return "files_write", request, lambda ep: float(is_file(ep.sandbox, name, line))
    request = rng.choice([f"Create a file called {name} containing the line {line}, then tell me how many bytes it is.",
                          f"Write '{line}' to {name} and tell me the file's size in bytes."])
    sizes = {len(line.encode()), len(line.encode()) + 1}
    return "files_bytes", request, lambda ep: float(is_file(ep.sandbox, name, line) and ep.finish == "answer" and
                                                     any(x in sizes for x in numbers(ep.answer)))


def files_pool(n=2000):
    from rl import Prompt
    system = system_prompt()
    return [Prompt(f"files/{i}", task, request, {}, check, system)
            for i, (task, request, check) in enumerate(map(files_item, range(n)))]


# ---------------------------------------------------------------------------------------------------------------------
# nl2sh: a task in words -> one bash command (as agent_eval.nl2sh: no system prompt, the first command is scored by
# running it), on a small tree of our own (not the eval's fixture), from templates (none of the eval's tasks). No
# template overlaps terminal_new's held-out types: rl_web7 was trained with "How many words are in {f}?" (= count_words)
# and a "backup" target name (~ backup_csv), since removed.

SHELL_FILES = {
    "README.md": "# demo\nA small project.\nSee docs/ for details.\n",
    "docs/report.txt": "Q1 sales were up\nQ2 sales were flat\nQ3 sales were down\nQ4 sales were up\n",
    "docs/notes.md": "todo: fix the build\nidea: add tests\ntodo: write docs\n",
    "src/main.py": "import util\n\nprint(util.add(2, 3))\n",
    "src/util.py": "def add(a, b):\n    return a + b\n",
    "data/users.csv": "name,city,age\nana,lyon,31\nben,oslo,25\ncara,lyon,42\ndan,porto,25\n",
    "data/log.txt": "INFO start\nWARN low disk\nINFO request\nERROR timeout\nINFO request\nERROR timeout\n",
}
SHELL_TEMPLATES = [  # (task, command); {f} a text file, {d} a directory, {w} a word of {f}, {n} a count, {x} a name
    ("Count the lines in {f}.", "wc -l < {f}"),
    ("Show the first {n} lines of {f}.", "head -n {n} {f}"), ("Show the last {n} lines of {f}.", "tail -n {n} {f}"),
    ("Print the lines of {f} that contain '{w}'.", "grep '{w}' {f}"),
    ("Count the lines of {f} that contain '{w}'.", "grep -c '{w}' {f}"),
    ("Print the lines of {f} that don't contain '{w}'.", "grep -v '{w}' {f}"),
    ("Sort the lines of {f} alphabetically.", "sort {f}"), ("Show the unique lines of {f}, sorted.", "sort -u {f}"),
    ("Print {f} with line numbers.", "cat -n {f}"), ("Print the lines of {f} in reverse order.", "tac {f}"),
    ("Print {f} in uppercase.", "tr a-z A-Z < {f}"), ("How many bytes is {f}?", "wc -c < {f}"),
    ("List the files under {d}, sorted.", "find {d} -type f | sort"),
    ("How many files are under {d}?", "find {d} -type f | wc -l"),
    ("Which files under {d} contain '{w}'? List them sorted.", "grep -rl '{w}' {d} | sort"),
    ("Create a directory named {x}.", "mkdir {x}"), ("Copy {f} to {x}.", "cp {f} {x}"),
    ("Rename {f} to {x}.", "mv {f} {x}"), ("Delete {f}.", "rm {f}"),
    ("Create an empty file called {x}.", "touch {x}"),
    ("Print the second column of data/users.csv (comma-separated).", "cut -d, -f2 data/users.csv"),
    ("Count how many times each line occurs in {f}.", "sort {f} | uniq -c"),
]


def shell_run(command):
    """(output, final files) of a bash command in a fresh sandbox holding SHELL_FILES (as agent_eval.run_command)."""
    from tools import Sandbox
    with Sandbox(SHELL_FILES, timeout=5) as box:
        output, _ = box.run("bash", command)
        return "\n".join(line.rstrip() for line in output.strip("\n").split("\n")), box.state()


def nl2sh_rows():
    """(task, reference command) of every template instance, minus agent_eval's NL2SH-ALFA tasks."""
    import csv
    from build_evals import hf_file
    with open(hf_file("westenfelder/NL2SH-ALFA", "test.csv")) as f:
        held_out = {scoring.normalize_answer(r["nl"]) for r in csv.DictReader(f)}
    files = [f for f in SHELL_FILES if not f.endswith(".py")]
    rows = []
    for task, command in SHELL_TEMPLATES:
        for f in files if "{f}" in task + command else [None]:
            words = sorted(set(re.findall(r"[a-z]{3,}", SHELL_FILES[f]))) if f else ["todo", "sales", "timeout", "add"]
            for d in ["docs", "data", "src", "."] if "{d}" in task else [None]:
                for w in (words if "{w}" in task else [None]):
                    for n in ([1, 2, 3] if "{n}" in task else [None]):
                        for x in (["saved", "out.txt", "archive"] if "{x}" in task else [None]):
                            fill = dict(f=f, d=d, w=w, n=n, x=x)
                            rows.append((task.format(**fill), command.format(**fill)))
    return [(t, c) for t, c in dict.fromkeys(rows) if scoring.normalize_answer(t) not in held_out]


def nl2sh_pool():
    """Reward: 1 if the episode's first call is bash and its command leaves the same output and files as the
    reference's (the rest of the episode isn't scored)."""
    from rl import Prompt

    def check(reference):
        def score(ep):
            first = ep.calls[0] if ep.calls else None
            return float(bool(first) and first["tool"] == "bash" and shell_run(first["code"]) == shell_run(reference))
        return score
    return [Prompt(f"nl2sh/{i}", "nl2sh", task, dict(SHELL_FILES), check(command), solution=command)
            for i, (task, command) in enumerate(nl2sh_rows())]


# ---------------------------------------------------------------------------------------------------------------------
# the pass-rate gate: is an env learnable from a checkpoint (pass@1 >= ~5%, pass@16 >= ~30%)?

POOLS = {"search": search_pool, "funcall": funcall_pool, "weather": weather_pool, "chat": chat_pool,
         "files": files_pool, "nl2sh": nl2sh_pool}


def gate(args):
    import types

    import numpy as np

    import checkpoint
    import rl
    from agent import Agent
    from agent_eval import health, pass_at_k
    from engine import Engine, Sampling
    from tokenizer import Tokenizer
    params, cfg, _ = checkpoint.load_model(checkpoint.model_path(args.run_name), dtype=args.dtype)
    engine = Engine(params, cfg, n_slots=args.n_slots, max_len=args.max_len)
    agent = Agent(engine, Tokenizer.load(cfg.tokenizer), Sampling(), workers=16)  # T=1, full softmax, as RL samples
    ns = types.SimpleNamespace(group=args.n_samples, max_calls=8, max_tokens=512, max_len=args.max_len,
                               require_answer=1, guide="")
    if "search" in args.envs:
        load_search_index()
    report = {}
    for env in args.envs.split(","):
        by_task = {}
        for p in shuffled(POOLS[env]()):
            by_task.setdefault(p.task, []).append(p)
        prompts = [p for ps in by_task.values() for p in ps[:args.per_task]]
        episodes, rewards, _ = rl.rollout(agent, prompts, ns)
        for task in sorted(by_task) + [env]:
            idx = [i for i, p in enumerate(prompts) if p.task == task or task == env]
            r = [x for i in idx for x in rewards[i]]
            report[task] = dict(n=len(idx), reward=float(np.mean(r)), full=float(np.mean([x >= 1 for x in r])),
                                **pass_at_k([x >= rl.PASS for x in r], args.n_samples),
                                **{k: round(v, 3) for k, v in health([ep for i in idx for ep in episodes[i]]).items()})
            print(task, json.dumps(report[task]), flush=True)
        for p, eps, r in list(zip(prompts, episodes, rewards))[::max(1, len(prompts) // 6)]:
            print(f"--- {p.task} | {p.instruction[-200:]!r}\n{eps[0].reply[:600]!r} -> {r[0]:.2f}", flush=True)
    out = os.path.join(os.path.dirname(checkpoint.run_path(args.run_name)),
                       f"envs_gate_{args.envs.replace(',', '_')}.json")
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), results=report), f, indent=1)
    k = f"pass@{args.n_samples}"
    print(f"\n{'task':<28} {'n':>5} {'reward':>7} {'pass@1':>7} {k:>8} {'calls':>6} {'answered':>8}")
    for task, r in report.items():
        print(f"{task:<28} {r['n']:>5} {r['reward']:>7.3f} {r['pass@1']:>7.3f} {r[k]:>8.3f} {r['calls']:>6.2f} "
              f"{r['answered']:>8.2f}")


def shuffled(pool):
    return random.Random(0).sample(pool, len(pool))


def show_items():
    for env, build in POOLS.items():
        pool = build()
        print(f"=== {env}: {len(pool)} items, tasks {sorted({p.task for p in pool})}")
        for p in shuffled(pool)[:4]:
            print(f"--- {p.task}\n{p.instruction[-300:]}")
    print(funcall_pool(1)[0].system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="pass@k of a run on each env (default: show a few items)")
    parser.add_argument("--run_name", default="mt_tool20c")
    parser.add_argument("--envs", default="search,funcall,weather,chat")
    parser.add_argument("--per_task", type=int, default=32, help="items per task (source / API / question kind)")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--n_slots", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    gate(args) if args.gate else show_items()
