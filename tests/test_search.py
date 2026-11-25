import json

import chat
import search
import tools
import weather

ARTICLES = [
    ("Charles Darwin", "Charles Robert Darwin was an English naturalist, geologist and biologist, widely known for his "
     "contributions to evolutionary biology. His book On the Origin of Species was published in 1859.\n\n"
     "Education \n\nDarwin studied medicine at the University of Edinburgh, then theology at Cambridge.\n\n"
     "References \n\nA reference."),
    ("Montreal", "Montreal is the largest city in the Canadian province of Quebec. " + "It has many parks. " * 60),
    ("Mercury (planet)", "Mercury is the first planet from the Sun and the smallest in the Solar System."),
    ("Mercury", "Mercury may refer to:\n\nMercury (planet)\n\nMercury (element)"),
]


def test_searcher(tmp_path):
    search.write_shard(ARTICLES[:2], tmp_path / "s0", max_passages=6, buckets=4096)
    search.write_shard(ARTICLES[2:], tmp_path / "s1", max_passages=6, buckets=4096)
    search.merge([tmp_path / "s0", tmp_path / "s1"], tmp_path, buckets=4096)
    s = search.Searcher.load(tmp_path)
    assert s.meta["pages"] == 3  # the disambiguation page is dropped
    assert s.reranks and s.title_matches("Is mercury hot? Ask Charles Darwin") == {2: 1, 0: 2}  # titles in the query
    assert [t for t, _ in s.search("who wrote the origin of species", 1)] == ["Charles Darwin"]
    assert s.search("smallest planet", 1)[0] == ("Mercury (planet)", ARTICLES[2][1])
    assert len(s.search("Montreal parks", 3)) == 3  # the long article is split into passages of <= ~100 words
    assert all(len(p.split()) <= 100 for _, p in s.search("Montreal parks", 3))
    assert s.search("zzzz") == []
    page = s.page("charles darwin")
    assert page.startswith("Charles Darwin\n\nCharles Robert Darwin") and "Cambridge" in page and "reference" not in page
    assert s.page("Charles Darwin#education") == "Charles Darwin#Education\n\nDarwin studied medicine at the " \
                                                 "University of Edinburgh, then theology at Cambridge."
    assert s.page("the planet Mercury").startswith("Mercury (planet)\n\n")  # unknown title: the best passage's page

    tools.SEARCHER = s
    with tools.Sandbox(isolate=False, functions={"add": lambda a: {"sum": a["x"] + a["y"]}}) as box:
        out, code = box.run("search", "Origin of Species")
        assert code is None and out.startswith("[1] Charles Darwin\nCharles Robert Darwin was")
        assert box.run("fetch", "Mercury (planet)")[0] == "Mercury (planet)\n\n" + ARTICLES[2][1]
        assert box.run("add", '{"x": 1, "y": 2}') == ('{"sum": 3}', None)
        assert json.loads(box.run("add", "x=1")[0])["error"]
        assert box.tool_names() == ["bash", "python", "search", "fetch", "weather", "add"]
    tools.SEARCHER = None


def test_top_k(tmp_path):
    """The top-k shortcut (a sampled score threshold, for many candidates) finds the k best passages."""
    articles = [(f"Park {i}", f"{'green ' * (i % 7)}parks {'trees ' * (i % 5)} in the city {i % 3}") for i in range(2000)]
    search.write_shard(articles, tmp_path / "s", max_passages=1, buckets=4096)
    search.merge([tmp_path / "s"], tmp_path, buckets=4096)
    s = search.Searcher.load(tmp_path)
    query = "green parks trees city 2"
    for t in set(search.terms(query, 4096)):
        a, b = s.indptr[t], s.indptr[t + 1]
        s.scores[s.postings[a:b]] += s.weights[a:b]
    scores = s.scores.copy()
    s.scores[:] = 0
    assert [scores[i] for i in s.candidates(query, 3)] == sorted(scores, reverse=True)[:3]  # (ties: any of them)


def test_weather_and_prompt():
    raw = weather.simulate("Montreal", "2025-01-10")
    assert raw == weather.simulate("montreal ", "2025-01-10")  # deterministic
    out = weather.summary(raw)
    assert out["location"] == "Montreal, Canada" and [d["date"] for d in out["daily"]] == ["2025-01-10", "2025-01-11", "2025-01-12"]
    assert out["daily"][0]["high"] < 15  # winter
    assert weather.parse("Paris, France, 2025-06-15") == ("Paris, France", "2025-06-15")
    with tools.Sandbox(isolate=False) as box:
        assert json.loads(box.run("weather", "Montreal, 2025-01-10")[0]) == out

    schema = {"name": "add", "description": "Adds two numbers", "parameters": {"type": "object", "properties": {}}}
    prompt = chat.tools_prompt(["bash", "search", schema])
    assert prompt.splitlines() == ["You can use these tools:", "- bash: run a bash command",
                                   "- search: search Wikipedia; input: a query; output: the 3 best passages",
                                   "Functions (input: a JSON object of arguments):", json.dumps(schema)]
