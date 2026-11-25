"""
Wikipedia search for the search / fetch tools: BM25 over ~100-word passages of English Wikipedia (wikimedia/wikipedia
20231101.en), each indexed with its article's title. Only the first max_passages passages of each article are indexed
(the lead and first sections, where most facts are); fetch serves whole articles.
    python search.py build [--max_passages=6]       # download + index into INDEX_DIR: ~7 min on 32 CPUs, 160G RAM
    python search.py query "who wrote on the origin of species"
    python search.py page "Charles Darwin#Education"
    python search.py upgrade --src=OLD --dir=NEW    # NEW: OLD (hard-linked) + the rerank files, ~2 min
    python search.py recall [--n=1000] [--rerank=0] [--eval]   # recall@3 on held-out dev (or eval) questions, latency
Files in INDEX_DIR, all memory-mapped (loading is instant, processes share the page cache):
    pages.bin, page_offsets.npy          article texts: the title, a blank line, then paragraphs (blank-line separated)
    passages.bin, passage_offsets.npy    the indexed passages; passage_page.npy: the article of each
    indptr.npy, postings.npy, weights.npy   inverted index over hashed terms: passage ids (int32) and BM25 weights
                                         (fp32), highest weight first within each term
    title_order.npy                      article ids sorted by casefolded title, for fetch
    page_first.npy, title_hash.npy, title_page.npy, titles.bin, title_offsets.npy   for reranking (write_rerank_files)
Terms: lowercased words without stopwords, plural "s" stripped, hashed into 2^25 buckets (no vocabulary to store).
Weights are precomputed per (term, passage), so a query is a sum of postings into a score array and a top-k. A query
reads at most max_postings postings per term (the best ones): only very common terms are cut, and they weigh little.
The full index: 6.13M articles, 20.8M passages, 826M postings; 33 GB on disk (pages 17G, passages 8.5G, postings and
weights 3.1G each). Recall@3 (answer string in the top 3, 1000 train questions each) and time per query on a Mila CPU:
    max_postings      NQ-open  TriviaQA  ms/query (median, p95)
    0 (exact BM25)    0.333    0.500     ~65, ~200
    500k              0.332    0.487     ~45, ~95
    200k (default)    0.313    0.466     ~22, ~45
    100k              0.284    0.443     ~11, ~24
Memory: memory-mapped, ~0.5 GB private + ~5 GB of shared page cache once warm; in_memory=True, ~7 GB private.
Reranking (an index with the rerank files, e.g. search_v2; see RERANK): the first stage's
top 100 plus the lead passages of articles whose title is in the query, by exact BM25 + lead / position / article-size
/ title-match bonuses. Recall@3 on held-out dev questions, NQ / TQA / PopQA / Hotpot: .29 / .62 / .35 / .36 -> .34 /
.64 / .58 / .44; ~20-70 ms per query (median), and it reads the candidates' passages (~12 GB page cache).
LiveWikipedia has the same interface over the Wikipedia API (the CLI's --live).
"""

import argparse
import bisect
import collections
import glob
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import threading
import time
import zlib
from array import array

import numpy as np

from config import BASE_DIR

INDEX_DIR = os.path.join(BASE_DIR, "search_v2")  # (with the rerank files; "search": plain BM25)
BUCKETS = 2**25
K1, B = 0.9, 0.4  # BM25 (Pyserini's defaults for passages)
MAX_POSTINGS = 200_000  # read per query term (see above)
RERANK = 100  # first-stage candidates reranked (with the lead passages of articles whose title is in the query)
# rerank score = exact BM25 + these, tuned on held-out train questions
LEAD, POSITION, LOG_BYTES = 3.0, -1.0, 1.0  # lead passage; x passage number (up to 5); x log(article bytes)
TITLE, TITLE_WORDS = 4.0, 2.0  # the article's title is a span of the query; x the span's words (up to 4)
TITLE_SPAN = 8  # longest query span looked up as a title
WORD = re.compile(r"[^\W_]+")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
STOP = frozenset("""a an and are as at be been but by can could did do does for from had has have he her him his how i if
in into is it its me my not of on or our she so than that the their them then there these they this those to us was we
were what when where which while who whom whose why will with would you your""".split())
JUNK = {"see also", "references", "external links", "further reading", "notes", "bibliography", "sources",
        "citations", "footnotes", "notes and references", "works cited", "references and notes", "general references"}


def terms(text, buckets=BUCKETS):
    """Hashed index terms of a text, with repeats."""
    out = []
    for w in WORD.findall(text.lower()):
        if w in STOP or (len(w) == 1 and not w.isdigit()):
            continue
        if len(w) > 3 and w[-1] == "s" and w[-2] not in "su":
            w = w[:-1]
        out.append(zlib.crc32(w.encode()) & (buckets - 1))
    return out


def title_key(title):
    """Lowercased words of a title without its parenthetical: "Mercury (planet)" -> "mercury"."""
    return " ".join(WORD.findall(re.sub(r"\s*\([^)]*\)$", "", title).lower()))


def key_hash(key):
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def is_heading(p):
    return len(p) < 100 and "\n" not in p and p[-1] not in ".!?:;,\"')]"


def paragraphs(text):
    """An article's paragraphs, up to its reference sections."""
    out = []
    for p in text.split("\n\n"):
        p = p.strip()
        if p.casefold() in JUNK:
            break
        if p:
            out.append(p)
    return out


def split_long(s, size):
    words = s.split()
    return [s] if len(words) <= 1.5 * size else [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def chunk(paras, size=100):
    """Paragraphs -> passages of whole sentences, up to ~size words; a heading starts a new passage unless the current
    one is short. Paragraphs within a passage are separated by newlines."""
    out, cur, n = [], [], 0
    for p in paras:
        if is_heading(p) and n >= size // 2:
            out.append(cur)
            cur, n = [], 0
        pieces = [q for s in (SENTENCE.split(p) if len(p.split()) > size else [p]) for q in split_long(s, size)]
        new = True
        for s in pieces:
            k = len(s.split())
            if n and n + k > size:
                out.append(cur)
                cur, n, new = [], 0, True
            if new:
                cur.append(s)
            else:
                cur[-1] += " " + s
            n, new = n + k, False
    if cur:
        out.append(cur)
    return ["\n".join(c) for c in out]


# --- building ---

def write_shard(articles, out_dir, max_passages, buckets=BUCKETS):
    """Indexes (title, text) articles into out_dir: their pages and passages, and the (passage, term, tf) postings."""
    os.makedirs(out_dir, exist_ok=True)
    page_off, pass_off, pass_page, lengths = array("q", [0]), array("q", [0]), array("i"), array("H")
    docs, tids, tfs = array("i"), array("I"), array("B")
    titles = []
    with open(f"{out_dir}/pages.bin", "wb") as pages, open(f"{out_dir}/passages.bin", "wb") as passages:
        for title, text in articles:
            if "may refer to" in text[:300]:  # disambiguation pages
                continue
            paras = paragraphs(text)
            if not paras:
                continue
            data = f"{title}\n\n" + "\n\n".join(paras)
            pages.write(data.encode())
            page_off.append(pages.tell())
            for passage in chunk(paras)[:max_passages]:
                tf = collections.Counter(terms(f"{title} {passage}", buckets))
                docs.extend([len(pass_page)] * len(tf))
                tids.extend(tf.keys())
                tfs.extend(min(f, 255) for f in tf.values())
                lengths.append(min(sum(tf.values()), 65535))
                pass_page.append(len(titles))
                passages.write(passage.encode())
                pass_off.append(passages.tell())
            titles.append(title)
    np.savez(f"{out_dir}/shard.npz", page_off=page_off, pass_off=pass_off, pass_page=pass_page, lengths=lengths,
             docs=docs, tids=tids, tfs=tfs)
    with open(f"{out_dir}/titles.json", "w") as f:
        json.dump(titles, f)


def save(path, a):
    """Writes a file next to its destination, then renames it: processes that have the old one mapped keep it."""
    with open(path + ".tmp", "wb") as f:
        np.save(f, a) if isinstance(a, np.ndarray) else a(f)
    os.replace(path + ".tmp", path)


def merge(shard_dirs, out_dir, buckets=BUCKETS, k1=K1, b=B):
    """Concatenates the shards into the index files of out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    shards = [dict(np.load(f"{d}/shard.npz")) for d in shard_dirs]
    titles = [t for d in shard_dirs for t in json.load(open(f"{d}/titles.json"))]
    for name, key, out_name in [("pages", "page_off", "page_offsets"), ("passages", "pass_off", "passage_offsets")]:
        offsets = [np.zeros(1, np.int64)]

        def concat(out):
            for d, s in zip(shard_dirs, shards):
                offsets.append(s[key][1:] + out.tell())
                with open(f"{d}/{name}.bin", "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 24)
        save(f"{out_dir}/{name}.bin", concat)
        save(f"{out_dir}/{out_name}.npy", np.concatenate(offsets))
    n_pages = np.cumsum([0] + [len(s["page_off"]) - 1 for s in shards])
    n_pass = np.cumsum([0] + [len(s["pass_page"]) for s in shards])
    save(f"{out_dir}/passage_page.npy", np.concatenate([s["pass_page"] + n for s, n in zip(shards, n_pages)]).astype(np.int32))
    lengths = np.concatenate([s["lengths"] for s in shards]).astype(np.float32)
    docs = np.concatenate([s["docs"] + n for s, n in zip(shards, n_pass)]).astype(np.int32)
    tids = np.concatenate([s.pop("tids") for s in shards])
    tfs = np.concatenate([s.pop("tfs") for s in shards])
    del shards
    df = np.bincount(tids, minlength=buckets)
    idf = np.log1p((len(lengths) - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = (k1 * (1 - b + b * lengths / lengths.mean())).astype(np.float32)
    weights = np.empty(len(docs), np.float32)  # (fp16 would halve it, but numpy adds them ~2x slower)
    for i in range(0, len(docs), 1 << 26):  # in slices, to bound the float32 temporaries
        s = slice(i, i + (1 << 26))
        tf = tfs[s].astype(np.float32)
        weights[s] = idf[tids[s]] * tf * (k1 + 1) / (tf + norm[docs[s]])
    del tfs
    # by term, then weight (descending: positive floats sort as their bits), then passage
    order = np.argsort((tids.astype(np.uint64) << 32) | (0xFFFFFFFF - weights.view(np.uint32)), kind="stable")
    del tids
    save(f"{out_dir}/postings.npy", docs[order])
    del docs
    save(f"{out_dir}/weights.npy", weights[order])
    del weights, order
    save(f"{out_dir}/indptr.npy", np.concatenate([[0], np.cumsum(df)]).astype(np.int64))
    save(f"{out_dir}/title_order.npy", np.array(sorted(range(len(titles)), key=lambda i: titles[i].casefold()), np.int32))
    write_rerank_files(out_dir, titles, np.load(f"{out_dir}/passage_page.npy"))
    meta = dict(buckets=buckets, k1=k1, b=b, pages=len(titles), passages=len(lengths), postings=int(df.sum()),
                avgdl=float(lengths.mean()))
    save(f"{out_dir}/meta.json", lambda f: f.write(json.dumps(meta).encode()))


def write_rerank_files(out_dir, titles, passage_page):
    """page_first.npy: each article's first passage (and the number of passages at the end); title_hash.npy /
    title_page.npy: the hashed title_key of every article, sorted, and the article of each; titles.bin /
    title_offsets.npy: the titles (~150 MB, so reranking doesn't read pages.bin)."""
    save(f"{out_dir}/page_first.npy", np.searchsorted(passage_page, np.arange(len(titles) + 1)).astype(np.int64))
    encoded = [t.encode() for t in titles]
    save(f"{out_dir}/title_offsets.npy", np.cumsum([0] + [len(t) for t in encoded]).astype(np.int64))
    save(f"{out_dir}/titles.bin", lambda f: f.write(b"".join(encoded)))
    hashes = np.array([key_hash(title_key(t)) for t in titles], np.uint64)
    order = np.argsort(hashes, kind="stable")
    save(f"{out_dir}/title_hash.npy", hashes[order])
    save(f"{out_dir}/title_page.npy", order.astype(np.int32))


def upgrade(src, out_dir, sample=200_000):
    """A reranking index in out_dir from one built before reranking (src, left as it is): src's files hard-linked,
    plus the rerank files and meta.json with the mean passage length (in terms, estimated on a sample)."""
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith((".npy", ".bin")) and not os.path.exists(f"{out_dir}/{name}"):
            os.link(f"{src}/{name}", f"{out_dir}/{name}")
    s = Searcher.load(src)
    titles = [s.title(i) for i in range(s.meta["pages"])]
    write_rerank_files(out_dir, titles, np.asarray(s.passage_page))
    ids = np.random.default_rng(0).choice(s.meta["passages"], min(sample, s.meta["passages"]), replace=False)
    avgdl = np.mean([len(terms(f"{titles[s.passage_page[i]]} {s.passage(i)}", s.meta["buckets"])) for i in ids])
    save(f"{out_dir}/meta.json", lambda f: f.write(json.dumps(dict(s.meta, avgdl=float(avgdl))).encode()))


def shard_file(job):
    import pyarrow.parquet as pq
    path, out_dir, max_passages = job
    if not os.path.exists(f"{out_dir}/titles.json"):
        rows = (r for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["title", "text"])
                for r in batch.to_pylist())
        write_shard(((r["title"], r["text"]) for r in rows), out_dir, max_passages)
    return out_dir


def build(out_dir, max_passages, workers, max_files=0):
    """max_files: only the first files of the dump (a small index, e.g. run.sh smoke)."""
    from huggingface_hub import list_repo_files, snapshot_download
    t0 = time.time()
    names = sorted(f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset")
                   if f.startswith("20231101.en/") and f.endswith(".parquet"))[:max_files or None]
    dump = snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns=names,
                             local_dir=os.path.join(out_dir, "dump"), max_workers=8)
    files = [f"{dump}/{name}" for name in names]
    print(f"downloaded {len(files)} files in {time.time() - t0:.0f}s")
    jobs = [(f, f"{out_dir}/shards/{i:03d}", max_passages) for i, f in enumerate(files)]
    with multiprocessing.Pool(workers) as pool:
        shard_dirs = []
        for d in pool.imap(shard_file, jobs):
            shard_dirs.append(d)
            print(f"{d} done, {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    merge(shard_dirs, out_dir)
    shutil.rmtree(f"{out_dir}/shards")
    print(f"merged in {time.time() - t1:.0f}s; total {time.time() - t0:.0f}s; {json.load(open(f'{out_dir}/meta.json'))}")


# --- searching ---

class Searcher:
    max_postings = MAX_POSTINGS

    @classmethod
    def load(cls, path=INDEX_DIR, in_memory=False):
        """in_memory: read the inverted index into RAM (~6.6 GB) instead of memory-mapping it, for fast queries from the
        start when the index is on a network file system."""
        self = cls()
        self.meta = json.load(open(f"{path}/meta.json"))
        arr = lambda name, mmap="r": np.load(f"{path}/{name}.npy", mmap_mode=mmap)
        self.indptr, self.postings, self.weights = (arr(n, None if in_memory else "r") for n in ("indptr", "postings", "weights"))
        self.page_offsets, self.passage_offsets, self.passage_page = arr("page_offsets"), arr("passage_offsets"), arr("passage_page")
        self.title_order = arr("title_order")
        self.reranks = os.path.exists(f"{path}/page_first.npy") and "avgdl" in self.meta  # (older indexes: BM25 only)
        if self.reranks:
            self.page_first, self.title_hash, self.title_page = arr("page_first"), arr("title_hash"), arr("title_page")
            self.title_offsets = arr("title_offsets")
            self.titles = np.memmap(f"{path}/titles.bin", np.uint8, mode="r")
        self.pages = np.memmap(f"{path}/pages.bin", np.uint8, mode="r")
        self.passages = np.memmap(f"{path}/passages.bin", np.uint8, mode="r")
        self.scores, self.lock = np.zeros(self.meta["passages"], np.float32), threading.Lock()
        return self

    def passage(self, i):
        return self.passages[self.passage_offsets[i]:self.passage_offsets[i + 1]].tobytes().decode()

    def page_text(self, i):
        return self.pages[self.page_offsets[i]:self.page_offsets[i + 1]].tobytes().decode()

    def title(self, i):
        if self.reranks:
            return self.titles[self.title_offsets[i]:self.title_offsets[i + 1]].tobytes().decode()
        start = self.page_offsets[i]
        return self.pages[start:min(start + 1024, self.page_offsets[i + 1])].tobytes().decode(errors="ignore").split("\n", 1)[0]

    def candidates(self, query, n):
        """Ids of the n passages with the best index BM25 scores (from at most max_postings postings per term)."""
        with self.lock:  # the score array is shared (and zeroed after each query)
            found = []
            for t in set(terms(query, self.meta["buckets"])):
                a, b = self.indptr[t], self.indptr[t + 1]
                b = min(b, a + self.max_postings) if self.max_postings else b
                if a < b:
                    docs = self.postings[a:b]
                    self.scores[docs] += self.weights[a:b]
                    found.append(docs)
            if not found:
                return []
            m = n * len(found)  # every passage is in cand at most once per term: the top m hold the n best
            found = np.concatenate(found)
            cand, scores = found, self.scores[found]
            if len(cand) > 64 * m:  # keep those at least as good as the m-th best of a sample (<= the m-th best of all)
                keep = scores >= np.partition(scores[::16], -m)[-m]
                cand, scores = cand[keep], scores[keep]
            best = cand[np.argpartition(-scores, m)[:m]] if len(cand) > m else cand
            best = sorted(set(best.tolist()), key=lambda i: (-self.scores[i], i))[:n]
            self.scores[found] = 0
        return best

    def title_matches(self, query):
        """{article: words of the span} of the articles whose title_key is a span of the query (not all stopwords)."""
        words = WORD.findall(query.lower())
        spans = {" ".join(words[i:j]): j - i for i in range(len(words)) for j in range(i + 1, min(len(words), i + TITLE_SPAN) + 1)
                 if not all(w in STOP for w in words[i:j])}
        out = {}
        for span, n in spans.items():
            h = np.uint64(key_hash(span))
            a, b = np.searchsorted(self.title_hash, h, "left"), np.searchsorted(self.title_hash, h, "right")
            for page in self.title_page[a:min(b, a + 50)].tolist():
                out[page] = max(out.get(page, 0), n)
        return out

    def top(self, query, k):
        """Ids of the k best passages. Reranking (an index with page_first.npy): the first stage's RERANK best and the
        first two passages of the articles named in the query, by exact BM25 plus the bonuses above (RERANK)."""
        if not self.reranks:
            return self.candidates(query, k)
        matches = self.title_matches(query)
        ids = self.candidates(query, RERANK)
        for page in matches:
            a, b = self.page_first[page], self.page_first[page + 1]
            ids += range(a, min(a + 2, b))
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        buckets, k1, b, avgdl = self.meta["buckets"], self.meta["k1"], self.meta["b"], self.meta["avgdl"]
        q = sorted(set(terms(query, buckets)))
        df = np.array([self.indptr[t + 1] - self.indptr[t] for t in q], np.float64)
        idf = np.log1p((self.meta["passages"] - df + 0.5) / (df + 0.5))
        pages = np.asarray(self.passage_page[ids], np.int64)
        tf, dl = np.zeros((len(ids), len(q))), np.zeros(len(ids))
        for r, (i, page) in enumerate(zip(ids, pages.tolist())):
            counts = collections.Counter(terms(f"{self.title(page)} {self.passage(i)}", buckets))
            tf[r] = [counts[t] for t in q]
            dl[r] = sum(counts.values())
        bm25 = (idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)[:, None])).sum(1)
        pos = np.asarray(ids) - self.page_first[pages]
        size = self.page_offsets[pages + 1] - self.page_offsets[pages]
        span = np.array([matches.get(p, 0) for p in pages.tolist()])
        score = (bm25 + LEAD * (pos == 0) + POSITION * np.minimum(pos, 5) + LOG_BYTES * np.log(size)
                 + (span > 0) * (TITLE + TITLE_WORDS * np.minimum(span, 4)))
        return [ids[j] for j in np.argsort(-score, kind="stable")[:k]]

    def search(self, query, k=3):
        """[(title, passage)] of the k best passages."""
        return [(self.title(self.passage_page[i]), self.passage(i)) for i in self.top(query, k)]

    def find(self, title):
        """The article id of a title (case-insensitive), or None."""
        q = title.strip().casefold()
        j = bisect.bisect_left(self.title_order, q, key=lambda i: self.title(i).casefold())
        return int(self.title_order[j]) if j < len(self.title_order) and self.title(self.title_order[j]).casefold() == q else None

    def page(self, title):
        """An article's text (its title line first) for "Title" or "Title#Section" (the text from that heading on).
        An unknown title gets the article of the best passage for it; "" if there is none."""
        title, _, section = title.partition("#")
        i = self.find(title)
        if i is None:
            top = self.top(title, 1)
            if not top:
                return ""
            i = int(self.passage_page[top[0]])
        return select_section(self.page_text(i), section)


def select_section(text, section):
    if not section.strip():
        return text
    head, _, body = text.partition("\n\n")
    paras = body.split("\n\n")
    q = section.strip().casefold()
    headings = [j for j, p in enumerate(paras) if is_heading(p)]
    for j in [j for j in headings if paras[j].casefold() == q] or [j for j in headings if q in paras[j].casefold()]:
        return f"{head}#{paras[j]}\n\n" + "\n\n".join(paras[j + 1:])
    return text  # no such section: the whole page


class LiveWikipedia:
    """The Searcher interface over the Wikipedia API (needs internet): search results' best passages, whole pages."""
    API = "https://en.wikipedia.org/w/api.php"

    def get(self, **params):
        from weather import get
        return get(self.API, format="json", formatversion=2, **params)

    def page(self, title):
        title, _, section = title.partition("#")
        pages = self.get(action="query", prop="extracts", explaintext=1, redirects=1, titles=title.strip())["query"]["pages"]
        if not pages or "missing" in pages[0]:
            hits = self.get(action="query", list="search", srsearch=title, srlimit=1)["query"]["search"]
            if not hits:
                return ""
            pages = self.get(action="query", prop="extracts", explaintext=1, titles=hits[0]["title"])["query"]["pages"]
        text = re.sub(r"\n*==+ *(.*?) *==+\n*", r"\n\n\1\n\n", pages[0].get("extract", ""))
        return select_section(f"{pages[0]['title']}\n\n" + "\n\n".join(paragraphs(text)), section)

    def search(self, query, k=3):
        hits = self.get(action="query", list="search", srsearch=query, srlimit=k)["query"]["search"]
        q, out = set(terms(query)), []
        for h in hits:
            passages = chunk(self.page(h["title"]).split("\n\n")[1:]) or [""]
            out.append((h["title"], max(passages, key=lambda p: len(q & set(terms(p))))))  # first of the best
        return out


# --- evaluation ---

def dev_questions(n, skip=1000):
    """{set: [(question, answers)]}: questions not in the web evals from NQ-open / TriviaQA rc.nocontext / HotpotQA
    distractor (hard, no yes/no) train and the PopQA test rows, shuffled (seed 0). The rerank weights were tuned on
    the first 1000 of each; recall skips them."""
    import random
    import pyarrow.parquet as pq
    import build_evals
    from scoring import normalize_answer
    held = set()
    for name in build_evals.QA_SETS:
        with open(os.path.join(build_evals.BASE_DIR, "evals", "web", f"{name}.jsonl")) as f:
            held |= {normalize_answer(json.loads(line)["question"]) for line in f}
    read = lambda repo, path, columns=None: pq.read_table(build_evals.hf_file(repo, path), columns=columns).to_pylist()
    sets = {"nq": [(r["question"], list(r["answer"])) for r in
                   read("google-research-datasets/nq_open", "nq_open/train-00000-of-00001.parquet")],
            "triviaqa": [(r["question"], [r["answer"]["value"]] + list(r["answer"]["aliases"])) for r in
                         read("mandarjoshi/trivia_qa", "rc.nocontext/train-00000-of-00001.parquet", ["question", "answer"])],
            "popqa": [(r["question"], r["answers"]) for r in build_evals.qa_rows("popqa")],
            "hotpotqa": [(r["question"], [r["answer"]]) for r in read("hotpotqa/hotpot_qa", "distractor/train-00000-of-00002.parquet",
                                                                      ["question", "answer", "level"])
                         if r["level"] == "hard" and r["answer"].lower() not in ("yes", "no")]}
    out = {}
    for name, rows in sets.items():
        rows = [r for r in {q.strip(): (q, a) for q, a in rows}.values() if normalize_answer(r[0]) not in held]
        random.Random(0).shuffle(rows)
        out[name] = rows[skip:skip + n]
    return out


def eval_questions(n):
    """{set: [(question, answers)]} of the web evals' QA sets (report once, never tune on them)."""
    import build_evals
    out = {}
    for name in build_evals.QA_SETS:
        with open(os.path.join(build_evals.BASE_DIR, "evals", "web", f"{name}.jsonl")) as f:
            out[name] = [(r["question"], r["answers"]) for r in map(json.loads, f)][:n]
    return out


def recall(searcher, sets, k=3):
    """Per set: the fraction of questions with a gold answer in the top-k passages (or their titles), the question
    itself as the query (as agent_eval's question_recall@3); and the time per query."""
    from scoring import has_answer
    for name, rows in sets.items():
        hits, ms = 0, []
        for q, answers in rows:
            t0 = time.time()
            results = searcher.search(q, k)
            ms.append(1000 * (time.time() - t0))
            hits += has_answer(" ".join(f"{t} {p}" for t, p in results), answers)
        print(f"{name}: recall@{k} {hits / len(rows):.3f} ({len(rows)} questions); ms/query: mean {np.mean(ms):.1f}, "
              f"median {np.median(ms):.1f}, p95 {np.percentile(ms, 95):.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "upgrade", "query", "page", "recall"])
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--dir", default=INDEX_DIR)
    parser.add_argument("--max_passages", type=int, default=6, help="passages indexed per article")
    parser.add_argument("--max_files", type=int, default=0, help="build: only the first files of the dump (0: all)")
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--in_memory", action="store_true", help="read the inverted index into RAM")
    parser.add_argument("--max_postings", default=str(MAX_POSTINGS), help="per query term; recall: comma-separated values to try")
    parser.add_argument("--src", default=INDEX_DIR, help="upgrade: the index to add the rerank files to (in --dir)")
    parser.add_argument("--eval", action="store_true", help="recall on the web evals' questions instead of dev ones")
    parser.add_argument("--rerank", type=int, default=1, help="0: first-stage BM25 only")
    args = parser.parse_args()
    if args.action == "build":
        build(args.dir, args.max_passages, args.workers, args.max_files)
    elif args.action == "upgrade":
        upgrade(args.src, args.dir)
    else:
        t0 = time.time()
        s = Searcher.load(args.dir, args.in_memory)
        caps = [int(float(c)) for c in args.max_postings.split(",")]
        s.max_postings = caps[0]
        s.reranks = s.reranks and bool(args.rerank)
        print(f"loaded in {time.time() - t0:.2f}s: {s.meta}, rerank {s.reranks}")
        if args.action == "query":
            t0 = time.time()
            results = s.search(args.text, args.k)
            print(f"{1000 * (time.time() - t0):.1f} ms")
            for i, (t, p) in enumerate(results, 1):
                print(f"[{i}] {t}\n{p}\n")
        elif args.action == "page":
            print(s.page(args.text)[:3000])
        else:
            sets = eval_questions(args.n) if args.eval else dev_questions(args.n)
            for s.max_postings in caps:
                print(f"max_postings={s.max_postings}")
                recall(s, sets, args.k)
            if os.path.exists("/proc/self/status"):
                print("".join(l for l in open("/proc/self/status") if l.startswith(("VmHWM", "RssAnon", "RssFile"))))
