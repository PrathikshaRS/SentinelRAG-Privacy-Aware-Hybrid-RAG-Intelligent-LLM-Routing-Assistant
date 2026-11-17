"""Retrieval eval: hit-rate@3 / @5 and latency for bm25, dense and hybrid.

Run from Backend/ with the venv active and the Chroma server running:
    python eval_retrieval.py

A question is a HIT at k if any of the top-k retrieved chunks comes from one of the
question's answer pages. Pages in eval_questions.json are the PRINTED page numbers
("Page : N" footers); the index stores them 0-based, so stored page = printed page - 1.
"""
import json
import statistics
import time
from collections import defaultdict

from app.rag import retrieve_chunks

USERNAME, SUBJECT = "testuser", "test"
PAGE_OFFSET = 1
MODES = ["bm25", "dense", "hybrid"]

with open("eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)


def expected(q):
    return {p - PAGE_OFFSET for p in q["pages"]}


def run(mode, k):
    """Returns per-question results for one mode at one k."""
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        res = retrieve_chunks(q["question"], USERNAME, SUBJECT, top_k=k, mode=mode)
        ms = (time.perf_counter() - t0) * 1000
        pages = [r["metadata"].get("page") for r in res]
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "hit": any(p in expected(q) for p in pages),
            "pages": pages,
            "ms": ms,
        })
    return rows


def pct(hits, n):
    return f"{hits}/{n} ({100 * hits / n:.0f}%)" if n else "n/a"


# warm-up: the first call loads the embedding model, which would skew latency
retrieve_chunks("warm up", USERNAME, SUBJECT, top_k=3, mode="dense")

results = {mode: {k: run(mode, k) for k in (3, 5)} for mode in MODES}

n = len(questions)
print(f"\n{n} questions, 1 document, subject '{SUBJECT}'\n")
print(f"{'mode':<8} {'hit@3':<14} {'hit@5':<14} {'mean ms':>8} {'median ms':>10}")
for mode in MODES:
    r3, r5 = results[mode][3], results[mode][5]
    lat = [r["ms"] for r in r3]
    print(f"{mode:<8} {pct(sum(r['hit'] for r in r3), n):<14} "
          f"{pct(sum(r['hit'] for r in r5), n):<14} "
          f"{statistics.mean(lat):>8.0f} {statistics.median(lat):>10.0f}")

print("\nhit@3 by question type")
types = sorted({q["type"] for q in questions})
print(f"{'mode':<8} " + " ".join(f"{t:<16}" for t in types))
for mode in MODES:
    cells = []
    for t in types:
        rows = [r for r in results[mode][3] if r["type"] == t]
        cells.append(f"{pct(sum(r['hit'] for r in rows), len(rows)):<16}")
    print(f"{mode:<8} " + " ".join(cells))

print("\nmisses at k=3 (question id: stored pages returned)")
for mode in MODES:
    misses = [r for r in results[mode][3] if not r["hit"]]
    print(f"  {mode}: " + (", ".join(f"Q{r['id']}: {r['pages']}" for r in misses) or "none"))

with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nsaved eval_results.json")