"""Retrieval eval: hit-rate@3 / @5 and latency for several retrieval variants.

Run from Backend/ with the venv active and the Chroma server running:
    python eval_retrieval.py                       # development questions
    python eval_retrieval.py --questions eval_holdout.json --out eval_holdout_results.json --final

A question is a HIT at k if any of the top-k retrieved chunks comes from one of the
question's answer pages. Pages in the questions file are PRINTED page numbers
("Page : N" footers); the index stores them 0-based, so stored page = printed page - 1.

The hold-out file should be run ONCE, after the configuration is frozen (--final is required).
Report every variant you tried, not only the best one.
"""
import argparse
import json
import statistics
import time

from app.rag import retrieve_chunks

USERNAME, SUBJECT = "testuser", "test"
PAGE_OFFSET = 1

VARIANTS = {
    "bm25": dict(mode="bm25"),
    "dense": dict(mode="dense"),
    "hybrid": dict(mode="hybrid"),                                           # baseline
    "hybrid_pool": dict(mode="hybrid", pool_size=30),                        # fixed candidate pool
    "hybrid_parent": dict(mode="hybrid", fusion="parent", pool_size=30),     # parent-level fusion
}

parser = argparse.ArgumentParser()
parser.add_argument("--questions", default="eval_questions.json")
parser.add_argument("--out", default="eval_results.json")
parser.add_argument("--final", action="store_true", help="confirm a one-time hold-out run")
args = parser.parse_args()

if "holdout" in args.questions.lower() and not args.final:
    raise SystemExit("This is the hold-out set: run it once, after the configuration is frozen. "
                     "Re-run with --final to confirm.")

with open(args.questions, encoding="utf-8") as f:
    questions = json.load(f)


def expected(q):
    return {p - PAGE_OFFSET for p in q["pages"]}


def run(name, k):
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        res = retrieve_chunks(q["question"], USERNAME, SUBJECT, top_k=k, **VARIANTS[name])
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

results = {name: {k: run(name, k) for k in (3, 5)} for name in VARIANTS}

n = len(questions)
print(f"\n{n} questions from {args.questions}, 1 document, subject '{SUBJECT}'\n")
print(f"{'variant':<15} {'hit@3':<14} {'hit@5':<14} {'mean ms':>8} {'median ms':>10}")
for name in VARIANTS:
    r3, r5 = results[name][3], results[name][5]
    lat = [r["ms"] for r in r3]
    print(f"{name:<15} {pct(sum(r['hit'] for r in r3), n):<14} "
          f"{pct(sum(r['hit'] for r in r5), n):<14} "
          f"{statistics.mean(lat):>8.0f} {statistics.median(lat):>10.0f}")

print("\nhit@3 by question type")
types = sorted({q["type"] for q in questions})
print(f"{'variant':<15} " + " ".join(f"{t:<16}" for t in types))
for name in VARIANTS:
    cells = []
    for t in types:
        rows = [r for r in results[name][3] if r["type"] == t]
        cells.append(f"{pct(sum(r['hit'] for r in rows), len(rows)):<16}")
    print(f"{name:<15} " + " ".join(cells))

# Does fusion add or lose hits relative to its two components? (k=3)
bm = {r["id"]: r["hit"] for r in results["bm25"][3]}
dn = {r["id"]: r["hit"] for r in results["dense"][3]}
print("\nfusion vs components at k=3")
print(f"  either component hits: {sum(bm[i] or dn[i] for i in bm)}/{n}")
for name in VARIANTS:
    if not name.startswith("hybrid"):
        continue
    h = {r["id"]: r["hit"] for r in results[name][3]}
    lost = [f"Q{i}" for i in h if (bm[i] or dn[i]) and not h[i]]
    gained = [f"Q{i}" for i in h if h[i] and not (bm[i] or dn[i])]
    print(f"  {name}: lost {len(lost)} {lost}, gained {len(gained)} {gained}")

print("\nmisses at k=3 (question id: stored pages returned; stored = printed - 1)")
for name in VARIANTS:
    misses = [r for r in results[name][3] if not r["hit"]]
    print(f"  {name}: " + (", ".join(f"Q{r['id']}: {r['pages']}" for r in misses) or "none"))

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {args.out}")