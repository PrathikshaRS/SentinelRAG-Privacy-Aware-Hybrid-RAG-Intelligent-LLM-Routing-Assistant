"""Diagnostic: is Chroma's dense search exact on this collection, and is the query embedding stable?

Run from Backend/ with the venv active and Chroma running:
    python check_dense.py
It changes nothing; it only reads the collection.
"""
import json

import numpy as np

from app.chroma import chroma_client
from app.rag import collection_name, embeddings

POOL = 25  # the largest candidate pool the eval uses (top_k=5 -> 25)

col = chroma_client.get_collection(collection_name("testuser", "test"))
print("collection metadata:", col.metadata)

data = col.get(include=["embeddings"])
ids = data["ids"]
E = np.array(data["embeddings"], dtype=np.float64)
norms = np.linalg.norm(E, axis=1)
print(f"chunks: {len(ids)}, dim: {E.shape[1]}, vector norm min/max: {norms.min():.6f}/{norms.max():.6f}")

with open("eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

max_embed_diff = 0.0
mismatches = []
for q in questions:
    v1 = np.array(embeddings.embed_query(q["question"]), dtype=np.float64)
    v2 = np.array(embeddings.embed_query(q["question"]), dtype=np.float64)
    max_embed_diff = max(max_embed_diff, float(np.abs(v1 - v2).max()))

    d = ((E - v1) ** 2).sum(axis=1)                       # squared L2 (Chroma's default space)
    dist = dict(zip(ids, d))
    exact = sorted(range(len(ids)), key=lambda i: (d[i], ids[i]))[:POOL]
    exact_ids = [ids[i] for i in exact]

    chroma_ids = col.query(query_embeddings=[v1.tolist()], n_results=POOL, include=[])["ids"][0]

    if chroma_ids != exact_ids:
        first = next(r for r, (a, b) in enumerate(zip(exact_ids, chroma_ids)) if a != b)
        mismatches.append(
            f"Q{q['id']}: same set={set(exact_ids) == set(chroma_ids)}, first difference at rank {first + 1}, "
            f"exact dist {dist[exact_ids[first]]:.6f} vs chroma's pick {dist[chroma_ids[first]]:.6f}"
        )

print(f"\nquery embedding: largest difference between two embeddings of the same text = {max_embed_diff:.2e}")
print(f"Chroma top-{POOL} differs from exact top-{POOL} for {len(mismatches)} of {len(questions)} questions")
for line in mismatches:
    print("  " + line)