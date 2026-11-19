import hashlib
import re
from collections import Counter
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi

from app.chroma import chroma_client

# ---------------------------------------------------------------- chunking
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n", " "]
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30, separators=["\n\n", "\n", " "]
)

MIN_CHILD_WORDS = 8  # child chunks shorter than this are noise (page markers, stray headers)

# -------------------------------------------------------------- embeddings
class LocalEmbeddings(Embeddings):
    """Local ONNX MiniLM embeddings (384-dim). No API key, no cost."""

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self._fn(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


embeddings = LocalEmbeddings()

# ------------------------------------------------------------- collections
def collection_name(username: str, subject: str) -> str:
    """Deterministic, Chroma-safe name (3-63 chars, limited charset)."""
    return "u_" + hashlib.sha1(f"{username}:{subject}".encode()).hexdigest()[:24]


def get_vectorstore_for_user(username: str, subject: str):
    return Chroma(
        client=chroma_client,
        collection_name=collection_name(username, subject),
        embedding_function=embeddings,
    )


# ----------------------------------------------------------- text cleaning
def _prep(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)            # collapse doubled spaces
    text = re.sub(r" ?\n ?", "\n", text)           # trim spaces around newlines
    text = re.sub(r"(?im)^page[ ]*:?[ ]*\d+[ ]*$", "", text)  # "Page : 12" lines
    return text


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def clean_pages(pages: List[Document]) -> List[Document]:
    """Collapse whitespace, strip page markers, and drop headers/footers repeated across pages."""
    texts = [_prep(p.page_content) for p in pages]

    counts = Counter()
    for t in texts:
        for line in {_norm(l) for l in t.split("\n") if l.strip()}:
            counts[line] += 1
    threshold = max(3, int(0.3 * len(pages)))
    repeated = {line for line, c in counts.items() if c >= threshold and len(line) <= 60}

    cleaned = []
    for p, t in zip(pages, texts):
        lines = [l for l in t.split("\n") if _norm(l) not in repeated]
        t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if t:
            cleaned.append(Document(page_content=t, metadata=dict(p.metadata)))
    return cleaned


# --------------------------------------------------------------- ingestion
def chunk_document_for_ingestion(docs: List[Document], username: str, subject: str, filename: str):
    """Clean pages, split into parent-child chunks. Each child carries its parent's text in metadata."""
    pages = clean_pages(docs)
    parent_docs = parent_splitter.split_documents(pages)
    final_chunks = []
    for parent in parent_docs:
        children = child_splitter.split_documents([parent])
        for child in children:
            if len(child.page_content.split()) < MIN_CHILD_WORDS:
                continue
            child.metadata.update({
                "parent_content": parent.page_content,
                "username": username,
                "subject": subject,
                "filename": filename,
            })
            final_chunks.append(child)
    return final_chunks


# --------------------------------------------------------------- retrieval
RRF_K = 60
CANDIDATES = 10
MODES = {"hybrid", "dense", "bm25"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def _build_where(filters: dict) -> Optional[dict]:
    """Chroma needs a single condition, or {"$and": [...]} for several."""
    items = [{k: v} for k, v in filters.items()]
    if not items:
        return None
    return items[0] if len(items) == 1 else {"$and": items}


def _detect_filename(query: str, metadatas: list) -> Optional[str]:
    """If the query names an ingested file, restrict retrieval to it."""
    q = query.lower()
    names = {m.get("filename") for m in metadatas if m.get("filename")}
    for name in sorted(names, key=len, reverse=True):
        stem = name.rsplit(".", 1)[0].lower()
        if len(stem) >= 4 and (name.lower() in q or stem in q):
            return name
    return None


def retrieve_chunks(
    query: str,
    username: str,
    subject: str,
    top_k: int = 3,
    mode: str = "hybrid",
    filter_meta: Optional[dict] = None,
    fusion: str = "child",
    pool_size: Optional[int] = None,
    dense_backend: str = "exact",
) -> list:
    """
    mode: "hybrid" (dense + BM25 fused with RRF), "dense", or "bm25".
    fusion: "child" (default) fuses rank lists by child-chunk id, then de-duplicates parents.
            "parent" maps every child to its parent first, so votes for the same parent combine.
    pool_size: candidates taken from each method. Default grows with top_k; pass a fixed
            number to make results at different top_k comparable.
    dense_backend: "exact" ranks every chunk by squared L2 distance in numpy (deterministic; fine for
            thousands of chunks, since BM25 already reads the whole collection). "ann" uses Chroma's
            approximate search, which returned different neighbours across restarts on this data.
    Returns dicts: id, score, child, parent, metadata. Parent texts are de-duplicated.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if fusion not in ("child", "parent"):
        raise ValueError("fusion must be 'child' or 'parent'")
    if dense_backend not in ("exact", "ann"):
        raise ValueError("dense_backend must be 'exact' or 'ann'")

    pool = pool_size or max(CANDIDATES, top_k * 5)

    collection = chroma_client.get_or_create_collection(collection_name(username, subject))
    need_vecs = mode in ("hybrid", "dense") and dense_backend == "exact"
    data = collection.get(include=["documents", "metadatas"] + (["embeddings"] if need_vecs else []))
    ids, texts, metas = data["ids"], data["documents"], data["metadatas"]
    vecs = data.get("embeddings") if need_vecs else None
    if not ids:
        return []

    # Deterministic corpus order: results must not depend on Chroma's row order
    # (BM25 scores and fused RRF scores tie often, and ties were broken by that order).
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids = [ids[i] for i in order]
    texts = [texts[i] for i in order]
    metas = [metas[i] for i in order]
    if vecs is not None:
        vecs = np.asarray(vecs, dtype=np.float64)[order]

    # Only simple equality filters on strings. The collection is already per user+subject.
    filters = {
        k: v for k, v in (filter_meta or {}).items()
        if k != "subject" and isinstance(v, str) and v
    }
    if "filename" not in filters:
        detected = _detect_filename(query, metas)
        if detected:
            filters["filename"] = detected

    keep = [i for i, m in enumerate(metas) if all(m.get(k) == v for k, v in filters.items())]
    if not keep:
        return []
    ids = [ids[i] for i in keep]
    texts = [texts[i] for i in keep]
    metas = [metas[i] for i in keep]
    if vecs is not None:
        vecs = vecs[keep]
    lookup = {cid: (txt, meta) for cid, txt, meta in zip(ids, texts, metas)}

    ranked_lists = []

    # ---- Dense (semantic) ----
    if mode in ("hybrid", "dense"):
        if dense_backend == "exact":
            qv = np.asarray(embeddings.embed_query(query), dtype=np.float64)
            dist = ((vecs - qv) ** 2).sum(axis=1)          # squared L2, Chroma's default space
            dense_order = sorted(
                range(len(ids)), key=lambda i: (round(float(dist[i]), 12), ids[i])
            )[:pool]
            ranked_lists.append([ids[i] for i in dense_order])
        else:
            res = collection.query(
                query_embeddings=[embeddings.embed_query(query)],
                n_results=min(pool, len(ids)),
                where=_build_where(filters),
                include=["documents"],
            )
            ranked_lists.append(res["ids"][0])

    # ---- BM25 (keyword) over the same filtered chunks ----
    if mode in ("hybrid", "bm25"):
        bm25 = BM25Okapi([_tokenize(t) or ["_"] for t in texts])
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))[:pool]
        ranked_lists.append([ids[i] for i in order if scores[i] > 0])

    # ---- Parent-level Reciprocal Rank Fusion (optional) ----
    if fusion == "parent":
        def parent_of(cid):
            child_text, meta = lookup[cid]
            return meta.get("parent_content") or child_text

        fused_p, rep = {}, {}
        for ranked in ranked_lists:
            seen_keys, rank = set(), 0
            for cid in ranked:
                pk = parent_of(cid)
                if pk in seen_keys:
                    continue
                seen_keys.add(pk)
                fused_p[pk] = fused_p.get(pk, 0.0) + 1.0 / (RRF_K + rank + 1)
                rep.setdefault(pk, cid)
                rank += 1
        ordered = sorted(fused_p, key=lambda pk: (-round(fused_p[pk], 12), rep[pk]))[:top_k]
        return [
            {
                "id": rep[pk],
                "score": fused_p[pk],
                "child": lookup[rep[pk]][0],
                "parent": pk,
                "metadata": lookup[rep[pk]][1],
            }
            for pk in ordered
        ]

    # ---- Reciprocal Rank Fusion (child level) ----
    fused = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    # ---- Map to parent chunks, de-duplicated ----
    results, seen = [], set()
    for cid in sorted(fused, key=lambda c: (-round(fused[c], 12), c)):
        child_text, meta = lookup[cid]
        parent = meta.get("parent_content") or child_text
        if parent in seen:
            continue
        seen.add(parent)
        results.append({
            "id": cid,
            "score": fused[cid],
            "child": child_text,
            "parent": parent,
            "metadata": meta,
        })
        if len(results) == top_k:
            break
    return results


def hybrid_retrieve(
    query: str,
    username: str,
    subject: str,
    top_k: int = 3,
    filter_meta: Optional[dict] = None,
    mode: str = "hybrid",
) -> List[Document]:
    """LangChain-style wrapper: returns parent chunks as Documents."""
    out = []
    for r in retrieve_chunks(query, username, subject, top_k, mode, filter_meta):
        meta = {k: v for k, v in r["metadata"].items() if k != "parent_content"}
        meta.update({"chunk_id": r["id"], "score": r["score"]})
        out.append(Document(page_content=r["parent"], metadata=meta))
    return out


async def extract_filters_from_query(query: str) -> dict:
    """Placeholder. LLM-based filter extraction returns with the routing stage, once a chat model is in place."""
    return {}