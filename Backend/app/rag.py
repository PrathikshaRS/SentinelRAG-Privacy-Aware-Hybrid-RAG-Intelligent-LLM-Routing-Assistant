import hashlib
import re
from collections import Counter
from typing import List, Optional

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
) -> list:
    """
    mode: "hybrid" (dense + BM25 fused with RRF), "dense", or "bm25".
    Returns dicts: id, score, child, parent, metadata. Parent texts are de-duplicated.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")

    pool = max(CANDIDATES, top_k * 5)

    collection = chroma_client.get_or_create_collection(collection_name(username, subject))
    data = collection.get(include=["documents", "metadatas"])
    ids, texts, metas = data["ids"], data["documents"], data["metadatas"]
    if not ids:
        return []

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
    lookup = {cid: (txt, meta) for cid, txt, meta in zip(ids, texts, metas)}

    ranked_lists = []

    # ---- Dense (semantic) ----
    if mode in ("hybrid", "dense"):
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
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:pool]
        ranked_lists.append([ids[i] for i in order if scores[i] > 0])

    # ---- Reciprocal Rank Fusion ----
    fused = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    # ---- Map to parent chunks, de-duplicated ----
    results, seen = [], set()
    for cid in sorted(fused, key=fused.get, reverse=True):
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