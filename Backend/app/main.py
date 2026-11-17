import os
import json
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from langchain_openai import ChatOpenAI

from app.config import settings
from app.database import get_db, engine, Base
from app.models import User, Document
from app.schemas import UserCreate, Token, QueryRequest
from app.security import hash_password, verify_password, create_access_token, create_refresh_token
from app.deps import get_current_user
from app.utils import validate_file
from app.rag import hybrid_retrieve, extract_filters_from_query, retrieve_chunks
from app.tasks import process_document
from app.celery_app import celery_app

# --- Chat model (still OpenAI; switches to a local Ollama model in a later stage) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=settings.OPENAI_API_KEY)


# --- FastAPI App (creates tables on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="SentinelRAG", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== AUTH ENDPOINTS ====================

@app.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == user.username))
    if result.scalar_one_or_none():
        raise HTTPException(400, "Username already exists")

    hashed = hash_password(user.password)
    new_user = User(username=user.username, hashed_password=hashed)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    access = create_access_token({"sub": user.username})
    refresh = create_refresh_token({"sub": user.username})
    return {"access_token": access, "refresh_token": refresh}


@app.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == form.username))
    db_user = result.scalar_one_or_none()
    if not db_user or not verify_password(form.password, db_user.hashed_password):
        raise HTTPException(401, "Invalid credentials")

    access = create_access_token({"sub": form.username})
    refresh = create_refresh_token({"sub": form.username})
    return {"access_token": access, "refresh_token": refresh}

# ==================== DOCUMENT UPLOAD (Async + Celery) ====================

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    subject: str,
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Validates files, saves to disk, queues a Celery task, returns task_id."""
    user_dir = f"documents/{current_user.username}/{subject}"
    os.makedirs(user_dir, exist_ok=True)

    results = []
    for file in files:
        internal_uuid, internal_filename, safe_original = await validate_file(file)
        file_path = os.path.join(user_dir, internal_filename)

        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()

        # Skip files already uploaded for this subject (failed uploads may be retried)
        dup = await db.execute(
            select(Document).where(
                Document.user_id == current_user.id,
                Document.subject == subject,
                Document.file_hash == file_hash,
                Document.status != "failed",
            )
        )
        if dup.scalars().first():
            results.append({
                "original_name": safe_original,
                "status": "duplicate",
                "detail": "This file was already uploaded for this subject.",
            })
            continue

        with open(file_path, "wb") as f:
            f.write(content)

        # Insert into DB with status 'pending'
        stmt = insert(Document).values(
            user_id=current_user.id,
            original_name=safe_original,
            internal_uuid=internal_uuid,
            subject=subject,
            file_path=file_path,
            file_hash=file_hash,
            status="pending",
        )
        await db.execute(stmt)
        await db.commit()

        result = await db.execute(select(Document).where(Document.internal_uuid == internal_uuid))
        doc = result.scalar_one()

        # Push Celery task (worker sets completed/failed and processed_at)
        task = process_document.delay(
            file_path, current_user.username, subject, safe_original, internal_uuid
        )

        await db.execute(
            update(Document)
            .where(Document.id == doc.id)
            .values(status="processing")
        )
        await db.commit()

        results.append({
            "original_name": safe_original,
            "task_id": task.id,
            "status": "processing",
        })

    return JSONResponse(content={"uploads": results})


@app.get("/task-status/{task_id}")
async def task_status(task_id: str, current_user: User = Depends(get_current_user)):
    from celery.result import AsyncResult
    task = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None,
    }

# ==================== LIST DOCUMENTS ====================

@app.get("/documents")
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.user_id == current_user.id))
    docs = result.scalars().all()

    grouped = {}
    for d in docs:
        grouped.setdefault(d.subject, []).append(
            {"original_name": d.original_name, "status": d.status, "error": d.error}
        )

    return {"documents": grouped}

# ==================== CHAT / QUERY (Streaming + Hybrid) ====================

@app.post("/chat/stream")
async def chat_stream(
    req: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Streaming endpoint using hybrid retrieval."""
    query = req.query
    subject = req.subject

    # 1. Filters (stub for now; LLM-based extraction returns with the routing stage)
    filter_meta = await extract_filters_from_query(query)
    filter_meta["subject"] = subject

    # 2. Hybrid retrieval (BM25 + Dense + RRF)
    docs = hybrid_retrieve(query, current_user.username, subject, top_k=3, filter_meta=filter_meta)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Build prompt
    prompt = f"""You are an assistant answering questions based only on the provided context.
If the answer is not in the context, say "I don't have enough information about that."

Context:
{context}

Question: {query}
Answer:"""

    # 4. Streaming response
    async def stream_generator():
        async for chunk in llm.astream(prompt):
            yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# ==================== SIMPLE QUERY (Non-streaming, for testing) ====================

@app.post("/query")
async def query_documents(
    req: QueryRequest,
    current_user: User = Depends(get_current_user),
):
    query = req.query
    subject = req.subject

    filter_meta = await extract_filters_from_query(query)
    filter_meta["subject"] = subject

    docs = hybrid_retrieve(query, current_user.username, subject, top_k=3, filter_meta=filter_meta)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Context:
{context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    return {
        "query": query,
        "answer": response.content,
        "sources": [
            {"content": d.page_content[:200], "filename": d.metadata.get("filename")}
            for d in docs
        ],
    }

# ==================== DEBUG: SEE WHAT RETRIEVAL RETURNS ====================

@app.post("/debug/retrieve")
async def debug_retrieve(
    req: QueryRequest,
    mode: str = "hybrid",
    top_k: int = 3,
    current_user: User = Depends(get_current_user),
):
    if mode not in ("hybrid", "dense", "bm25"):
        raise HTTPException(400, "mode must be hybrid, dense or bm25")
    results = retrieve_chunks(req.query, current_user.username, req.subject, top_k=top_k, mode=mode)
    return {
        "query": req.query,
        "mode": mode,
        "results": [
            {
                "id": r["id"],
                "score": round(r["score"], 4),
                "filename": r["metadata"].get("filename"),
                "page": r["metadata"].get("page"),
                "child": r["child"],
                "parent_preview": r["parent"][:300],
            }
            for r in results
        ],
    }

# ==================== HEALTH CHECK ====================

@app.get("/")
async def root():
    return {"status": "healthy", "version": "2.0"}