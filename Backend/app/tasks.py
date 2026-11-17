from datetime import datetime, timezone

from sqlalchemy import create_engine, update
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.celery_app import celery_app
from app.config import settings
from app.models import Document as DocumentRow
from app.rag import chunk_document_for_ingestion, get_vectorstore_for_user

sync_engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True)


def _set_status(internal_uuid: str, status: str, error: str = None):
    values = {"status": status, "error": error}
    if status in ("completed", "failed"):
        values["processed_at"] = datetime.now(timezone.utc)
    with sync_engine.begin() as conn:
        conn.execute(
            update(DocumentRow)
            .where(DocumentRow.internal_uuid == internal_uuid)
            .values(**values)
        )


@celery_app.task(name="process_document")
def process_document(file_path, username, subject, filename, internal_uuid):
    try:
        loader = (
            PyPDFLoader(file_path)
            if file_path.lower().endswith(".pdf")
            else TextLoader(file_path, encoding="utf-8")
        )
        pages = loader.load()
        chunks = chunk_document_for_ingestion(pages, username, subject, filename)
        if not chunks:
            raise ValueError("No extractable text (scanned PDF or empty file)")

        ids = [f"{internal_uuid}_{i}" for i in range(len(chunks))]
        get_vectorstore_for_user(username, subject).add_documents(chunks, ids=ids)

        _set_status(internal_uuid, "completed")
        return {"chunks": len(chunks)}
    except Exception as exc:
        _set_status(internal_uuid, "failed", error=str(exc)[:500])
        raise