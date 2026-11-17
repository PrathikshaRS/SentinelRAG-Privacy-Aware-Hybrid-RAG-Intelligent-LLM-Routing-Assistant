"""Wipes the test data for testuser/test: Chroma collection, document rows, saved files.
Run from Backend/ with the venv active. Safe to re-run."""
import shutil

from sqlalchemy import create_engine, text

from app.config import settings
from app.chroma import chroma_client
from app.rag import collection_name

USERNAME, SUBJECT = "testuser", "test"

name = collection_name(USERNAME, SUBJECT)
try:
    chroma_client.delete_collection(name)
    print("deleted Chroma collection", name)
except Exception as e:
    print("no Chroma collection to delete:", e)

engine = create_engine(settings.DATABASE_URL_SYNC)
with engine.begin() as conn:
    res = conn.execute(
        text(
            "DELETE FROM documents WHERE subject = :s "
            "AND user_id = (SELECT id FROM users WHERE username = :u)"
        ),
        {"s": SUBJECT, "u": USERNAME},
    )
    print("deleted document rows:", res.rowcount)

shutil.rmtree(f"documents/{USERNAME}/{SUBJECT}", ignore_errors=True)
print("removed saved files")