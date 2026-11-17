import os
import uuid
import re
import magic
from fastapi import HTTPException, UploadFile

ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}

def sanitise_filename(original: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', original)

async def validate_file(file: UploadFile) -> str:
    # Read first 1KB for MIME detection
    head = await file.read(1024)
    await file.seek(0)  # reset pointer
    
    mime = magic.from_buffer(head, mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"File type {mime} not allowed. Only text/plain and application/pdf.")
    
    # Generate internal UUID filename
    safe_original = sanitise_filename(file.filename)
    internal_uuid = str(uuid.uuid4())
    internal_filename = f"{internal_uuid}_{safe_original}"
    return internal_uuid, internal_filename, safe_original