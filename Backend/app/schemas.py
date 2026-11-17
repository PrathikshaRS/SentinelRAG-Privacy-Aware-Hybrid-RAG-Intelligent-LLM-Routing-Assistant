from pydantic import BaseModel, Field
from typing import List, Optional

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class QueryRequest(BaseModel):
    query: str
    subject: str  # always required now
    conversation_id: Optional[int] = None

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str