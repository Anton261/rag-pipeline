from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=20)
