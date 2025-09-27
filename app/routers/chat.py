# app/routers/chat.py
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from ..deps import get_settings
from ..core.config import Settings
from ..services.chat_service import ChatService

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    # accept several shapes so UI/clients don't 422:
    query: Optional[str] = None
    question: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None

    def as_text(self) -> str:
        if self.query:
            return self.query
        if self.question:
            return self.question
        if self.prompt:
            return self.prompt
        if self.messages:
            # prefer last user message
            for m in reversed(self.messages):
                if m.role.lower() == "user":
                    return m.content
            # fallback to last message
            if self.messages:
                return self.messages[-1].content
        raise ValueError("Body must include 'query'/'question'/'prompt' or 'messages'")

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, settings: Settings = Depends(get_settings)):
    try:
        text = req.as_text()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    svc = ChatService(settings)
    answer = await svc.answer(text)
    return ChatResponse(answer=answer)

# Handy for curl/browser tests:
#   GET /v1/chat?query=hello
@router.get("/chat", response_model=ChatResponse)
async def chat_get(query: str = Query(...), settings: Settings = Depends(get_settings)):
    svc = ChatService(settings)
    answer = await svc.answer(query)
    return ChatResponse(answer=answer)
