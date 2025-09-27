from __future__ import annotations

import json
from typing import Any, AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from ..deps import get_settings
from ..core.config import Settings
from ..services.chat_service import ChatService

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None

    def as_text(self) -> str:
        if self.query: return self.query
        if self.question: return self.question
        if self.prompt: return self.prompt
        if self.messages:
            for m in reversed(self.messages):
                if m.role.lower() == "user": return m.content
            return self.messages[-1].content
        raise ValueError("Body must include 'query'/'question'/'prompt' or 'messages'")


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, settings: Settings = Depends(get_settings)):
    try:
        text = req.as_text()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    svc = ChatService(settings)
    try:
        # Run the blocking call in a thread pool to avoid freezing the server
        answer, sources = await run_in_threadpool(svc.answer_with_sources, text)
        return ChatResponse(answer=answer, sources=sources)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")


@router.get("/chat", response_model=ChatResponse)
async def chat_get(query: str = Query(...), settings: Settings = Depends(get_settings)):
    svc = ChatService(settings)
    try:
        # Run the blocking call in a thread pool
        answer, sources = await run_in_threadpool(svc.answer_with_sources, query)
        return ChatResponse(answer=answer, sources=sources)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")


# ---------- Streaming (SSE) ----------
def _sse_line(obj: Any) -> str:
    payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {payload}\n\n"


@router.get("/chat/stream")
async def chat_stream(query: str = Query(...), settings: Settings = Depends(get_settings)):
    svc = ChatService(settings)

    async def gen() -> AsyncIterator[str]:
        # Anti-buffer padding and initial ping
        yield ":" + (" " * 2048) + "\n\n"
        yield "event: ping\ndata: 0\n\n"
        
        try:
            # Run the blocking retrieval part in a thread pool, then stream the results
            stream_generator = await run_in_threadpool(svc.stream_answer, query)
            any_tokens = False
            for token in stream_generator:
                if token:
                    any_tokens = True
                    yield _sse_line({"delta": token})
            
            if not any_tokens:
                yield _sse_line({"delta": ""})
            yield _sse_line("[DONE]")
        except Exception as e:
            yield _sse_line({"error": str(e)})

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
        "Content-Encoding": "identity",
    }
    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8", headers=headers)


@router.post("/chat/stream")
async def chat_stream_post(req: ChatRequest, settings: Settings = Depends(get_settings)):
    try:
        q = req.as_text()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return await chat_stream(query=q, settings=settings)