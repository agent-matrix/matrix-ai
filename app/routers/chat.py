from fastapi import APIRouter, Depends, HTTPException
from ..deps import get_settings
from ..core.config import Settings
from ..core.schema import ChatRequest, ChatResponse
from ..services.chat_service import chat_answer

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def v1_chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings)
):
    """Answers questions about the MatrixHub ecosystem using RAG."""
    try:
        return await chat_answer(req, settings=settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {e}")
