# Placeholder for Stage-2 RAG chat service
from ..core.schema import ChatRequest, ChatResponse
from ..core.config import Settings

async def chat_answer(req: ChatRequest, settings: Settings) -> ChatResponse:
    """Placeholder chat function."""
    return ChatResponse(
        answer="The RAG chat service is not yet enabled in Stage-1.",
        sources=[]
    )
