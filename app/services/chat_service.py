from __future__ import annotations
from ..core.config import Settings
from ..core.inference.client import RouterRequestsClient

SYSTEM_PROMPT = (
    "You are MATRIX-AI, a concise, helpful assistant for the Matrix EcoSystem. "
    "Answer clearly and briefly. If unsure, say so."
)

class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = RouterRequestsClient(
            model=settings.model.name,
            fallback=settings.model.fallback,
            provider=settings.model.provider,
            max_retries=2,
            connect_timeout=10.0,
            read_timeout=60.0,
        )

    async def answer(self, query: str) -> str:
        # non-stream (compatible with current UI)
        return self.client.chat_nonstream(
            SYSTEM_PROMPT, query,
            max_tokens=self.settings.model.max_new_tokens,
            temperature=self.settings.model.temperature,
        )

    # Expose a generator for streaming endpoints
    def stream_answer(self, query: str):
        return self.client.chat_stream(
            SYSTEM_PROMPT, query,
            max_tokens=self.settings.model.max_new_tokens,
            temperature=self.settings.model.temperature,
        )
