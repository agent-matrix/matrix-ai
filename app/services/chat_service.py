# app/services/chat_service.py
from __future__ import annotations
from ..core.config import Settings
from ..core.inference.client import HFClient

SYSTEM_PROMPT = (
    "You are MATRIX-AI, a concise, helpful assistant for the Matrix EcoSystem. "
    "Answer clearly and briefly. If unsure, say so."
)

class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = HFClient(model=settings.model.name)

    async def answer(self, query: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:"
        text = await self.client.generate(
            prompt=prompt,
            max_new_tokens=self.settings.model.max_new_tokens,
            temperature=self.settings.model.temperature,
        )
        return (text or "").strip()
