from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from ..core.config import Settings
from ..core.inference.client import RouterRequestsClient
from ..core.rag.retriever import Retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are MATRIX-AI, a concise, helpful assistant for the Matrix EcoSystem. "
    "Answer clearly and briefly. If unsure, say so."
)

# --- Singleton instance for the expensive Retriever class ---
_retriever_instance: Retriever | None = None

def get_retriever(settings: Settings) -> Retriever | None:
    """Initializes and returns a single instance of the Retriever."""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    kb_path = os.getenv("RAG_KB_PATH", "data/kb.jsonl")
    try:
        if Path(kb_path).exists():
            _retriever_instance = Retriever(kb_path=kb_path, top_k=settings.rag.top_k)
            logger.info("RAG enabled with KB at %s (top_k=%d)", kb_path, settings.rag.top_k)
        else:
            logger.info("RAG KB not found at %s — running LLM-only.", kb_path)
    except Exception as e:
        logger.warning("RAG disabled (failed to initialize Retriever: %s)", e)

    return _retriever_instance


class ChatService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = RouterRequestsClient(
            model=settings.model.name,
            fallback=settings.model.fallback,
            provider=getattr(settings.model, "provider", None),
            max_retries=2,
        )
        # Get the singleton retriever instance
        self.retriever = get_retriever(settings)

    def _build_context(self, query: str) -> Tuple[str, List[str]]:
        if not self.retriever:
            return "", []
        docs = self.retriever.retrieve(query, self.settings.rag.top_k)
        if not docs:
            return "", []
        blocks = [f"[{i+1}] {d['text']} (source: {d['source']})" for i, d in enumerate(docs)]
        context = "CONTEXT (use only these facts; if missing, say you don't know):\n" + "\n\n".join(blocks)
        sources = [d["source"] for d in docs]
        return context, sources

    def _augment(self, query: str) -> Tuple[str, List[str]]:
        """
        Build the final user message (with optional CONTEXT) and return sources.
        """
        ctx, sources = self._build_context(query)
        
        # --- THIS IS THE CORRECTED PROMPT ---
        if ctx:
            # New, clearer instruction format
            augmented = f"{ctx}\n\nBased only on the context provided above, answer the following question.\nQuestion: {query}"
        else:
            # If no context, just pass the original query
            augmented = query
            
        return augmented, sources

    # Note: These methods are now called from a thread pool in the router
    def answer_with_sources(self, query: str) -> Tuple[str, List[str]]:
        user_msg, sources = self._augment(query)
        text = self.client.chat_nonstream(
            SYSTEM_PROMPT, user_msg,
            max_tokens=self.settings.model.max_new_tokens,
            temperature=self.settings.model.temperature,
        )
        return text, sources

    def stream_answer(self, query: str):
        user_msg, _ = self._augment(query)
        return self.client.chat_stream(
            SYSTEM_PROMPT, user_msg,
            max_tokens=self.settings.model.max_new_tokens,
            temperature=self.settings.model.temperature,
        )