from __future__ import annotations

import os
from typing import Any, Dict, List
import httpx


class MatrixLLMClient:
    """OpenAI-compatible MatrixLLM boundary.

    matrix-ai no longer needs provider-specific routing for v2 deliberation.
    MatrixLLM owns provider/model selection and failover.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("MATRIX_LLM_BASE_URL", "http://localhost:11435/v1").rstrip("/")
        self.api_key = os.getenv("MATRIX_LLM_API_KEY", "")
        self.model = os.getenv("MATRIX_LLM_MODEL", "deepseek-r1")
        self.timeout = float(os.getenv("MATRIX_LLM_TIMEOUT", "60"))

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])
