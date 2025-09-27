import os
import logging
import httpx
from typing import Optional, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class HFClient:
    def __init__(self, model: str, fallback: Optional[str] = None, timeout: int = 20):
        self.model = model
        self.fallback = fallback
        self.timeout = timeout

        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set. Put it in .env or export it before starting.")

        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        self.api_base = "https://api-inference.huggingface.co/models"

    async def _post(self, model: str, payload: dict) -> Any:
        url = f"{self.api_base}/{model}"
        # wait_for_model=true is helpful if the container is cold
        params = {"wait_for_model": "true"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=self.headers, json=payload, params=params)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _extract_text(data: Union[dict, list, str]) -> str:
        # HF can return list[{"generated_text": "..."}] or {"generated_text": "..."} or str
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return str(data[0]["generated_text"])
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"])
        if isinstance(data, str):
            return data
        # Some serverless returns {"error": "..."} with 200—handle gently
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"Hugging Face error: {data['error']}")
        raise RuntimeError(f"Unexpected HF response format: {data!r}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _generate_once(self, model: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max(1, int(max_new_tokens)),
                "temperature": float(max(temperature, 0.01)),
                "return_full_text": False,
            },
        }
        data = await self._post(model, payload)
        return self._extract_text(data)

    async def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        # Try primary
        try:
            return await self._generate_once(self.model, prompt, max_new_tokens, temperature)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            body = e.response.text
            logger.error("HTTP error from HF API for model %s: %s", self.model, body)
            # If not authorized / not found / gated, try fallback if defined
            if code in (401, 403, 404) and self.fallback and self.fallback != self.model:
                logger.warning("Falling back to model %s due to %s", self.fallback, code)
                try:
                    return await self._generate_once(self.fallback, prompt, max_new_tokens, temperature)
                except Exception:
                    # re-raise original meaningful error below
                    pass
            # Give a readable hint for common cause with Llama
            if code in (401, 403, 404) and "meta-llama" in self.model.lower():
                raise PermissionError(
                    "Hugging Face returned 404/403 for a gated model. "
                    "Make sure your HF account accepted the model license and your HF_TOKEN has access. "
                    f"Model={self.model}"
                ) from e
            raise
        except Exception as e:
            logger.error("Failed to call HF API for model %s: %s", self.model, e)
            # Try fallback for transient or parsing errors
            if self.fallback and self.fallback != self.model:
                try:
                    logger.warning("Falling back to model %s due to generic failure", self.fallback)
                    return await self._generate_once(self.fallback, prompt, max_new_tokens, temperature)
                except Exception:
                    pass
            raise
