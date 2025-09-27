import os
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class HFClient:
    def __init__(self, model: str, timeout: int = 20):
        self.model = model
        self.timeout = timeout
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set.")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.01), # Temp must be > 0
                "return_full_text": False,
            }
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()
                return result[0]['generated_text']
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from HF API for model {self.model}: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Failed to call HF API for model {self.model}: {e}")
                raise
