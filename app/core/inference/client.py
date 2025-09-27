# app/core/inference/client.py
import os, json, time, logging
from typing import Dict, List, Optional, Iterator, Tuple

import requests

logger = logging.getLogger(__name__)

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

def _require_token() -> str:
    tok = os.getenv("HF_TOKEN")
    if not tok:
        raise ValueError("HF_TOKEN is not set. Put it in .env or export it before starting.")
    return tok

def _model_with_provider(model: str, provider: Optional[str]) -> str:
    if provider and ":" not in model:
        return f"{model}:{provider}"
    return model

def _mk_messages(system_prompt: Optional[str], user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    return msgs

def _timeout_tuple(connect: float = 10.0, read: float = 60.0) -> Tuple[float, float]:
    return (connect, read)

class RouterRequestsClient:
    """
    Simple requests-only client for HF Router Chat Completions.
    Supports non-streaming (returns str) and streaming (yields token strings).
    """
    def __init__(self, model: str, fallback: Optional[str] = None, provider: Optional[str] = None,
                 max_retries: int = 2, connect_timeout: float = 10.0, read_timeout: float = 60.0):
        self.model = model
        self.fallback = fallback if fallback != model else None
        self.provider = provider
        self.headers = {"Authorization": f"Bearer {_require_token()}"}
        self.max_retries = max(0, int(max_retries))
        self.timeout = _timeout_tuple(connect_timeout, read_timeout)

    # -------- Non-stream (single text) --------
    def chat_nonstream(
        self,
        system_prompt: Optional[str],
        user_text: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> str:
        payload = {
            "model": _model_with_provider(self.model, self.provider),
            "messages": _mk_messages(system_prompt, user_text),
            "temperature": float(max(0.0, temperature)),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        if frequency_penalty is not None:
            payload["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty is not None:
            payload["presence_penalty"] = float(presence_penalty)

        text, ok = self._try_once(payload)
        if ok:
            return text

        # fallback (if configured)
        if self.fallback:
            payload["model"] = _model_with_provider(self.fallback, self.provider)
            text, ok = self._try_once(payload)
            if ok:
                return text

        raise RuntimeError(f"Chat non-stream failed: model={self.model} fallback={self.fallback}")

    def _try_once(self, payload: dict) -> Tuple[str, bool]:
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(ROUTER_URL, headers=self.headers, json=payload, timeout=self.timeout)
                if r.status_code >= 400:
                    logger.error("Router error %s: %s", r.status_code, r.text)
                    last_err = RuntimeError(f"{r.status_code}: {r.text}")
                    time.sleep(min(1.5 * (attempt + 1), 3.0))
                    continue
                data = r.json()
                return data["choices"][0]["message"]["content"], True
            except Exception as e:
                logger.error("Router request failure: %s", e)
                last_err = e
                time.sleep(min(1.5 * (attempt + 1), 3.0))
        if last_err:
            logger.error("Router exhausted retries: %s", last_err)
        return "", False

    # -------- Streaming (yield token deltas) --------
    def chat_stream(
        self,
        system_prompt: Optional[str],
        user_text: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> Iterator[str]:
        payload = {
            "model": _model_with_provider(self.model, self.provider),
            "messages": _mk_messages(system_prompt, user_text),
            "temperature": float(max(0.0, temperature)),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        if frequency_penalty is not None:
            payload["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty is not None:
            payload["presence_penalty"] = float(presence_penalty)

        # primary
        ok = False
        for token in self._stream_once(payload):
            ok = True
            yield token
        if ok:
            return
        # fallback stream if primary produced nothing (or died immediately)
        if self.fallback:
            payload["model"] = _model_with_provider(self.fallback, self.provider)
            for token in self._stream_once(payload):
                yield token

    def _stream_once(self, payload: dict) -> Iterator[str]:
        try:
            with requests.post(ROUTER_URL, headers=self.headers, json=payload, stream=True, timeout=self.timeout) as r:
                if r.status_code >= 400:
                    logger.error("Router stream error %s: %s", r.status_code, r.text)
                    return
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception as e:
                        logger.warning("Stream JSON parse issue: %s | line=%r", e, line)
                        continue
        except Exception as e:
            logger.error("Stream request failure: %s", e)
            return

    # -------- Planning (non-stream) --------
    def plan_nonstream(self, system_prompt: str, user_text: str,
                       max_tokens: int, temperature: float) -> str:
        return self.chat_nonstream(system_prompt, user_text, max_tokens, temperature)
