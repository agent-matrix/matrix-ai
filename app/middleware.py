import time
import logging
import json
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

# Try to import python-json-logger; fall back to a tiny JSON formatter if missing.
try:
    from pythonjsonlogger import jsonlogger  # type: ignore[import-not-found]
    _HAS_PY_JSON_LOGGER = True
except Exception:
    _HAS_PY_JSON_LOGGER = False

from .deps import get_settings
from .core.rate_limit import RateLimiter
from .core.logging import add_trace_id

# ---- Fallback JSON formatter (if python-json-logger isn't available) ----
class _SimpleJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "asctime": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "name": record.name,
            "levelname": record.levelname,
            "message": record.getMessage(),
            # We attach trace_id via logger.info(..., extra={"trace_id": "..."}).
            "trace_id": getattr(record, "trace_id", None),
        }
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            # Last-ditch plain log if JSON serialization ever fails
            return (
                f'{payload["asctime"]} {payload["name"]} {payload["levelname"]} '
                f'{payload["message"]} trace_id={payload["trace_id"]}'
            )

# Setup structured logging
logger = logging.getLogger("matrix-ai")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if _HAS_PY_JSON_LOGGER:
        # Same fields you had; python-json-logger builds JSON from this format string
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(trace_id)s"
        )
    else:
        formatter = _SimpleJsonFormatter()
        logging.getLogger("uvicorn.error").warning(
            "python-json-logger not found; using a minimal JSON formatter."
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_rate_limiter = RateLimiter()

def attach_middlewares(app: FastAPI) -> None:
    """Attaches all required middlewares to the FastAPI app."""
    # NOTE: We keep GZip, but your SSE endpoints already set `Content-Encoding: identity`
    # so they won't be buffered/compressed.
    app.add_middleware(GZipMiddleware, minimum_size=512)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def rate_limit_and_log_middleware(request: Request, call_next: Callable):
        # Attach per-request trace id
        add_trace_id(request)

        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"

        # Simple fixed-window limiter
        if not _rate_limiter.allow(
            client_ip, request.url.path, settings.limits.rate_per_min
        ):
            return Response(status_code=429, content="Rate limit exceeded")

        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

        logger.info(
            f'"{request.method} {request.url.path}" {response.status_code}',
            extra={"trace_id": getattr(request.state, "trace_id", "N/A")},
        )
        return response
