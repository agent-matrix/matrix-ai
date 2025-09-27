import time
import logging
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pythonjsonlogger import jsonlogger
from .deps import get_settings
from .core.rate_limit import RateLimiter
from .core.logging import add_trace_id

# Setup structured logging
logger = logging.getLogger("matrix-ai")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(trace_id)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_rate_limiter = RateLimiter()

def attach_middlewares(app: FastAPI):
    """Attaches all required middlewares to the FastAPI app."""
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
        add_trace_id(request)
        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"

        if not _rate_limiter.allow(client_ip, request.url.path, settings.limits.rate_per_min):
            return Response(status_code=429, content="Rate limit exceeded")

        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

        logger.info(
            f'"{request.method} {request.url.path}" {response.status_code}',
            extra={'trace_id': getattr(request.state, 'trace_id', 'N/A')}
        )
        return response
