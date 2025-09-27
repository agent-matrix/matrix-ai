from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

# Your existing middleware bundle (req id, rate limit, etag, etc.)
from .middleware import attach_middlewares

# Core API routers
from .routers import health, plan, chat

# Optional UI (Home/Chat/Dev). If missing, we gracefully fall back to a JSON root.
try:
    from .ui import router as ui_router  # type: ignore
    HAS_UI = True
except Exception:  # pragma: no cover
    HAS_UI = False


TAGS_METADATA = [
    {"name": "Health", "description": "Liveness / readiness probes and basic service metadata."},
    {"name": "Planning", "description": "AI plan generation for Matrix Guardian (/v1/plan)."},
    {"name": "Chat", "description": "Lightweight RAG/Q&A about Matrix System (/v1/chat)."},
    {"name": "UI", "description": "Minimal web UI (Home, Chat, Dev) if enabled."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lightweight startup/shutdown hooks.
    Stores process start time for basic diagnostics and logs boot/shutdown.
    """
    app.state.started_at = time.time()
    app.state.version = os.getenv("APP_VERSION", "1.0.0")
    logging.getLogger("uvicorn.error").info(
        "matrix-ai starting (version=%s)", app.state.version
    )
    try:
        yield
    finally:
        uptime = time.time() - getattr(app.state, "started_at", time.time())
        logging.getLogger("uvicorn.error").info(
            "matrix-ai shutting down (uptime=%.2fs)", uptime
        )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="matrix-ai",
        version=os.getenv("APP_VERSION", "1.0.0"),
        description="AI planning microservice for the Matrix EcoSystem",
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Middlewares (request-id, gzip, rate-limit, idempotency headers, etc.)
    attach_middlewares(app)

    # Core routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(plan.router, prefix="/v1", tags=["Planning"])
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])

    # Optional UI (adds '/', '/chat', '/dev')
    if HAS_UI:
        app.include_router(ui_router, tags=["UI"])
    else:
        # Minimal root so HF Spaces / root health probes pass even without UI
        @app.get("/", include_in_schema=False)
        async def root() -> Dict[str, Any]:
            return {
                "ok": True,
                "service": "matrix-ai",
                "version": app.version,
                "docs": "/docs",
                "endpoints": {
                    "plan": "/v1/plan",
                    "chat": "/v1/chat",
                    "healthz": "/healthz",
                },
            }

        # Optional convenience redirect to API docs
        @app.get("/home", include_in_schema=False)
        async def home_redirect():
            return RedirectResponse(url="/docs", status_code=302)

    return app


app = create_app()
