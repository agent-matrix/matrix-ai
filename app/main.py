from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# -----------------------------------------------------------------------------
# Early: load .env (so HF_TOKEN, ADMIN_TOKEN, etc. are available locally)
# -----------------------------------------------------------------------------
def _load_env_file(paths: list[str]) -> None:
    """Load environment variables from the first existing path in `paths`.
    Prefer python-dotenv if present; otherwise use a tiny fallback parser."""
    logger = logging.getLogger("uvicorn.error")

    # 1) Try python-dotenv (best)
    try:
        from dotenv import load_dotenv  # type: ignore
        for p in paths:
            if os.path.exists(p):
                load_dotenv(dotenv_path=p, override=False)
                logger.info("Loaded environment from %s", p)
                return
        logger.info("No .env file found in %s (skipping)", paths)
        return
    except Exception:
        # 2) Fallback: simple parser
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[len("export ") :].strip()
                        if "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key, val = key.strip(), val.strip()
                        # strip optional quotes
                        if (val.startswith('"') and val.endswith('"')) or (
                            val.startswith("'") and val.endswith("'")
                        ):
                            val = val[1:-1]
                        # do not clobber existing env (Space Secrets)
                        os.environ.setdefault(key, val)
                logger.info("Loaded environment from %s (fallback parser)", p)
                return
            except Exception as e:
                logger.warning("Failed loading env from %s: %s", p, e)

    logger.info("No .env loaded (none found / parsers failed)")

# Try typical locations for local dev. HF Spaces will ignore this and use Secrets.
_load_env_file([".env", "configs/.env", ".env.local", "configs/.env.local"])


# -----------------------------------------------------------------------------
# Middlewares
# -----------------------------------------------------------------------------
# Prefer the canonical package name; if your repo uses "middlewares/", this tries both.
try:
    from .middleware import attach_middlewares  # singular
except Exception:
    try:
        from .middlewares import attach_middlewares  # plural
    except Exception:
        def attach_middlewares(app: FastAPI) -> None:  # no-op fallback
            logging.getLogger("uvicorn.error").warning(
                "attach_middlewares not found; continuing without custom middlewares."
            )

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------
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
    app.state.started_at = time.time()
    app.state.version = os.getenv("APP_VERSION", "1.0.0")

    # Minimal diagnostics; HF_TOKEN presence matters for inference
    hf_token_present = bool(os.getenv("HF_TOKEN"))
    logging.getLogger("uvicorn.error").info(
        "matrix-ai starting (version=%s, port=%s, hf_token_present=%s)",
        app.state.version,
        os.getenv("PORT", "7860"),
        "yes" if hf_token_present else "no",
    )
    try:
        yield
    finally:
        uptime = time.time() - getattr(app.state, "started_at", time.time())
        logging.getLogger("uvicorn.error").info(
            "matrix-ai shutting down (uptime=%.2fs)", uptime
        )


def create_app() -> FastAPI:
    app = FastAPI(
        title="matrix-ai",
        version=os.getenv("APP_VERSION", "1.0.0"),
        description="AI planning microservice for the Matrix EcoSystem",
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Middlewares (request-id, gzip, rate-limit, etc.)
    attach_middlewares(app)

    # Core routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(plan.router, prefix="/v1", tags=["Planning"])
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])

    # Optional UI (adds '/', '/chat', '/dev')
    if HAS_UI:
        app.include_router(ui_router, tags=["UI"])
    else:
        # Minimal root so HF root probes pass even without UI
        @app.get("/", include_in_schema=False)
        async def root() -> Dict[str, Any]:
            return {
                "ok": True,
                "service": "matrix-ai",
                "version": app.version,
                "docs": "/docs",
                "endpoints": {"plan": "/v1/plan", "chat": "/v1/chat", "healthz": "/healthz"},
            }

        @app.get("/home", include_in_schema=False)
        async def home_redirect():
            return RedirectResponse(url="/docs", status_code=302)

    return app


app = create_app()
