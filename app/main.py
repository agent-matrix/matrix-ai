from fastapi import FastAPI
from .middleware import attach_middlewares
from .routers import health, plan, chat

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="matrix-ai",
        version="0.1.0",
        description="AI service for the Matrix EcoSystem"
    )
    attach_middlewares(app)
    app.include_router(health.router, tags=["Health"])
    app.include_router(plan.router, prefix="/v1", tags=["Planning"])
    app.include_router(chat.router, prefix="/v1", tags=["Chat"])
    return app

app = create_app()
