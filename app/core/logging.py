import uuid
from fastapi import Request

def add_trace_id(request: Request) -> None:
    """Injects a unique trace_id into the request state."""
    if not hasattr(request.state, "trace_id"):
        request.state.trace_id = str(uuid.uuid4())
