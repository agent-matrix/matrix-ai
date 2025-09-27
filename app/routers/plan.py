from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..deps import get_settings
from ..core.config import Settings
from ..core.schema import PlanRequest, PlanResponse
from ..services.plan_service import generate_plan

router = APIRouter()


class PlanRequestIn(BaseModel):
    """
    Permissive boundary model so the Dev UI (and Guardian) can send richer payloads.
    We normalize to the strict PlanRequest after basic checks.
    """
    mode: Optional[str] = "plan"
    context: Dict[str, Any]
    constraints: Dict[str, Any]


@router.post("/plan", response_model=PlanResponse)
async def v1_plan(req_in: PlanRequestIn, settings: Settings = Depends(get_settings)):
    """
    Generate a structured remediation plan from health/context.
    - Accepts permissive input (extra keys allowed).
    - Coerces to strict PlanRequest (pydantic) before calling the service.
    """
    if (req_in.mode or "plan") != "plan":
        raise HTTPException(
            status_code=400,
            detail=f"Mode '{req_in.mode}' is not enabled. Only 'plan' is supported in Stage 1.",
        )

    try:
        # Coerce to strict schema; pydantic will validate & coerce types
        req = PlanRequest.model_validate(
            {
                "mode": "plan",
                "context": req_in.context,
                "constraints": req_in.constraints,
            }
        )
    except Exception as e:
        # Return a clear validation error rather than generic 500
        raise HTTPException(status_code=422, detail=f"Invalid plan payload: {e}")

    try:
        return await generate_plan(req, settings=settings)
    except Exception as e:
        # Surface inference/backend errors as 503 (service unavailable)
        raise HTTPException(status_code=503, detail=f"Inference service failed: {e}")
