from fastapi import APIRouter, Depends, HTTPException
from ..deps import get_settings
from ..core.config import Settings
from ..core.schema import PlanRequest, PlanResponse
from ..services.plan_service import generate_plan

router = APIRouter()

@router.post("/plan", response_model=PlanResponse)
async def v1_plan(
    req: PlanRequest,
    settings: Settings = Depends(get_settings)
):
    """Generates a structured remediation plan based on application health context."""
    if req.mode != "plan":
        raise HTTPException(
            status_code=400,
            detail=f"Mode '{req.mode}' is not enabled. Only 'plan' is supported in Stage 1."
        )
    try:
        data = await generate_plan(req, settings=settings)
        return data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Inference service failed: {e}")
