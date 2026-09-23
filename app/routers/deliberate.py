from __future__ import annotations

from fastapi import APIRouter

from ..core.deliberation_schema import DeliberationRequest, DeliberationResponse
from ..services.deliberation_service import DeliberationService

router = APIRouter()


@router.post("/deliberate", response_model=DeliberationResponse)
async def deliberate(req: DeliberationRequest) -> DeliberationResponse:
    return await DeliberationService().deliberate(req)
