from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class DeliberationRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=8000)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    max_candidates: int = Field(default=3, ge=1, le=5)
    max_steps: int = Field(default=8, ge=1, le=20)
    risk_ceiling: Literal["low", "medium", "high"] = "medium"


class PlanStepV2(BaseModel):
    step_id: str
    objective: str
    capability: str
    depends_on: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(min_length=1)
    verifiers: List[str] = Field(min_length=1)
    rollback: Optional[str] = None
    risk: Literal["low", "medium", "high", "critical"] = "low"


class PlanIRV2(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    plan_id: str
    goal: str
    strategy: str
    uncertainty: float = Field(ge=0, le=1)
    alternatives_considered: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    steps: List[PlanStepV2] = Field(min_length=1)


class DeliberationResponse(BaseModel):
    selected: PlanIRV2
    candidates: List[PlanIRV2]
    selection_reason: str
