from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Mode = Literal["plan", "summary", "patch-diff"]

class PlanConstraints(BaseModel):
    risk: Optional[str] = "low"
    max_steps: int = Field(default=3, ge=1, le=10)

class PlanContext(BaseModel):
    app_id: str
    symptoms: List[str] = Field(default_factory=list)
    lkg: Optional[str] = None

class PlanRequest(BaseModel):
    mode: Mode = "plan"
    context: PlanContext
    constraints: PlanConstraints = Field(default_factory=PlanConstraints)

class PlanResponse(BaseModel):
    plan_id: str
    steps: List[str]
    risk: str
    explanation: str

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=512)

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
