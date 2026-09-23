from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from ..core.deliberation_schema import (
    DeliberationRequest,
    DeliberationResponse,
    PlanIRV2,
    PlanStepV2,
)
from .matrixllm_client import MatrixLLMClient


SYSTEM = """You are Matrix AI's deliberation engine.
You PROPOSE; you never authorize or execute.
Return JSON only with a top-level key 'candidates'. Each candidate has:
strategy, uncertainty (0..1), assumptions[], and steps[].
Each step must have objective, capability, success_criteria[], verifiers[],
risk and optional rollback. Prefer capabilities supplied by the caller.
Generate materially different strategies, not paraphrases.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("no JSON object in MatrixLLM response")
    return json.loads(text[first:last + 1])


def _fallback(req: DeliberationRequest) -> Dict[str, Any]:
    capability = req.capabilities[0] if req.capabilities else "observe.read"
    return {
        "candidates": [{
            "strategy": "cautious-single-path",
            "uncertainty": 0.7,
            "assumptions": ["MatrixLLM unavailable or returned invalid structured output"],
            "steps": [{
                "objective": req.goal,
                "capability": capability,
                "success_criteria": ["objective-specific verification passes"],
                "verifiers": ["independent_verifier"],
                "risk": "medium",
                "rollback": "restore previous durable state",
            }],
        }]
    }


class DeliberationService:
    def __init__(self, llm: MatrixLLMClient | None = None) -> None:
        self.llm = llm or MatrixLLMClient()

    async def deliberate(self, req: DeliberationRequest) -> DeliberationResponse:
        prompt = json.dumps({
            "goal": req.goal,
            "context": req.context,
            "capabilities": req.capabilities,
            "max_candidates": req.max_candidates,
            "max_steps": req.max_steps,
            "risk_ceiling": req.risk_ceiling,
        }, ensure_ascii=False)
        try:
            raw = await self.llm.chat([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ])
            payload = _extract_json(raw)
        except Exception:
            payload = _fallback(req)

        raw_candidates: List[Dict[str, Any]] = list(payload.get("candidates") or [])[:req.max_candidates]
        if not raw_candidates:
            raw_candidates = _fallback(req)["candidates"]

        plans: List[PlanIRV2] = []
        for idx, candidate in enumerate(raw_candidates, 1):
            steps: List[PlanStepV2] = []
            for sidx, step in enumerate(list(candidate.get("steps") or [])[:req.max_steps], 1):
                steps.append(PlanStepV2(
                    step_id=f"s{sidx}",
                    objective=str(step.get("objective") or req.goal),
                    capability=str(step.get("capability") or (req.capabilities[0] if req.capabilities else "observe.read")),
                    depends_on=[f"s{sidx-1}"] if sidx > 1 else [],
                    success_criteria=list(step.get("success_criteria") or ["independent verification passes"]),
                    verifiers=list(step.get("verifiers") or ["independent_verifier"]),
                    rollback=step.get("rollback"),
                    risk=str(step.get("risk") or "medium"),
                ))
            seed = f"{req.goal}:{idx}:{candidate.get('strategy','strategy')}"
            plans.append(PlanIRV2(
                plan_id=hashlib.sha256(seed.encode()).hexdigest()[:16],
                goal=req.goal,
                strategy=str(candidate.get("strategy") or f"candidate-{idx}"),
                uncertainty=float(candidate.get("uncertainty", 0.5)),
                alternatives_considered=[],
                assumptions=list(candidate.get("assumptions") or []),
                steps=steps or [PlanStepV2(
                    step_id="s1",
                    objective=req.goal,
                    capability=req.capabilities[0] if req.capabilities else "observe.read",
                    success_criteria=["independent verification passes"],
                    verifiers=["independent_verifier"],
                    risk="medium",
                )],
            ))

        strategies = [p.strategy for p in plans]
        for p in plans:
            p.alternatives_considered = [s for s in strategies if s != p.strategy]

        selected = min(plans, key=lambda p: p.uncertainty)
        return DeliberationResponse(
            selected=selected,
            candidates=plans,
            selection_reason="Selected lowest reported uncertainty; Matrix OS/Guardian still decide whether it may proceed.",
        )
