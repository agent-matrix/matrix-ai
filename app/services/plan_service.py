from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.schema import PlanRequest, PlanResponse
from ..core.config import Settings
from ..core.redact import redact
from ..core.inference.client import RouterRequestsClient

logger = logging.getLogger(__name__)

# ----------------------------
# Prompts
# ----------------------------
SYSTEM_PLANNER = (
    "You are MATRIX-AI Planner. Produce a short, safe JSON plan. "
    "Bounded steps, minimal risk, and explain briefly."
)

_PROMPT_TEMPLATE_CACHE: Optional[str] = None


def _get_prompt_template() -> str:
    """
    Load core/prompts/plan.txt once (cached).
    Fallback to a minimal instruction if missing.
    """
    global _PROMPT_TEMPLATE_CACHE
    if _PROMPT_TEMPLATE_CACHE is not None:
        return _PROMPT_TEMPLATE_CACHE

    try:
        path = Path(__file__).parent.parent / "core" / "prompts" / "plan.txt"
        _PROMPT_TEMPLATE_CACHE = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("FATAL: core/prompts/plan.txt not found. Using fallback template.")
        _PROMPT_TEMPLATE_CACHE = (
            "Generate a JSON plan with keys: plan_id, steps, risk, explanation. "
            "Keep steps short, safe, and auditable."
        )
    return _PROMPT_TEMPLATE_CACHE


def _render_context(req: PlanRequest) -> str:
    """
    Render a compact context string from the request.
    (Matches your earlier shape: app_id, symptoms, lkg, constraints.)
    """
    app_id = getattr(req.context, "app_id", None) or getattr(req.context, "entity_uid", "unknown")
    symptoms = getattr(req.context, "symptoms", []) or []
    lkg = getattr(req.context, "lkg", None) or getattr(req.context, "lkg_version", None) or "N/A"

    max_steps = getattr(req.constraints, "max_steps", 3)
    risk = getattr(req.constraints, "risk", "low")

    return (
        "Context:\n"
        f"- app_id: {app_id}\n"
        f"- symptoms: {', '.join(symptoms) if symptoms else 'none'}\n"
        f"- lkg_version: {lkg}\n"
        f"- constraints: max_steps={max_steps}, risk={risk}"
    )


def _build_prompt(req: PlanRequest) -> str:
    """
    Compose final prompt with system guidance + template + redacted context.
    """
    template = _get_prompt_template()
    context_str = _render_context(req)
    safe_context = redact(context_str)

    # You can tweak ordering if desired; this is clear and stable.
    return f"{SYSTEM_PLANNER}\n\n{template}\n\n{safe_context}\n\nJSON Response:"


# ----------------------------
# Output parsing
# ----------------------------
def _extract_json_block(text: str) -> Dict[str, Any]:
    """
    Try hard to recover a JSON object from LLM text.
    Supports ```json fences and "first { ... last }".
    Raises ValueError if no JSON object can be extracted.
    """
    s = text.strip()

    # Fenced block: ```json ... ```
    if "```" in s:
        fence_start = s.find("```")
        lang_tag = s.find("\n", fence_start + 3)
        if lang_tag != -1:
            fence_close = s.find("```", lang_tag + 1)
            if fence_close != -1:
                fenced = s[lang_tag + 1 : fence_close].strip()
                return json.loads(fenced)

    # Plain: first "{" to last "}"
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first : last + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON object found in output.")


def _safe_parse_or_fallback(raw_output: str, context_for_id: str) -> Dict[str, Any]:
    """
    Parse the model output into a dict, or return a safe fallback plan.
    """
    try:
        obj = _extract_json_block(raw_output)
        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON is not an object.")

        # Minimal normalization: ensure keys exist
        if "plan_id" not in obj or not obj["plan_id"]:
            obj["plan_id"] = hashlib.md5(context_for_id.encode()).hexdigest()[:12]
        if "steps" not in obj or not obj["steps"]:
            obj["steps"] = [
                "Pin to the last-known-good (LKG) version and re-run health probes."
            ]
        if "risk" not in obj or not obj["risk"]:
            obj["risk"] = "low"
        if "explanation" not in obj or not obj["explanation"]:
            obj["explanation"] = "Autofilled explanation."

        return obj

    except Exception as e:
        logger.warning("LLM output parsing failed: %s. Applying fallback plan.", e)
        return {
            "plan_id": hashlib.md5(context_for_id.encode()).hexdigest()[:12],
            "steps": [
                "Pin to the last-known-good (LKG) version and re-run health probes."
            ],
            "risk": "low",
            "explanation": (
                "Fallback plan: A safe default was applied due to a model output parsing error."
            ),
        }


# ----------------------------
# Service (requests-only, non-stream)
# ----------------------------
class PlanService:
    """
    Planner uses HF Router (requests-only). Always non-stream for plan generation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = RouterRequestsClient(
            model=settings.model.name,
            fallback=settings.model.fallback,
            provider=settings.model.provider,
            max_retries=2,
            connect_timeout=10.0,
            read_timeout=60.0,
        )

    async def generate(self, req: PlanRequest) -> PlanResponse:
        """
        Build prompt -> call Router (non-stream) -> robustly parse -> PlanResponse.
        """
        final_prompt = _build_prompt(req)
        # run the blocking requests call in a worker thread to avoid blocking the event loop
        raw_text = await asyncio.to_thread(
            self.client.plan_nonstream,
            SYSTEM_PLANNER,
            final_prompt,
            self.settings.model.max_new_tokens,
            self.settings.model.temperature,
        )
        parsed = _safe_parse_or_fallback(raw_text, final_prompt)
        return PlanResponse.model_validate(parsed)


# ----------------------------
# Back-compat function (keeps existing imports working)
# ----------------------------
async def generate_plan(req: PlanRequest, settings: Settings) -> PlanResponse:
    """
    Backward-compatible entry point:
    previous code called services.plan.generate_plan(...)
    """
    service = PlanService(settings)
    return await service.generate(req)
