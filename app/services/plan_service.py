# app/services/plan_service.py
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Iterable

from ..core.schema import PlanRequest, PlanResponse
from ..core.config import Settings
from ..core.redact import redact
from ..core.inference.client import ChatClient  # use the multi-provider cascade

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
# Compatibility adapter for tests & legacy call sites
# ----------------------------
Message = Dict[str, str]

class HFClient:
    """
    Backward-compatible adapter that mirrors the old interface:
        HFClient(model=...).generate(prompt: str) -> str (async)

    Under the hood it uses the new multi-provider cascade (ChatClient).
    The 'model' arg is accepted for compatibility but selection is driven
    by Settings/provider_order; we keep it so tests can assert the call.
    """
    def __init__(self, model: str, settings: Optional[Settings] = None):
        self._model = model  # kept for compatibility / tests
        self._client = ChatClient(settings)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages: Iterable[Message] = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages = list(messages) + [{"role": "user", "content": prompt}]

        # ChatClient.chat is sync; run it in a thread so this stays async-compatible
        def _call() -> str:
            return self._client.chat(
                messages,
                temperature=temperature,
                max_new_tokens=max_tokens,
                stream=False,
            )

        return await asyncio.to_thread(_call)


# ----------------------------
# Service (uses cascade via HFClient; non-stream for plan generation)
# ----------------------------
class PlanService:
    """
    Planner uses the multi-provider cascade (via HFClient adapter).
    Always non-stream for plan generation to simplify parsing.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # IMPORTANT: use keyword arg 'model=' so tests can assert called_with(model=...)
        self.llm = HFClient(model=settings.model.name, settings=settings)

    async def generate(self, req: PlanRequest) -> PlanResponse:
        """
        Build prompt -> call LLM (non-stream) -> robustly parse -> PlanResponse.
        Includes a one-shot JSON reformat retry if the first output isn't valid JSON.
        """
        final_prompt = _build_prompt(req)

        # 1) First pass: ask for the plan
        raw_text = await self.llm.generate(
            final_prompt,
            temperature=float(self.settings.model.temperature),
            max_tokens=int(self.settings.model.max_new_tokens),
            system_prompt=SYSTEM_PLANNER,
        )

        # 2) If not valid JSON, ask the model to strictly reformat to JSON only (no fences)
        needs_reformat = False
        try:
            _ = _extract_json_block(raw_text)
        except Exception:
            needs_reformat = True

        if needs_reformat:
            reformat = (
                "Format the following content as a strict JSON object with EXACT keys "
                "plan_id, steps (array of strings), risk (low|medium|high), explanation (string). "
                "Output ONLY JSON. No backticks. No extra keys.\n\nCONTENT:\n"
                + raw_text
            )
            raw_text = await self.llm.generate(
                reformat,
                temperature=max(0.05, float(self.settings.model.temperature) * 0.75),
                max_tokens=int(self.settings.model.max_new_tokens),
                system_prompt=SYSTEM_PLANNER,
            )

        # 3) Parse safely (or fallback) and validate against schema
        parsed = _safe_parse_or_fallback(raw_text, final_prompt)
        return PlanResponse.model_validate(parsed)


# ----------------------------
# Back-compat function (keeps existing imports working)
# ----------------------------
async def generate_plan(req: PlanRequest, settings: Settings) -> PlanResponse:
    """
    Backward-compatible entry point:
    previous code called services.plan_service.generate_plan(...)
    """
    service = PlanService(settings)
    return await service.generate(req)
