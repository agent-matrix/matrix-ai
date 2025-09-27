import hashlib
import json
import logging
from pathlib import Path
from ..core.schema import PlanRequest, PlanResponse
from ..core.config import Settings
from ..core.inference.client import HFClient
from ..core.redact import redact

logger = logging.getLogger(__name__)
_PROMPT_TEMPLATE: str | None = None

def _get_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        try:
            path = Path(__file__).parent.parent / "core/prompts/plan.txt"
            _PROMPT_TEMPLATE = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("FATAL: core/prompts/plan.txt not found.")
            _PROMPT_TEMPLATE = "Generate a JSON plan with keys: plan_id, steps, risk, explanation."
    return _PROMPT_TEMPLATE

def _create_final_prompt(req: PlanRequest) -> str:
    template = _get_prompt_template()
    context_str = f"Context:\n- app_id: {req.context.app_id}\n- symptoms: {', '.join(req.context.symptoms)}\n- lkg_version: {req.context.lkg or 'N/A'}\n- constraints: max_steps={req.constraints.max_steps}, risk={req.constraints.risk}"
    safe_context = redact(context_str)
    return f"{template}\n\n{safe_context}\n\nJSON Response:"

def _parse_llm_output(raw_output: str, context_str: str) -> dict:
    try:
        start = raw_output.find('{')
        end = raw_output.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = raw_output[start:end+1]
            return json.loads(json_str)
        raise ValueError("No valid JSON object found in output.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"LLM output parsing failed: {e}. Applying safe fallback plan.")
        return {
            "plan_id": hashlib.md5(context_str.encode()).hexdigest()[:12],
            "steps": ["Pin to the last-known-good (LKG) version and re-run health probes."],
            "risk": "low",
            "explanation": "Fallback plan: A safe default was applied due to a model output parsing error."
        }

async def generate_plan(req: PlanRequest, settings: Settings) -> PlanResponse:
    final_prompt = _create_final_prompt(req)
    client = HFClient(model=settings.model.name)
    raw_response = await client.generate(
        prompt=final_prompt,
        max_new_tokens=settings.model.max_new_tokens,
        temperature=settings.model.temperature,
    )
    parsed_data = _parse_llm_output(raw_response, final_prompt)
    return PlanResponse.model_validate(parsed_data)
