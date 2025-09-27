import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.core.schema import PlanRequest, PlanContext
from app.services.plan_service import generate_plan
from app.core.config import Settings

@pytest.mark.asyncio
async def test_generate_plan_successful_parse():
    """Tests successful plan generation and parsing."""
    mock_client = MagicMock()
    mock_client.generate = AsyncMock(return_value='{"plan_id": "123", "steps": ["step 1"], "risk": "low", "explanation": "test"}')

    with patch('app.services.plan_service.HFClient', return_value=mock_client) as mock_hf_client:
        req = PlanRequest(context=PlanContext(app_id="test-app", symptoms=["timeout"]))
        settings = Settings()
        response = await generate_plan(req, settings)

        assert response.plan_id == "123"
        assert response.steps == ["step 1"]
        mock_hf_client.assert_called_with(model=settings.model.name)

@pytest.mark.asyncio
async def test_generate_plan_parsing_fallback():
    """Tests the fallback mechanism when LLM output is invalid JSON."""
    mock_client = MagicMock()
    mock_client.generate = AsyncMock(return_value='This is not valid json')

    with patch('app.services.plan_service.HFClient', return_value=mock_client):
        req = PlanRequest(context=PlanContext(app_id="test-app", symptoms=["timeout"]))
        settings = Settings()
        response = await generate_plan(req, settings)

        assert response.explanation.startswith("Fallback plan:")
        assert len(response.steps) > 0
