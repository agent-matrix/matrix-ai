import os
import importlib
import pytest

def test_settings_provider_order_env(monkeypatch):
    from app.core.config import Settings
    monkeypatch.setenv("PROVIDER_ORDER", "router,gemini,groq")
    s = Settings.load()
    assert s.provider_order == ["router", "gemini", "groq"]

def test_client_import_and_chat_function():
    mod = importlib.import_module("app.core.inference.client")
    assert hasattr(mod, "chat")
    assert callable(mod.chat)

@pytest.mark.parametrize("order", [
    "groq,gemini,router",
    "gemini,router",
    "router",
])
def test_provider_initialization(monkeypatch, order):
    # Provide fake keys so providers construct; we won't call the APIs here.
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("PROVIDER_ORDER", order)

    from app.core.config import Settings
    from app.core.inference.providers import MultiProviderChat

    s = Settings.load()
    chain = MultiProviderChat(s)
    assert len(chain.providers) >= 1
