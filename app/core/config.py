from __future__ import annotations
import os
import yaml
from pydantic import BaseModel, AnyHttpUrl
from typing import Optional

class ModelCfg(BaseModel):
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    fallback: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_new_tokens: int = 256
    temperature: float = 0.2

class LimitsCfg(BaseModel):
    rate_per_min: int = 60
    cache_size: int = 256

class RagCfg(BaseModel):
    index_dataset: Optional[str] = None
    top_k: int = 4

class MatrixHubCfg(BaseModel):
    base_url: AnyHttpUrl = "https://api.matrixhub.io"

class SecurityCfg(BaseModel):
    admin_token: Optional[str] = None

class Settings(BaseModel):
    model: ModelCfg = ModelCfg()
    limits: LimitsCfg = LimitsCfg()
    rag: RagCfg = RagCfg()
    matrixhub: MatrixHubCfg = MatrixHubCfg()
    security: SecurityCfg = SecurityCfg()

    @staticmethod
    def load() -> Settings:
        """Loads settings from YAML and overrides with environment variables."""
        path = os.getenv("SETTINGS_FILE", "configs/settings.yaml")
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        settings = Settings.model_validate(data)

        # Environment variable overrides
        if "MODEL_NAME" in os.environ:
            settings.model.name = os.environ["MODEL_NAME"]
        if "INDEX_DATASET" in os.environ:
            settings.rag.index_dataset = os.environ["INDEX_DATASET"]
        if "RATE_LIMITS" in os.environ:
            settings.limits.rate_per_min = int(os.environ["RATE_LIMITS"])
        if "ADMIN_TOKEN" in os.environ:
            settings.security.admin_token = os.environ["ADMIN_TOKEN"]

        return settings
