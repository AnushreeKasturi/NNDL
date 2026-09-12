from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Legal Risk Classifier API"
    app_env: str = "development"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database_url: str = "postgresql+psycopg2://postgres:postgres@postgres:5432/legalrisk"
    redis_url: str = "redis://redis:6379/0"
    queue_name: str = "legal-risk"

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_exp_minutes: int = 60 * 12

    default_model: str = "nlpaueb/legal-bert-base-uncased"
    infer_max_length: int = 1024
    infer_stride: int = 128
    infer_threshold: float = 0.5


settings = Settings()

