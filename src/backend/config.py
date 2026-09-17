from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "swatterfly-phase1"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:3000"
    replay_seed: int = 20260917
    replay_frames: int = 48
    replay_interval_seconds: float = 0.25
    nebius_api_key: str | None = Field(default=None, validation_alias="NEBIUS_API_KEY")
    nebius_base_url: str = Field(
        default="https://api.tokenfactory.nebius.com/v1/",
        validation_alias="NEBIUS_BASE_URL",
    )
    nebius_model: str = "glm-5.3"
    nebius_fallback_model: str = "glm-5.2"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
