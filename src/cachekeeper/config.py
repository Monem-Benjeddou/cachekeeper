from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from cachekeeper.core.rules import RULES


def config_file() -> Path:
    return Path.home() / ".config" / "cachekeeper" / "config.toml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CACHEKEEPER_", env_ignore_empty=True, extra="forbid")

    upstream: str = "http://localhost:11434"
    backend: Literal["auto", "ollama", "llamacpp", "lmstudio"] = "auto"
    rules: tuple[str, ...] = ()
    keep_bodies: bool = False
    retention_days: int = Field(default=7, gt=0)
    upstream_timeout_seconds: float = Field(default=600, gt=0)
    host: str = "127.0.0.1"
    port: int = Field(default=4141, gt=0, lt=65536)
    db_path: Path = Path.home() / ".local" / "share" / "cachekeeper" / "cachekeeper.db"

    @classmethod
    def settings_customise_sources(  # noqa: PLR0913, PLR0917 - signature fixed by pydantic-settings
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, TomlConfigSettingsSource(settings_cls, toml_file=config_file())

    @field_validator("rules")
    @classmethod
    def known_rules(cls, rules: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(rules) - {rule.name for rule in RULES}
        if unknown:
            raise ValueError(f"unknown rules: {', '.join(sorted(unknown))}")
        return rules
