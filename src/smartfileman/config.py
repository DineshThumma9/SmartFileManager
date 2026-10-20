from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Values are safe local defaults, not secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(
        default=Path(".smartfile"),
        validation_alias=AliasChoices("SMARTFILE_DATA_DIR", "data_dir"),
    )
    allowed_roots: str = Field(
        default=".",
        validation_alias=AliasChoices("SMARTFILE_ALLOWED_ROOTS", "allowed_roots"),
    )
    ollama_host: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("OLLAMA_HOST", "ollama_host"),
    )
    ollama_model: str = Field(
        default="embeddinggemma-2:740m",
        validation_alias=AliasChoices("OLLAMA_MODEL", "ollama_model"),
    )
    ollama_naming_model: str = Field(
        default="",
        validation_alias=AliasChoices("OLLAMA_NAMING_MODEL", "ollama_naming_model"),
    )
    ollama_naming_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias=AliasChoices("OLLAMA_NAMING_TIMEOUT_SECONDS", "ollama_naming_timeout_seconds"),
    )

    def allowed_root_paths(self, cwd: Path | None = None) -> tuple[Path, ...]:
        base = (cwd or Path.cwd()).resolve()
        roots = []
        for raw_root in self.allowed_roots.split(","):
            value = raw_root.strip()
            if value:
                roots.append((base / value).resolve() if not Path(value).is_absolute() else Path(value).resolve())
        return tuple(roots or (base,))

    def storage_dir(self, root: Path) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else root / self.data_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
