from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContentUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_type: str
    ordinal: int
    content_hash: str
    text: str = ""
    locator: dict[str, str | int] = Field(default_factory=dict)
    extractor_version: str = "1"


class ScannedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    root_path: str
    relative_path: str
    filename: str
    extension: str
    mime_type: str
    content_hash: str
    size_bytes: int
    modified_time_ns: int
    units: list[ContentUnit] = Field(default_factory=list)
    status: Literal["ready", "unsupported", "error"] = "ready"
    error: str | None = None


class PlanOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    file_id: str
    source_relative: str
    destination_relative: str
    content_hash: str
    category: str
    subject: str
    confidence: float = Field(ge=0, le=1)
    reason: str
    status: Literal["ready", "needs-review", "conflict"] = "ready"


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    root_path: str
    mode: Literal["fuzzy", "semantic"]
    created_at: datetime = Field(default_factory=utc_now)
    operations: list[PlanOperation] = Field(default_factory=list)


class MoveRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move_id: str
    operation_id: str
    root_path: str
    source_relative: str
    destination_relative: str
    content_hash: str
    moved_at: datetime = Field(default_factory=utc_now)
    undone_at: datetime | None = None


class ScanResult(BaseModel):
    root_path: str
    files: list[ScannedFile] = Field(default_factory=list)

    @property
    def ready_files(self) -> list[ScannedFile]:
        return [file for file in self.files if file.status == "ready"]
