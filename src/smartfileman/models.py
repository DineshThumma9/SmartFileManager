from datetime import datetime

from sqlalchemy import Column, LargeBinary, Text
from sqlmodel import Field, SQLModel


class FileRecord(SQLModel, table=True):
    __tablename__ = "files"

    file_id: str = Field(primary_key=True)
    root_path: str = Field(index=True)
    relative_path: str = Field(index=True)
    filename: str
    extension: str
    mime_type: str
    content_hash: str = Field(index=True)
    size_bytes: int
    modified_time_ns: int
    status: str = "ready"
    updated_at: datetime


class ContentUnitRecord(SQLModel, table=True):
    __tablename__ = "content_units"

    unit_id: str = Field(primary_key=True)
    file_id: str = Field(foreign_key="files.file_id", index=True)
    unit_type: str
    ordinal: int
    content_hash: str = Field(index=True)
    text: str = Field(sa_column=Column(Text, nullable=False, default=""))
    locator_json: str = Field(default="{}", sa_column=Column(Text, nullable=False))
    extractor_version: str


class EmbeddingRecord(SQLModel, table=True):
    __tablename__ = "embeddings"

    embedding_id: str = Field(primary_key=True)
    unit_id: str = Field(foreign_key="content_units.unit_id", index=True)
    provider: str
    model_name: str
    model_digest: str = "unknown"
    dimension: int
    vector_blob: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    input_hash: str = Field(index=True)
    created_at: datetime
