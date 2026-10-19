import json
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from .db import create_database, open_session
from .embeddings.base import EmbeddingProvider
from .models import ContentUnitRecord, EmbeddingRecord, FileRecord
from .schemas import ScanResult


def _vector_blob(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unit_id(file_id: str, unit_type: str, ordinal: int, content_hash: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_id}:{unit_type}:{ordinal}:{content_hash}"))


def index_scan(scan_result: ScanResult, database_path: Path, provider: EmbeddingProvider) -> dict[str, int]:
    engine = create_database(database_path)
    embedded = 0
    reused = 0
    units_seen = 0

    with open_session(engine) as session:
        for file in scan_result.ready_files:
            file_record = FileRecord(
                file_id=file.file_id,
                root_path=file.root_path,
                relative_path=file.relative_path,
                filename=file.filename,
                extension=file.extension,
                mime_type=file.mime_type,
                content_hash=file.content_hash,
                size_bytes=file.size_bytes,
                modified_time_ns=file.modified_time_ns,
                status=file.status,
                updated_at=datetime.now(timezone.utc),
            )
            session.merge(file_record)
            pending: list[tuple[object, str, str]] = []
            for unit in file.units:
                units_seen += 1
                unit_id = _unit_id(file.file_id, unit.unit_type, unit.ordinal, unit.content_hash)
                session.merge(
                    ContentUnitRecord(
                        unit_id=unit_id,
                        file_id=file.file_id,
                        unit_type=unit.unit_type,
                        ordinal=unit.ordinal,
                        content_hash=unit.content_hash,
                        text=unit.text,
                        locator_json=json.dumps(unit.locator, sort_keys=True),
                        extractor_version=unit.extractor_version,
                    )
                )
                existing = session.exec(
                    select(EmbeddingRecord).where(
                        EmbeddingRecord.unit_id == unit_id,
                        EmbeddingRecord.provider == provider.provider_name,
                        EmbeddingRecord.model_name == provider.model_name,
                        EmbeddingRecord.input_hash == unit.content_hash,
                    )
                ).first()
                if existing:
                    reused += 1
                    continue
                pending.append((unit, unit_id, f"{file.filename}\n{unit.text}"))

            vectors = provider.embed([text for _, _, text in pending])
            for (unit, unit_id, _), vector in zip(pending, vectors, strict=True):
                session.add(
                    EmbeddingRecord(
                        embedding_id=str(uuid.uuid4()),
                        unit_id=unit_id,
                        provider=provider.provider_name,
                        model_name=provider.model_name,
                        dimension=len(vector),
                        vector_blob=_vector_blob(vector),
                        input_hash=unit.content_hash,
                        created_at=datetime.now(timezone.utc),
                    )
                )
                embedded += 1
        session.commit()

    return {"files": len(scan_result.ready_files), "units": units_seen, "embedded": embedded, "reused": reused}
