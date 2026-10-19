import hashlib
import json
import math
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from .db import create_database, open_session
from .embeddings.base import EmbeddingProvider
from .models import ContentUnitRecord, EmbeddingRecord, FileRecord
from .schemas import ContentUnit, ScanResult, ScannedFile

EMBEDDING_CACHE_VERSION = "cache-v3-content-first"
EMBEDDING_BATCH_SIZE = 16


def _vector_blob(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _vector_from_blob(blob: bytes, dimension: int) -> list[float]:
    expected_size = dimension * struct.calcsize("f")
    if len(blob) != expected_size:
        raise ValueError("Stored embedding has an unexpected size.")
    return list(struct.unpack(f"<{dimension}f", blob))


def _unit_id(file_id: str, unit: ContentUnit) -> str:
    locator = json.dumps(unit.locator, sort_keys=True, separators=(",", ":"))
    seed = f"{file_id}:{unit.unit_type}:{unit.ordinal}:{locator}:{unit.content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _embedding_payload(file: ScannedFile, unit: ContentUnit) -> str:
    chunk_index = unit.locator.get("chunk_index", 0)
    context = f"Filename: {file.filename}\nPath: {file.relative_path}\n" if chunk_index == 0 else ""
    return f"{EMBEDDING_CACHE_VERSION}\n{context}{unit.text}"


def _embedding_input_hash(payload: str) -> str:
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _model_digest(provider: EmbeddingProvider) -> str:
    return str(getattr(provider, "model_digest", "unknown"))


def _cached_embedding(
    session,
    unit_id: str,
    input_hash: str,
    provider: EmbeddingProvider,
) -> EmbeddingRecord | None:
    exact = session.exec(
        select(EmbeddingRecord).where(
            EmbeddingRecord.unit_id == unit_id,
            EmbeddingRecord.provider == provider.provider_name,
            EmbeddingRecord.model_name == provider.model_name,
            EmbeddingRecord.input_hash == input_hash,
        )
    ).first()
    if exact:
        return exact

    # Content-based fallback allows a chunk to be reused when an earlier
    # insertion changes the ordinal of later chunks.
    return session.exec(
        select(EmbeddingRecord).where(
            EmbeddingRecord.provider == provider.provider_name,
            EmbeddingRecord.model_name == provider.model_name,
            EmbeddingRecord.input_hash == input_hash,
        )
    ).first()


def _store_embedding(session, unit_id: str, input_hash: str, vector: list[float], provider: EmbeddingProvider) -> None:
    session.add(
        EmbeddingRecord(
            embedding_id=str(uuid.uuid4()),
            unit_id=unit_id,
            provider=provider.provider_name,
            model_name=provider.model_name,
            model_digest=_model_digest(provider),
            dimension=len(vector),
            vector_blob=_vector_blob(vector),
            input_hash=input_hash,
            created_at=datetime.now(timezone.utc),
        )
    )


def index_scan(scan_result: ScanResult, database_path: Path, provider: EmbeddingProvider) -> dict[str, int]:
    engine = create_database(database_path)
    embedded = 0
    reused = 0
    units_seen = 0

    with open_session(engine) as session:
        for file in scan_result.ready_files:
            session.merge(
                FileRecord(
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
            )

            pending: list[tuple[ContentUnit, str, str, str]] = []
            for unit in file.units:
                units_seen += 1
                unit_id = _unit_id(file.file_id, unit)
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
                payload = _embedding_payload(file, unit)
                input_hash = _embedding_input_hash(payload)
                existing = _cached_embedding(session, unit_id, input_hash, provider)
                if existing:
                    if existing.unit_id != unit_id:
                        _store_embedding(
                            session,
                            unit_id,
                            input_hash,
                            _vector_from_blob(existing.vector_blob, existing.dimension),
                            provider,
                        )
                    reused += 1
                    continue
                pending.append((unit, unit_id, payload, input_hash))

            for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
                batch = pending[start : start + EMBEDDING_BATCH_SIZE]
                vectors = provider.embed([payload for _, _, payload, _ in batch])
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding provider returned an incomplete batch.")
                for (_, unit_id, _, input_hash), vector in zip(batch, vectors, strict=True):
                    _store_embedding(session, unit_id, input_hash, vector, provider)
                    embedded += 1

        session.commit()

    return {"files": len(scan_result.ready_files), "units": units_seen, "embedded": embedded, "reused": reused}


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def load_cached_file_vectors(
    scan_result: ScanResult,
    database_path: Path,
    provider: EmbeddingProvider,
) -> dict[str, list[float]]:
    """Aggregate current chunk vectors into one vector per scanned file."""

    engine = create_database(database_path)
    vectors_by_file: dict[str, list[list[float]]] = {}
    with open_session(engine) as session:
        for file in scan_result.ready_files:
            current_vectors: list[list[float]] = []
            for unit in file.units:
                unit_id = _unit_id(file.file_id, unit)
                payload_hash = _embedding_input_hash(_embedding_payload(file, unit))
                record = session.exec(
                    select(EmbeddingRecord).where(
                        EmbeddingRecord.unit_id == unit_id,
                        EmbeddingRecord.provider == provider.provider_name,
                        EmbeddingRecord.model_name == provider.model_name,
                        EmbeddingRecord.input_hash == payload_hash,
                    )
                ).first()
                if record:
                    current_vectors.append(_vector_from_blob(record.vector_blob, record.dimension))
            if current_vectors:
                dimension = len(current_vectors[0])
                mean = [sum(vector[index] for vector in current_vectors) / len(current_vectors) for index in range(dimension)]
                vectors_by_file[file.file_id] = _normalize(mean)
    return vectors_by_file
