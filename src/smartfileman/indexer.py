import hashlib
import json
import math
import struct
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from .db import create_database, open_session
from .embeddings.base import EmbeddingProvider
from .models import ContentUnitRecord, EmbeddingRecord, FileRecord
from .readers import MEDIA_EXTENSIONS, normalize_media_filename
from .schemas import ContentUnit, ScanResult, ScannedFile

EMBEDDING_CACHE_VERSION = "cache-v4-content-first-no-source-path"
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_QUERY_BATCH_SIZE = 500
ProgressCallback = Callable[[str, float], None]


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
    if chunk_index == 0 and file.extension in MEDIA_EXTENSIONS:
        context = f"Media title: {normalize_media_filename(Path(file.filename))}\n"
    elif chunk_index == 0:
        # Semantic content must not inherit labels from an existing folder tree.
        context = f"Filename: {file.filename}\n"
    else:
        context = ""
    return f"{EMBEDDING_CACHE_VERSION}\n{context}{unit.text}"


def _embedding_input_hash(payload: str) -> str:
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _model_digest(provider: EmbeddingProvider, reset_unknown: bool = False) -> str:
    resolver = getattr(provider, "resolve_model_digest", None)
    if callable(resolver):
        digest = str(resolver())
    else:
        digest = str(getattr(provider, "model_digest", "unknown"))
    if digest and digest != "unknown":
        return digest

    if reset_unknown or not getattr(provider, "_run_cache_identity", None):
        provider._run_cache_identity = f"run:{uuid.uuid4()}"
    return provider._run_cache_identity


def _load_cached_embeddings(
    session,
    input_hashes: set[str],
    provider: EmbeddingProvider,
    model_digest: str,
) -> tuple[dict[tuple[str, str], EmbeddingRecord], dict[str, EmbeddingRecord]]:
    """Load candidate cache rows in bounded queries instead of one query per unit."""

    exact: dict[tuple[str, str], EmbeddingRecord] = {}
    by_input: dict[str, EmbeddingRecord] = {}
    hashes = sorted(input_hashes)
    for start in range(0, len(hashes), EMBEDDING_QUERY_BATCH_SIZE):
        batch = hashes[start : start + EMBEDDING_QUERY_BATCH_SIZE]
        conditions = [
            EmbeddingRecord.provider == provider.provider_name,
            EmbeddingRecord.model_name == provider.model_name,
            EmbeddingRecord.input_hash.in_(batch),
        ]
        conditions.append(EmbeddingRecord.model_digest == model_digest)
        records = session.exec(select(EmbeddingRecord).where(*conditions)).all()
        for record in records:
            exact.setdefault((record.unit_id, record.input_hash), record)
            by_input.setdefault(record.input_hash, record)
    return exact, by_input


def _store_embedding(
    session,
    unit_id: str,
    input_hash: str,
    vector: list[float],
    provider: EmbeddingProvider,
    model_digest: str,
) -> None:
    session.add(
        EmbeddingRecord(
            embedding_id=str(uuid.uuid4()),
            unit_id=unit_id,
            provider=provider.provider_name,
            model_name=provider.model_name,
            model_digest=model_digest,
            dimension=len(vector),
            vector_blob=_vector_blob(vector),
            input_hash=input_hash,
            created_at=datetime.now(timezone.utc),
        )
    )


def index_scan(
    scan_result: ScanResult,
    database_path: Path,
    provider: EmbeddingProvider,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    engine = create_database(database_path)
    embedded = 0
    reused = 0
    units_seen = 0

    with open_session(engine) as session:
        model_digest = _model_digest(provider, reset_unknown=True)
        work_items: list[tuple[ScannedFile, ContentUnit, str, str, str]] = []
        total_files = max(1, len(scan_result.ready_files))
        if progress_callback:
            progress_callback("Indexing content and embeddings", 0.0)
        for file_index, file in enumerate(scan_result.ready_files, start=1):
            for unit in file.units:
                units_seen += 1
                unit_id = _unit_id(file.file_id, unit)
                payload = _embedding_payload(file, unit)
                input_hash = _embedding_input_hash(payload)
                work_items.append((file, unit, unit_id, payload, input_hash))
            if progress_callback:
                progress_callback("Indexing content and embeddings", 0.45 * file_index / total_files)

        file_ids = sorted(file.file_id for file in scan_result.ready_files)
        existing_files: dict[str, FileRecord] = {}
        for start in range(0, len(file_ids), EMBEDDING_QUERY_BATCH_SIZE):
            batch = file_ids[start : start + EMBEDDING_QUERY_BATCH_SIZE]
            existing_files.update(
                {record.file_id: record for record in session.exec(select(FileRecord).where(FileRecord.file_id.in_(batch))).all()}
            )
        now = datetime.now(timezone.utc)
        for file in scan_result.ready_files:
            current = existing_files.get(file.file_id)
            values = {
                "root_path": file.root_path,
                "relative_path": file.relative_path,
                "filename": file.filename,
                "extension": file.extension,
                "mime_type": file.mime_type,
                "content_hash": file.content_hash,
                "size_bytes": file.size_bytes,
                "modified_time_ns": file.modified_time_ns,
                "status": file.status,
            }
            if current is None:
                session.add(FileRecord(file_id=file.file_id, **values, updated_at=now))
            elif any(getattr(current, field) != value for field, value in values.items()):
                for field, value in values.items():
                    setattr(current, field, value)
                current.updated_at = now

        unit_ids = sorted(unit_id for _, _, unit_id, _, _ in work_items)
        existing_units: dict[str, ContentUnitRecord] = {}
        for start in range(0, len(unit_ids), EMBEDDING_QUERY_BATCH_SIZE):
            batch = unit_ids[start : start + EMBEDDING_QUERY_BATCH_SIZE]
            existing_units.update(
                {record.unit_id: record for record in session.exec(select(ContentUnitRecord).where(ContentUnitRecord.unit_id.in_(batch))).all()}
            )
        for file, unit, unit_id, _, _ in work_items:
            locator_json = json.dumps(unit.locator, sort_keys=True)
            values = {
                "file_id": file.file_id,
                "unit_type": unit.unit_type,
                "ordinal": unit.ordinal,
                "content_hash": unit.content_hash,
                "text": unit.text,
                "locator_json": locator_json,
                "extractor_version": unit.extractor_version,
            }
            current = existing_units.get(unit_id)
            if current is None:
                session.add(ContentUnitRecord(unit_id=unit_id, **values))
            elif any(getattr(current, field) != value for field, value in values.items()):
                for field, value in values.items():
                    setattr(current, field, value)

        input_hashes = {input_hash for _, _, _, _, input_hash in work_items}
        exact_cache, input_cache = _load_cached_embeddings(session, input_hashes, provider, model_digest)
        pending: list[tuple[ContentUnit, str, str, str]] = []
        for _, unit, unit_id, payload, input_hash in work_items:
            existing = exact_cache.get((unit_id, input_hash)) or input_cache.get(input_hash)
            if existing:
                vector = _vector_from_blob(existing.vector_blob, existing.dimension)
                if existing.unit_id != unit_id:
                    _store_embedding(session, unit_id, input_hash, vector, provider, model_digest)
                reused += 1
                continue
            pending.append((unit, unit_id, payload, input_hash))

        if progress_callback:
            progress_callback("Indexing content and embeddings", 0.5)
        for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
            batch = pending[start : start + EMBEDDING_BATCH_SIZE]
            vectors = provider.embed([payload for _, _, payload, _ in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding provider returned an incomplete batch.")
            for (_, unit_id, _, input_hash), vector in zip(batch, vectors, strict=True):
                _store_embedding(session, unit_id, input_hash, vector, provider, model_digest)
                embedded += 1
            if progress_callback:
                progress_callback(
                    "Indexing content and embeddings",
                    0.5 + 0.5 * min(1.0, (start + len(batch)) / max(1, len(pending))),
                )

        session.commit()
        if progress_callback:
            progress_callback("Indexing content and embeddings", 1.0)

    return {"files": len(scan_result.ready_files), "units": units_seen, "embedded": embedded, "reused": reused}


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def load_cached_file_vectors(
    scan_result: ScanResult,
    database_path: Path,
    provider: EmbeddingProvider,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list[float]]:
    """Aggregate current chunk vectors into one vector per scanned file."""

    engine = create_database(database_path)
    vectors_by_file: dict[str, list[float]] = {}
    with open_session(engine) as session:
        model_digest = _model_digest(provider)
        unit_refs: list[tuple[str, str, str]] = []
        for file in scan_result.ready_files:
            for unit in file.units:
                unit_id = _unit_id(file.file_id, unit)
                payload_hash = _embedding_input_hash(_embedding_payload(file, unit))
                unit_refs.append((file.file_id, unit_id, payload_hash))

        records_by_key: dict[tuple[str, str], EmbeddingRecord] = {}
        unit_ids = sorted({unit_id for _, unit_id, _ in unit_refs})
        if progress_callback:
            progress_callback("Loading cached file vectors", 0.0)
        for start in range(0, len(unit_ids), EMBEDDING_QUERY_BATCH_SIZE):
            batch = unit_ids[start : start + EMBEDDING_QUERY_BATCH_SIZE]
            conditions = [
                EmbeddingRecord.provider == provider.provider_name,
                EmbeddingRecord.model_name == provider.model_name,
                EmbeddingRecord.unit_id.in_(batch),
            ]
            conditions.append(EmbeddingRecord.model_digest == model_digest)
            records = session.exec(select(EmbeddingRecord).where(*conditions)).all()
            for record in records:
                existing = records_by_key.get((record.unit_id, record.input_hash))
                if existing is None or record.created_at >= existing.created_at:
                    records_by_key[(record.unit_id, record.input_hash)] = record
            if progress_callback:
                progress_callback(
                    "Loading cached file vectors",
                    min(1.0, (start + len(batch)) / max(1, len(unit_ids))),
                )

        vectors_by_file_refs: dict[str, list[list[float]]] = {}
        for file_id, unit_id, payload_hash in unit_refs:
            record = records_by_key.get((unit_id, payload_hash))
            if record:
                vectors_by_file_refs.setdefault(file_id, []).append(
                    _vector_from_blob(record.vector_blob, record.dimension)
                )

        for file_id, current_vectors in vectors_by_file_refs.items():
            if current_vectors:
                dimension = len(current_vectors[0])
                mean = [sum(vector[index] for vector in current_vectors) / len(current_vectors) for index in range(dimension)]
                vectors_by_file[file_id] = _normalize(mean)
        if progress_callback:
            progress_callback("Loading cached file vectors", 1.0)
    return vectors_by_file
