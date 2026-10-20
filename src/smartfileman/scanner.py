import hashlib
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path

from .paths import reject_symlink, resolve_root
from .readers import (
    MAX_CHUNK_CHARS,
    MEDIA_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    extract_units,
    extension_for,
    mime_type_for,
    normalize_media_filename,
)
from .schemas import ContentUnit, ScannedFile, ScanResult

EXTRACTION_CACHE_VERSION = "extract-v2-media-normalized"
ProgressCallback = Callable[[str, float], None]


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def stable_file_id(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"smartfile:{root.resolve()}:{relative}"))


def _extraction_cache_key(content_digest: str, max_chunk_chars: int, path: Path) -> str:
    media_identity = normalize_media_filename(path) if extension_for(path) in MEDIA_EXTENSIONS else ""
    return f"{EXTRACTION_CACHE_VERSION}:{max_chunk_chars}:{media_identity}:{content_digest}"


def _load_extraction_cache(path: Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        if not isinstance(entries, dict):
            return {}
        return {
            str(key): value
            for key, value in entries.items()
            if isinstance(value, list) and all(isinstance(item, dict) for item in value)
        }
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _save_extraction_cache(path: Path | None, entries: dict[str, list[dict[str, object]]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = {"version": EXTRACTION_CACHE_VERSION, "entries": entries}
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _candidate_paths(root: Path, excluded_paths: tuple[Path, ...] = ()):
    excluded = {path.resolve() for path in excluded_paths if path.resolve() != root}
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in {".smartfile", "_organized"}
            and (directory_path / name).resolve() not in excluded
            and not (directory_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            yield directory_path / filename


def scan(
    root: str | Path,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    extraction_cache_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    excluded_paths: tuple[str | Path, ...] = (),
) -> ScanResult:
    resolved_root = resolve_root(root)
    files: list[ScannedFile] = []
    cache_path = Path(extraction_cache_path) if extraction_cache_path is not None else None
    extraction_cache = _load_extraction_cache(cache_path)
    cache_changed = False

    candidate_paths = list(_candidate_paths(resolved_root, tuple(Path(item) for item in excluded_paths)))
    total_paths = max(1, len(candidate_paths))
    if progress_callback:
        progress_callback("Scanning files", 0.0)
    for path_index, path in enumerate(candidate_paths, start=1):
        if not path.is_file() or path.is_symlink():
            continue
        relative_parts = path.relative_to(resolved_root).parts
        if {".smartfile", "_organized"}.intersection(relative_parts) or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        reject_symlink(path)
        relative = path.relative_to(resolved_root).as_posix()
        digest = content_hash(path)
        try:
            cache_key = _extraction_cache_key(digest, max_chunk_chars, path)
            cached_units = extraction_cache.get(cache_key)
            if cached_units is not None:
                try:
                    units = [ContentUnit.model_validate(unit) for unit in cached_units]
                except Exception:
                    units = extract_units(path, max_chunk_chars=max_chunk_chars)
                    if content_hash(path) != digest:
                        raise RuntimeError("Source changed during extraction.")
                    extraction_cache[cache_key] = [unit.model_dump(mode="json") for unit in units]
                    cache_changed = True
            else:
                units = extract_units(path, max_chunk_chars=max_chunk_chars)
                if content_hash(path) != digest:
                    raise RuntimeError("Source changed during extraction.")
                extraction_cache[cache_key] = [unit.model_dump(mode="json") for unit in units]
                cache_changed = True
            files.append(
                ScannedFile(
                    file_id=stable_file_id(resolved_root, path),
                    root_path=str(resolved_root),
                    relative_path=relative,
                    filename=path.name,
                    extension=extension_for(path),
                    mime_type=mime_type_for(path),
                    content_hash=digest,
                    size_bytes=path.stat().st_size,
                    modified_time_ns=path.stat().st_mtime_ns,
                    units=units,
                )
            )
        except Exception as error:  # A bad file must not abort the whole scan.
            files.append(
                ScannedFile(
                    file_id=stable_file_id(resolved_root, path),
                    root_path=str(resolved_root),
                    relative_path=relative,
                    filename=path.name,
                    extension=extension_for(path),
                    mime_type=mime_type_for(path),
                    content_hash=digest,
                    size_bytes=path.stat().st_size,
                    modified_time_ns=path.stat().st_mtime_ns,
                    status="error",
                    error=str(error),
                )
            )

        if progress_callback:
            progress_callback("Scanning files", path_index / total_paths)

    if cache_changed:
        _save_extraction_cache(cache_path, extraction_cache)
    if progress_callback:
        progress_callback("Scanning files", 1.0)
    return ScanResult(root_path=str(resolved_root), files=files)
