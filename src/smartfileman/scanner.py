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


def _verify_unchanged(path: Path, digest: str, stat_before_extraction: os.stat_result) -> None:
    """Confirm the file did not change while it was being read.

    Hashing the file a second time would catch any change, but it doubles
    the read cost of every file in the collection. The size and modification
    time are compared first, and only a file whose metadata moved is hashed
    again. A file that changed while preserving both its size and its
    modification time is a deliberate attempt to defeat the check, and the
    second hash catches it.
    """

    stat_after = path.stat()
    if (
        stat_after.st_size == stat_before_extraction.st_size
        and stat_after.st_mtime_ns == stat_before_extraction.st_mtime_ns
    ):
        return
    if content_hash(path) != digest:
        raise RuntimeError("Source changed during extraction.")


def _hash_stable_file(path: Path) -> tuple[str, os.stat_result]:
    """Hash a file and return the metadata that proves the hash is current.

    Hashing a file that is being written concurrently produces a digest of
    no particular version of the file. The size and modification time are
    recorded before and after the read, and a file whose metadata moved
    during the read is rejected rather than indexed under a digest that
    describes neither the old nor the new content.
    """

    stat_before = path.stat()
    digest = content_hash(path)
    stat_after = path.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError("Source changed while it was being read.")
    return digest, stat_after


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
        digest = ""
        try:
            digest, stat_after_hash = _hash_stable_file(path)
            cache_key = _extraction_cache_key(digest, max_chunk_chars, path)
            cached_units = extraction_cache.get(cache_key)
            if cached_units is not None:
                try:
                    units = [ContentUnit.model_validate(unit) for unit in cached_units]
                except Exception:
                    units = extract_units(path, max_chunk_chars=max_chunk_chars)
                    _verify_unchanged(path, digest, stat_after_hash)
                    extraction_cache[cache_key] = [unit.model_dump(mode="json") for unit in units]
                    cache_changed = True
            else:
                units = extract_units(path, max_chunk_chars=max_chunk_chars)
                _verify_unchanged(path, digest, stat_after_hash)
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
                    size_bytes=stat_after_hash.st_size,
                    modified_time_ns=stat_after_hash.st_mtime_ns,
                    units=units,
                )
            )
        except Exception as error:  # A bad file must not abort the whole scan.
            try:
                failed_stat = path.stat()
            except OSError:
                failed_stat = None
            files.append(
                ScannedFile(
                    file_id=stable_file_id(resolved_root, path),
                    root_path=str(resolved_root),
                    relative_path=relative,
                    filename=path.name,
                    extension=extension_for(path),
                    mime_type=mime_type_for(path),
                    # The digest may be unknown if hashing itself failed, but
                    # a file must always carry a well-formed placeholder so
                    # that the rest of the pipeline can rely on the field.
                    content_hash=digest if digest else "sha256:" + "0" * 64,
                    size_bytes=failed_stat.st_size if failed_stat else 0,
                    modified_time_ns=failed_stat.st_mtime_ns if failed_stat else 0,
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
