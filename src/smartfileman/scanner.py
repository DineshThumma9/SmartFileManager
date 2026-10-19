import hashlib
import uuid
from pathlib import Path

from .paths import reject_symlink, resolve_root
from .readers import MAX_CHUNK_CHARS, SUPPORTED_EXTENSIONS, extract_units, extension_for, mime_type_for
from .schemas import ScannedFile, ScanResult


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def stable_file_id(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"smartfile:{root.resolve()}:{relative}"))


def scan(root: str | Path, max_chunk_chars: int = MAX_CHUNK_CHARS) -> ScanResult:
    resolved_root = resolve_root(root)
    files: list[ScannedFile] = []

    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if {".smartfile", "_organized"}.intersection(path.parts) or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        reject_symlink(path)
        relative = path.relative_to(resolved_root).as_posix()
        try:
            units = extract_units(path, max_chunk_chars=max_chunk_chars)
            files.append(
                ScannedFile(
                    file_id=stable_file_id(resolved_root, path),
                    root_path=str(resolved_root),
                    relative_path=relative,
                    filename=path.name,
                    extension=extension_for(path),
                    mime_type=mime_type_for(path),
                    content_hash=content_hash(path),
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
                    content_hash=content_hash(path),
                    size_bytes=path.stat().st_size,
                    modified_time_ns=path.stat().st_mtime_ns,
                    status="error",
                    error=str(error),
                )
            )

    return ScanResult(root_path=str(resolved_root), files=files)
