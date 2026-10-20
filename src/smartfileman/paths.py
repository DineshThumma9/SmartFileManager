import os
from pathlib import Path


class UnsafePathError(ValueError):
    pass


def resolve_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise UnsafePathError(f"Source root is not a directory: {root}")
    return path


def ensure_within(root: Path, candidate: Path, label: str = "path") -> Path:
    resolved_root = root.resolve()
    lexical_candidate = candidate.absolute()
    try:
        relative_candidate = lexical_candidate.relative_to(resolved_root)
    except ValueError:
        relative_candidate = None
    if relative_candidate is not None:
        current = resolved_root
        for part in relative_candidate.parts:
            if part in {".", ".."}:
                raise UnsafePathError(f"{label} contains traversal components: {candidate}")
            current /= part
            if current.is_symlink():
                raise UnsafePathError(f"{label} contains an unsupported symlink: {candidate}")
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise UnsafePathError(f"{label} is outside the selected root: {candidate}") from error
    return resolved_candidate


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise UnsafePathError(f"Symlinks are not supported: {path}")


def safe_destination(root: Path, destination_relative: str) -> Path:
    if os.path.isabs(destination_relative):
        raise UnsafePathError("Destination must be relative to the selected root")
    return ensure_within(root, root / destination_relative, "destination")
