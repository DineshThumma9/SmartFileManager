from pathlib import Path

from .config import Settings, get_settings
from .embeddings import FakeEmbeddingProvider, OllamaEmbeddingProvider
from .indexer import index_scan, load_cached_file_vectors
from .operations import apply_plan, load_move, load_plan, save_plan, undo_move
from .paths import UnsafePathError, resolve_root
from .planner import build_plan
from .scanner import scan


def ensure_allowed_root(root: str | Path, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    resolved = resolve_root(root)
    if not any(resolved == allowed or allowed in resolved.parents for allowed in settings.allowed_root_paths()):
        raise UnsafePathError(f"Selected root is not allowed: {resolved}")
    return resolved


def provider_for(kind: str, settings: Settings | None = None, model_name: str | None = None):
    settings = settings or get_settings()
    if kind == "fake":
        return FakeEmbeddingProvider()
    if kind == "ollama":
        return OllamaEmbeddingProvider(settings.ollama_host, model_name or settings.ollama_model)
    raise ValueError(f"Unknown embedding provider: {kind}")


def make_preview(
    root: str | Path,
    mode: str = "fuzzy",
    provider_kind: str = "ollama",
    settings: Settings | None = None,
    embedding_model: str | None = None,
):
    settings = settings or get_settings()
    resolved_root = ensure_allowed_root(root, settings)
    scan_result = scan(resolved_root)
    provider = provider_for(provider_kind, settings, embedding_model) if mode == "semantic" else None
    file_vectors = None
    if provider is not None:
        database_path = settings.storage_dir(resolved_root) / "index.sqlite3"
        index_scan(scan_result, database_path, provider)
        file_vectors = load_cached_file_vectors(scan_result, database_path, provider)
    plan = build_plan(scan_result, mode=mode, file_vectors=file_vectors)
    save_plan(plan, settings.storage_dir(resolved_root))
    return plan


def index(
    root: str | Path,
    provider_kind: str = "ollama",
    settings: Settings | None = None,
    embedding_model: str | None = None,
):
    settings = settings or get_settings()
    resolved_root = ensure_allowed_root(root, settings)
    provider = provider_for(provider_kind, settings, embedding_model)
    result = index_scan(scan(resolved_root), settings.storage_dir(resolved_root) / "index.sqlite3", provider)
    return {"root_path": str(resolved_root), **result}


def apply(root: str | Path, plan_id: str, operation_ids: set[str], approved: bool, settings: Settings | None = None):
    settings = settings or get_settings()
    resolved_root = ensure_allowed_root(root, settings)
    plan = load_plan(plan_id, settings.storage_dir(resolved_root))
    if Path(plan.root_path).resolve() != resolved_root:
        raise UnsafePathError("Plan root does not match the selected root.")
    return apply_plan(plan, settings.storage_dir(resolved_root), operation_ids, approved)


def undo(root: str | Path, move_id: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    resolved_root = ensure_allowed_root(root, settings)
    move = load_move(move_id, settings.storage_dir(resolved_root))
    if Path(move.root_path).resolve() != resolved_root:
        raise UnsafePathError("Move root does not match the selected root.")
    undo_move(move, settings.storage_dir(resolved_root))
