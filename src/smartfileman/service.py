from pathlib import Path
from collections.abc import Callable

from .config import Settings, get_settings
from .embeddings import FakeEmbeddingProvider, OllamaEmbeddingProvider
from .indexer import index_scan, load_cached_file_vectors, load_cached_identity_vectors
from .naming import DeterministicClusterNamer, OllamaClusterNamer
from .operations import apply_plan, load_move, load_plan, save_plan, undo_move
from .paths import UnsafePathError, resolve_root
from .planner import build_plan
from .scanner import scan

ProgressCallback = Callable[[str, float], None]


def _mapped_progress(
    callback: ProgressCallback | None,
    start: float,
    end: float,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def report(stage: str, fraction: float) -> None:
        callback(stage, start + (end - start) * max(0.0, min(1.0, fraction)))

    return report


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


def namer_for(
    kind: str,
    settings: Settings | None = None,
    model_name: str | None = None,
    cache_path: Path | None = None,
):
    settings = settings or get_settings()
    if kind == "ollama":
        return OllamaClusterNamer(
            settings.ollama_host,
            model_name or settings.ollama_naming_model or None,
            cache_path=cache_path,
            timeout_seconds=settings.ollama_naming_timeout_seconds,
        )
    return DeterministicClusterNamer()


def make_preview(
    root: str | Path,
    mode: str = "fuzzy",
    provider_kind: str = "ollama",
    settings: Settings | None = None,
    embedding_model: str | None = None,
    naming_model: str | None = None,
    progress_callback: ProgressCallback | None = None,
    naming_provider: str | None = None,
):
    settings = settings or get_settings()
    resolved_root = ensure_allowed_root(root, settings)
    extraction_cache = settings.storage_dir(resolved_root) / "extraction-cache.json"
    scan_result = scan(
        resolved_root,
        extraction_cache_path=extraction_cache,
        excluded_paths=(settings.storage_dir(resolved_root),),
        progress_callback=_mapped_progress(progress_callback, 0.0, 20.0 if mode == "semantic" else 60.0),
    )
    provider = provider_for(provider_kind, settings, embedding_model) if mode == "semantic" else None
    file_vectors = None
    identity_vectors = None
    # The naming provider is chosen independently of the embedding provider so
    # that offline fake embeddings can be paired with a real naming model.
    resolved_naming_provider = naming_provider
    if resolved_naming_provider is None:
        resolved_naming_provider = provider_kind if naming_model else "deterministic"
    namer = (
        namer_for(
            resolved_naming_provider,
            settings,
            naming_model,
            cache_path=settings.storage_dir(resolved_root) / "naming-cache.json",
        )
        if mode == "semantic"
        else None
    )
    if provider is not None:
        database_path = settings.storage_dir(resolved_root) / "index.sqlite3"
        index_scan(
            scan_result,
            database_path,
            provider,
            progress_callback=_mapped_progress(progress_callback, 20.0, 60.0),
        )
        file_vectors = load_cached_file_vectors(
            scan_result,
            database_path,
            provider,
            progress_callback=_mapped_progress(progress_callback, 60.0, 72.0),
        )
        identity_vectors = load_cached_identity_vectors(scan_result, database_path, provider)
        plan_progress = _mapped_progress(progress_callback, 72.0, 96.0)
    else:
        plan_progress = _mapped_progress(progress_callback, 60.0, 96.0)
    plan = build_plan(
        scan_result,
        mode=mode,
        file_vectors=file_vectors,
        namer=namer,
        progress_callback=plan_progress,
        identity_vectors=identity_vectors,
    )
    save_plan(plan, settings.storage_dir(resolved_root))
    if progress_callback:
        progress_callback("Preview ready", 100.0)
    return plan


def index(
    root: str | Path,
    provider_kind: str = "ollama",
    settings: Settings | None = None,
    embedding_model: str | None = None,
    progress_callback: ProgressCallback | None = None,
):
    settings = settings or get_settings()
    resolved_root = ensure_allowed_root(root, settings)
    provider = provider_for(provider_kind, settings, embedding_model)
    extraction_cache = settings.storage_dir(resolved_root) / "extraction-cache.json"
    result = index_scan(
        scan(
            resolved_root,
            extraction_cache_path=extraction_cache,
            excluded_paths=(settings.storage_dir(resolved_root),),
            progress_callback=_mapped_progress(progress_callback, 0.0, 30.0),
        ),
        settings.storage_dir(resolved_root) / "index.sqlite3",
        provider,
        progress_callback=_mapped_progress(progress_callback, 30.0, 100.0),
    )
    if progress_callback:
        progress_callback("Index ready", 100.0)
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
