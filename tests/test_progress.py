from pathlib import Path

from smartfileman.config import Settings
from smartfileman.scanner import scan
from smartfileman.service import make_preview


def _settings_for(root: Path) -> Settings:
    return Settings(data_dir=Path(".smartfile"), allowed_roots=str(root))


def test_scan_reports_progress_to_completion(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("motion and force", encoding="utf-8")
    updates: list[tuple[str, float]] = []

    scan(tmp_path, progress_callback=lambda stage, fraction: updates.append((stage, fraction)))

    assert updates
    assert updates[-1] == ("Scanning files", 1.0)


def test_semantic_preview_reports_user_facing_stages(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("motion and force", encoding="utf-8")
    updates: list[tuple[str, float]] = []

    make_preview(
        tmp_path,
        mode="semantic",
        provider_kind="fake",
        settings=_settings_for(tmp_path),
        progress_callback=lambda stage, percentage: updates.append((stage, percentage)),
    )

    stages = [stage for stage, _ in updates]
    assert stages[0] == "Scanning files"
    assert any(stage == "Indexing content and embeddings" for stage in stages)
    assert any(stage == "Loading cached file vectors" for stage in stages)
    assert any(stage.startswith("Naming cluster") for stage in stages)
    assert updates[-1] == ("Preview ready", 100.0)
