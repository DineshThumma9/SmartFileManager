import json
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from .config import get_settings
from .operations import OperationError, load_plan, resolve_operation_ids
from .rendering import filtered_plan, render_apply_results, render_plan
from .scanner import scan
from .service import apply, index, make_preview, undo


class OrganizationMode(str, Enum):
    fuzzy = "fuzzy"
    semantic = "semantic"


class EmbeddingProvider(str, Enum):
    ollama = "ollama"
    fake = "fake"


class NamingProvider(str, Enum):
    ollama = "ollama"
    deterministic = "deterministic"


app = typer.Typer(
    help=(
        "Local-first file organization with fuzzy and semantic modes.\n\n"
        "Examples:\n"
        "  smartfile scan ~/Downloads\n"
        "  smartfile preview ~/Downloads --mode semantic\n"
        "  smartfile organize ~/Downloads --mode fuzzy\n"
        "  smartfile apply ~/Downloads PLAN_ID"
    ),
    no_args_is_help=True,
)
console = Console()


def _with_progress(callback):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting", total=100)

        def report(stage: str, percentage: float) -> None:
            progress.update(task_id, description=stage, completed=max(0.0, min(100.0, percentage)))

        return callback(report)


def _root_argument() -> typer.Argument:
    return typer.Argument(..., exists=True, file_okay=False, help="Directory to scan or organize.")


@app.command("scan")
def scan_command(root: Path = _root_argument()):
    """Scan supported files and display detected units without changing files."""
    result = scan(root)
    file_table = Table("File", "Type", "Chunks", "Status")
    for file in result.files:
        file_table.add_row(file.relative_path, file.extension, str(len(file.units)), file.status)
    console.print(file_table)
    console.print(f"[bold]{len(result.files)}[/bold] supported file(s)")


def _provider_help() -> str:
    return "Embedding provider. ollama uses the local model; fake is deterministic and offline-friendly."


def _naming_provider_help() -> str:
    return (
        "Cluster naming provider. ollama asks a local model for a label; "
        "deterministic derives the label from file evidence only."
    )


@app.command("index")
def index_command(
    root: Path = _root_argument(),
    provider: EmbeddingProvider = typer.Option(EmbeddingProvider.ollama, help=_provider_help()),
    embedding_model: str | None = typer.Option(None, "--embedding-model", help="Override the Ollama embedding model tag."),
):
    """Extract chunks and cache embeddings for later semantic previews."""
    result = _with_progress(
        lambda report: index(
            root,
            provider.value,
            embedding_model=embedding_model,
            progress_callback=report,
        )
    )
    console.print_json(json.dumps(result))


@app.command()
def preview(
    root: Path = _root_argument(),
    mode: OrganizationMode = typer.Option(
        OrganizationMode.fuzzy,
        help="fuzzy uses names/paths; semantic uses chunk embeddings and inferred clusters.",
    ),
    provider: EmbeddingProvider = typer.Option(EmbeddingProvider.ollama, help=_provider_help()),
    embedding_model: str | None = typer.Option(None, "--embedding-model", help="Override the Ollama embedding model tag."),
    naming_model: str | None = typer.Option(None, "--naming-model", help="Optional local Gemma/Ollama model used only to name inferred clusters."),
    naming_provider: NamingProvider | None = typer.Option(None, "--naming-provider", help=_naming_provider_help()),
    output: Path | None = typer.Option(None, help="Also export the internal plan as JSON at this path."),
    details: bool = typer.Option(True, "--details/--no-details", help="Show evidence and conflict reasons in the visual preview."),
):
    """Create a visual organization preview without moving files."""
    plan = _with_progress(
        lambda report: make_preview(
            root,
            mode.value,
            provider.value,
            embedding_model=embedding_model,
            naming_model=naming_model,
            naming_provider=naming_provider.value if naming_provider else None,
            progress_callback=report,
        )
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    render_plan(console, plan, details=details)


def _select_operations(plan, selectors: list[str] | None, yes: bool) -> set[str]:
    if selectors:
        return resolve_operation_ids(plan, selectors)
    if yes:
        return resolve_operation_ids(plan)
    answer = typer.prompt("Select operations: all, none, or comma-separated IDs", default="all").strip()
    if answer.lower() == "none":
        return set()
    if answer.lower() == "all":
        return resolve_operation_ids(plan)
    return resolve_operation_ids(plan, [item.strip() for item in answer.split(",") if item.strip()])


@app.command("organize")
def organize_command(
    root: Path = _root_argument(),
    mode: OrganizationMode = typer.Option(OrganizationMode.fuzzy, help="fuzzy uses names/paths; semantic uses chunk embeddings."),
    provider: EmbeddingProvider = typer.Option(EmbeddingProvider.ollama, help=_provider_help()),
    embedding_model: str | None = typer.Option(None, "--embedding-model", help="Override the Ollama embedding model tag."),
    naming_model: str | None = typer.Option(None, "--naming-model", help="Optional local Gemma/Ollama model used only to name inferred clusters."),
    naming_provider: NamingProvider | None = typer.Option(None, "--naming-provider", help=_naming_provider_help()),
    yes: bool = typer.Option(False, "--yes", help="Select all ready operations and skip interactive confirmation."),
):
    """Preview, select, confirm, and apply organization in one visual workflow."""
    plan = _with_progress(
        lambda report: make_preview(
            root,
            mode.value,
            provider.value,
            embedding_model=embedding_model,
            naming_model=naming_model,
            naming_provider=naming_provider.value if naming_provider else None,
            progress_callback=report,
        )
    )
    render_plan(console, plan)
    try:
        selected = _select_operations(plan, None, yes)
    except OperationError as error:
        raise typer.BadParameter(str(error)) from error
    if not selected:
        console.print("[yellow]Cancelled; no files changed.[/yellow]")
        return
    console.print("\n[bold]Selected operations:[/bold]")
    render_plan(console, filtered_plan(plan, selected), details=True)
    if not yes and not typer.confirm(f"Apply {len(selected)} selected operation(s)?", default=False):
        console.print("[yellow]Cancelled; no files changed.[/yellow]")
        return
    results = apply(root, plan.plan_id, selected, approved=True)
    render_apply_results(console, results)


@app.command("apply")
def apply_command(
    root: Path = _root_argument(),
    plan_id: str = typer.Argument(..., help="Plan ID printed by preview or organize."),
    yes: bool = typer.Option(False, "--yes", help="Apply all ready operations without prompting; safety checks remain active."),
    operation: list[str] | None = typer.Option(None, "--operation", help="Full operation ID or unique prefix; repeat this option."),
):
    """Apply selected operations from a saved preview plan."""
    settings = get_settings()
    plan = load_plan(plan_id, settings.storage_dir(root.resolve()))
    try:
        selected = _select_operations(plan, operation, yes)
    except OperationError as error:
        raise typer.BadParameter(str(error)) from error
    if not selected:
        console.print("[yellow]No operations selected; no files changed.[/yellow]")
        return
    render_plan(console, filtered_plan(plan, selected), details=True)
    if not yes and not typer.confirm(f"Apply {len(selected)} selected operation(s) from {plan_id}?", default=False):
        console.print("[yellow]Cancelled; no files changed.[/yellow]")
        return
    results = apply(root, plan_id, selected, approved=True)
    render_apply_results(console, results)


@app.command("undo")
def undo_command(
    root: Path = _root_argument(),
    move_id: str = typer.Argument(..., help="Move ID printed after a successful apply."),
):
    """Undo one recorded move after verifying the file has not changed."""
    undo(root, move_id)
    console.print(f"[green]Undone:[/green] {move_id}")


def main() -> None:
    app()
