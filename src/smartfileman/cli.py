import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .operations import load_plan
from .scanner import scan
from .service import apply, index, make_preview, undo

app = typer.Typer(help="Previewable local-first smart file organization.")
console = Console()


@app.command("scan")
def scan_command(root: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Scan a root and display supported files without changing them."""
    result = scan(root)
    table = Table("File", "Type", "Units", "Status")
    for file in result.files:
        table.add_row(file.relative_path, file.extension, str(len(file.units)), file.status)
    console.print(table)
    console.print(f"[bold]{len(result.files)}[/bold] supported file(s)")


@app.command("index")
def index_command(
    root: Path = typer.Argument(..., exists=True, file_okay=False),
    provider: str = typer.Option("ollama", help="Embedding provider: ollama or fake."),
):
    """Index content units and cache their embeddings."""
    result = index(root, provider)
    console.print_json(json.dumps(result))


@app.command()
def preview(
    root: Path = typer.Argument(..., exists=True, file_okay=False),
    mode: str = typer.Option("fuzzy", help="Organization mode: fuzzy or semantic."),
    provider: str = typer.Option("ollama", help="Embedding provider for semantic mode."),
    output: Path | None = typer.Option(None, help="Optional additional plan JSON path."),
):
    """Create and display an organization plan without moving files."""
    if mode not in {"fuzzy", "semantic"}:
        raise typer.BadParameter("mode must be fuzzy or semantic")
    plan = make_preview(root, mode, provider)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    table = Table("Operation", "Source", "Destination", "Confidence", "Status")
    for operation in plan.operations:
        table.add_row(
            operation.operation_id[:8],
            operation.source_relative,
            operation.destination_relative,
            f"{operation.confidence:.0%}",
            operation.status,
        )
    console.print(table)
    console.print(f"Plan: [bold]{plan.plan_id}[/bold]")


@app.command("apply")
def apply_command(
    root: Path = typer.Argument(..., exists=True, file_okay=False),
    plan_id: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Confirm all ready operations."),
    operation: list[str] = typer.Option([], "--operation", help="Specific operation ID; repeatable."),
):
    """Apply approved operations from a saved plan."""
    settings = get_settings()
    plan = load_plan(plan_id, settings.storage_dir(root.resolve()))
    selected = set(operation) or {item.operation_id for item in plan.operations if item.status == "ready"}
    approved = yes or typer.confirm(f"Apply {len(selected)} operation(s)?")
    results = apply(root, plan_id, selected, approved)
    console.print_json(json.dumps(results))


@app.command("undo")
def undo_command(
    root: Path = typer.Argument(..., exists=True, file_okay=False),
    move_id: str = typer.Argument(...),
):
    """Undo one recorded move."""
    undo(root, move_id)
    console.print(f"Undone: {move_id}")


def main() -> None:
    app()
