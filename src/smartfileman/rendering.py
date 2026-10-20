import re
from collections import defaultdict

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .schemas import Plan, PlanOperation


STATUS_STYLE = {
    "ready": "green",
    "needs-review": "yellow",
    "conflict": "red",
}


def _status_text(operation: PlanOperation) -> Text:
    label = operation.status.replace("-", " ").upper()
    return Text(label, style=STATUS_STYLE.get(operation.status, "white"))


def _operation_table(operations: list[PlanOperation]) -> Table:
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Status", width=12)
    table.add_column("ID", width=10)
    table.add_column("Source", overflow="fold")
    table.add_column("Destination", overflow="fold")
    table.add_column("Peer similarity", justify="right", width=16)
    for operation in operations:
        similarity = "—" if "no peer support" in operation.reason else f"{operation.confidence:.0%}"
        table.add_row(
            _status_text(operation),
            operation.operation_id[:8],
            operation.source_relative,
            operation.destination_relative,
            similarity,
        )
    return table


def _evidence_from_reason(reason: str) -> list[str]:
    match = re.search(r"evidence:\s*([^;]+)", reason, flags=re.IGNORECASE)
    if not match:
        return []
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", match.group(1).lower())
        if token not in {"similar", "embeddings", "content"}
    ]


def _group_panel(subject: str, category: str, operations: list[PlanOperation], details: bool) -> Panel:
    evidence: list[str] = []
    for operation in operations:
        for token in _evidence_from_reason(operation.reason):
            if token not in evidence:
                evidence.append(token)
    evidence = evidence[:8]
    summary = Text()
    summary.append(f"{subject} / {category}\n", style="bold")
    summary.append(f"{len(operations)} file(s) proposed\n", style="dim")
    if details:
        summary.append(f"Evidence: {' '.join(evidence) or 'content similarity'}\n", style="cyan")
        for operation in operations:
            if operation.status != "ready":
                summary.append(f"{operation.status}: {operation.reason}\n", style=STATUS_STYLE.get(operation.status, "white"))
    return Panel(Group(summary, _operation_table(operations)), border_style="blue")


def render_plan(console: Console, plan: Plan, details: bool = True) -> None:
    operations_by_group: dict[tuple[str, str], list[PlanOperation]] = defaultdict(list)
    for operation in plan.operations:
        operations_by_group[(operation.subject, operation.category)].append(operation)

    ready = sum(operation.status == "ready" for operation in plan.operations)
    review = sum(operation.status == "needs-review" for operation in plan.operations)
    conflicts = sum(operation.status == "conflict" for operation in plan.operations)
    console.print(
        Panel(
            f"[bold]Root:[/bold] {plan.root_path}\n"
            f"[bold]Mode:[/bold] {plan.mode}\n"
            f"[bold]Plan:[/bold] {plan.plan_id}\n\n"
            f"[green]{ready} ready[/green] · [yellow]{review} needs review[/yellow] · [red]{conflicts} conflicts[/red]",
            title="SMARTFILE PREVIEW",
            border_style="cyan",
        )
    )
    if not operations_by_group:
        console.print("[dim]No supported files were found.[/dim]")
        return
    for (subject, category), operations in sorted(operations_by_group.items()):
        console.print(_group_panel(subject, category, sorted(operations, key=lambda item: item.source_relative), details))


def filtered_plan(plan: Plan, operation_ids: set[str]) -> Plan:
    return plan.model_copy(update={"operations": [operation for operation in plan.operations if operation.operation_id in operation_ids]})


def render_apply_results(console: Console, results: list[dict[str, str]]) -> None:
    table = Table("Operation", "Status", "Move ID")
    for result in results:
        table.add_row(result.get("operation_id", "-"), result.get("status", "-"), result.get("move_id", "-"))
    console.print(Panel(table, title="APPLY RESULTS", border_style="green"))
