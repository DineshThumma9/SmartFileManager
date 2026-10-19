from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .operations import load_plan
from .service import apply, index, make_preview, undo

mcp = FastMCP("smartfileman")


@mcp.tool()
def smartfile_preview(root: str, mode: str = "fuzzy", provider: str = "ollama") -> dict:
    """Create a non-destructive organization plan and return its operations."""
    return make_preview(root, mode, provider).model_dump(mode="json")


@mcp.tool()
def smartfile_get_plan(root: str, plan_id: str) -> dict:
    """Retrieve a saved plan by ID."""
    settings = get_settings()
    return load_plan(plan_id, settings.storage_dir(Path(root).resolve())).model_dump(mode="json")


@mcp.tool()
def smartfile_index(root: str, provider: str = "ollama") -> dict:
    """Index files and reuse cached embeddings when possible."""
    return index(root, provider)


@mcp.tool()
def smartfile_apply(root: str, plan_id: str, operation_ids: list[str], approved: bool = False) -> list[dict[str, str]]:
    """Apply selected operations from a saved plan after explicit approval."""
    return apply(root, plan_id, set(operation_ids), approved)


@mcp.tool()
def smartfile_undo(root: str, move_id: str) -> dict[str, str]:
    """Undo a recorded move after verifying its content hash."""
    undo(root, move_id)
    return {"move_id": move_id, "status": "undone"}


def main() -> None:
    mcp.run(transport="stdio")
