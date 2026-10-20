import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

from .paths import ensure_within, safe_destination
from .scanner import content_hash
from .schemas import MoveRecord, Plan


class OperationError(ValueError):
    pass


def _artifact_target(storage_dir: Path, directory: str, artifact_id: str) -> Path:
    if (
        not artifact_id
        or artifact_id in {".", ".."}
        or Path(artifact_id).name != artifact_id
        or "/" in artifact_id
        or "\\" in artifact_id
    ):
        raise OperationError(f"Invalid artifact ID: {artifact_id}")
    if storage_dir.is_symlink():
        raise OperationError("Application storage directory must not be a symlink.")
    target_dir = storage_dir / directory
    if target_dir.exists() and target_dir.is_symlink():
        raise OperationError(f"Application {directory} directory must not be a symlink.")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{artifact_id}.json"
    if target.is_symlink():
        raise OperationError(f"Application artifact must not be a symlink: {target}")
    return target


def _atomic_write(target: Path, content: str) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_operation_ids(plan: Plan, selectors: Sequence[str] | None = None) -> set[str]:
    """Resolve full IDs or unique displayed prefixes to ready operations."""

    ready = {operation.operation_id: operation for operation in plan.operations if operation.status == "ready"}
    if not selectors:
        return set(ready)
    selected: set[str] = set()
    for selector in selectors:
        matches = [operation_id for operation_id in ready if operation_id == selector or operation_id.startswith(selector)]
        if not matches:
            raise OperationError(f"Unknown or non-ready operation: {selector}")
        if len(matches) > 1:
            raise OperationError(f"Ambiguous operation prefix: {selector}")
        selected.add(matches[0])
    return selected


def save_plan(plan: Plan, storage_dir: Path) -> Path:
    target = _artifact_target(storage_dir, "plans", plan.plan_id)
    _atomic_write(target, plan.model_dump_json(indent=2))
    return target


def load_plan(plan_id: str, storage_dir: Path) -> Plan:
    target = _artifact_target(storage_dir, "plans", plan_id)
    if not target.is_file():
        raise OperationError(f"Plan not found: {plan_id}")
    plan = Plan.model_validate_json(target.read_text(encoding="utf-8"))
    if plan.plan_id != plan_id:
        raise OperationError(f"Plan ID does not match its artifact: {plan_id}")
    return plan


def _record_move(record: MoveRecord, storage_dir: Path) -> None:
    target = _artifact_target(storage_dir, "moves", record.move_id)
    _atomic_write(target, record.model_dump_json(indent=2))


def load_move(move_id: str, storage_dir: Path) -> MoveRecord:
    target = _artifact_target(storage_dir, "moves", move_id)
    if not target.is_file():
        raise OperationError(f"Move not found: {move_id}")
    move = MoveRecord.model_validate_json(target.read_text(encoding="utf-8"))
    if move.move_id != move_id:
        raise OperationError(f"Move ID does not match its artifact: {move_id}")
    return move


def apply_plan(plan: Plan, storage_dir: Path, operation_ids: set[str], approved: bool) -> list[dict[str, str]]:
    if not approved:
        raise OperationError("Apply requires explicit approval.")
    root = Path(plan.root_path).resolve()
    ready_ids = {operation.operation_id for operation in plan.operations if operation.status == "ready"}
    invalid_ids = operation_ids - ready_ids
    if invalid_ids:
        raise OperationError(f"Unknown or non-ready operation(s): {', '.join(sorted(invalid_ids))}")
    results = []

    for operation in plan.operations:
        if operation.operation_id not in operation_ids:
            continue
        if operation.status != "ready":
            results.append({"operation_id": operation.operation_id, "status": operation.status})
            continue

        source = ensure_within(root, root / operation.source_relative, "source")
        destination = safe_destination(root, operation.destination_relative)
        if not source.is_file() or source.is_symlink():
            raise OperationError(f"Source is unavailable or unsafe: {operation.source_relative}")
        if content_hash(source) != operation.content_hash:
            raise OperationError(f"Source changed after preview: {operation.source_relative}")
        if destination.exists() or destination.is_symlink():
            results.append({"operation_id": operation.operation_id, "status": "conflict"})
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            source.unlink()
        except FileExistsError:
            results.append({"operation_id": operation.operation_id, "status": "conflict"})
            continue

        move_id = f"move-{uuid.uuid4()}"
        record = MoveRecord(
            move_id=move_id,
            operation_id=operation.operation_id,
            root_path=str(root),
            source_relative=operation.source_relative,
            destination_relative=operation.destination_relative,
            content_hash=operation.content_hash,
        )
        try:
            _record_move(record, storage_dir)
        except Exception as error:
            try:
                os.link(destination, source)
                destination.unlink()
            except Exception as rollback_error:
                raise OperationError(
                    "Move record could not be saved and the move could not be rolled back."
                ) from rollback_error
            raise OperationError("Move record could not be saved; the move was rolled back.") from error
        results.append({"operation_id": operation.operation_id, "move_id": move_id, "status": "moved"})

    return results


def undo_move(move: MoveRecord, storage_dir: Path) -> None:
    root = Path(move.root_path).resolve()
    source = ensure_within(root, root / move.source_relative, "restore source")
    destination = ensure_within(root, root / move.destination_relative, "restore destination")
    if not destination.is_file() or destination.is_symlink():
        raise OperationError("Moved destination is unavailable or unsafe.")
    if source.exists():
        raise OperationError("Original source path is occupied; refusing to overwrite it.")
    if content_hash(destination) != move.content_hash:
        raise OperationError("Moved file changed after apply; refusing to undo it.")
    source.parent.mkdir(parents=True, exist_ok=True)
    os.link(destination, source)
    destination.unlink()
    move.undone_at = move.undone_at or datetime.now(timezone.utc)
    _record_move(move, storage_dir)
