import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .paths import ensure_within, safe_destination
from .scanner import content_hash
from .schemas import MoveRecord, Plan


class OperationError(ValueError):
    pass


def save_plan(plan: Plan, storage_dir: Path) -> Path:
    plans_dir = storage_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    target = plans_dir / f"{plan.plan_id}.json"
    target.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_plan(plan_id: str, storage_dir: Path) -> Plan:
    target = storage_dir / "plans" / f"{plan_id}.json"
    if not target.is_file():
        raise OperationError(f"Plan not found: {plan_id}")
    return Plan.model_validate_json(target.read_text(encoding="utf-8"))


def _record_move(record: MoveRecord, storage_dir: Path) -> None:
    moves_dir = storage_dir / "moves"
    moves_dir.mkdir(parents=True, exist_ok=True)
    (moves_dir / f"{record.move_id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")


def load_move(move_id: str, storage_dir: Path) -> MoveRecord:
    target = storage_dir / "moves" / f"{move_id}.json"
    if not target.is_file():
        raise OperationError(f"Move not found: {move_id}")
    return MoveRecord.model_validate_json(target.read_text(encoding="utf-8"))


def apply_plan(plan: Plan, storage_dir: Path, operation_ids: set[str], approved: bool) -> list[dict[str, str]]:
    if not approved:
        raise OperationError("Apply requires explicit approval.")
    root = Path(plan.root_path).resolve()
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
        _record_move(record, storage_dir)
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
