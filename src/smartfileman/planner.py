import hashlib
import math
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .naming import ClusterNamer, DeterministicClusterNamer, representative_evidence
from .paths import safe_destination
from .schemas import Plan, PlanOperation, ScanResult, ScannedFile

CATEGORY_BY_EXTENSION = {
    ".jpg": "Images", ".jpeg": "Images", ".png": "Images", ".gif": "Images",
    ".webp": "Images", ".bmp": "Images", ".svg": "Images",
    ".mp4": "Videos", ".mov": "Videos", ".mkv": "Videos", ".webm": "Videos", ".avi": "Videos",
    ".mp3": "Audio", ".wav": "Audio", ".m4a": "Audio",
    ".pdf": "Documents", ".docx": "Documents", ".pptx": "Documents",
    ".txt": "Documents", ".md": "Documents", ".markdown": "Documents", ".csv": "Documents",
}

STOP_WORDS = {
    "about", "after", "also", "and", "are", "been", "from", "for", "have",
    "into", "more", "not", "that", "the", "their", "this", "through", "with",
    "your", "document", "documents", "file", "files", "notes", "page", "slide",
}
SEMANTIC_DISTANCE_THRESHOLD = 0.18
SEMANTIC_REVIEW_THRESHOLD = 0.64


def category_for(file: ScannedFile) -> str:
    return CATEGORY_BY_EXTENSION.get(file.extension, "Other")


def _evidence_tokens(text: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in STOP_WORDS]
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]


def _label_from_text(text: str, fallback: str = "Needs review") -> str:
    evidence = _evidence_tokens(text)
    return " - ".join(token.title() for token in evidence) if evidence else fallback


def fuzzy_subject(file: ScannedFile) -> tuple[str, float, str]:
    """Suggest a label from file evidence without a fixed subject taxonomy."""

    evidence = _evidence_tokens(f"{file.relative_path} {' '.join(unit.text for unit in file.units)}")
    if not evidence:
        return "Needs review", 0.2, "No useful topic terms were found in the file evidence."
    confidence = min(0.45 + len(evidence) * 0.12, 0.8)
    return " - ".join(token.title() for token in evidence), confidence, f"Evidence terms: {', '.join(evidence)}"


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - sum(a * b for a, b in zip(left, right, strict=True))


def _complete_linkage_clusters(vectors: list[list[float]], threshold: float) -> list[list[int]]:
    """Cluster without DBSCAN chaining unrelated files through bridge points."""

    clusters = [[index] for index in range(len(vectors))]
    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for left_index, left_cluster in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right_cluster = clusters[right_index]
                distance = max(
                    _cosine_distance(vectors[left], vectors[right])
                    for left in left_cluster
                    for right in right_cluster
                )
                candidate = (distance, left_index, right_index)
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] > threshold:
            break
        _, left_index, right_index = best
        clusters[left_index].extend(clusters[right_index])
        del clusters[right_index]
    return clusters


def _cluster_terms(files: Sequence[ScannedFile], all_files: Sequence[ScannedFile]) -> list[str]:
    cluster_counts: dict[str, int] = defaultdict(int)
    corpus_counts: dict[str, int] = defaultdict(int)
    for file in files:
        tokens = set(_evidence_tokens(f"{file.relative_path} {' '.join(unit.text for unit in file.units)}"))
        for token in tokens:
            cluster_counts[token] += 1
    for file in all_files:
        tokens = set(_evidence_tokens(f"{file.relative_path} {' '.join(unit.text for unit in file.units)}"))
        for token in tokens:
            corpus_counts[token] += 1
    document_count = max(1, len(all_files))
    ranked = sorted(
        cluster_counts,
        key=lambda token: (
            -(cluster_counts[token] / len(files)) * math.log((document_count + 1) / (corpus_counts[token] + 1)),
            token,
        ),
    )
    return ranked[:4]


def _semantic_labels(
    files: Sequence[ScannedFile],
    file_vectors: dict[str, list[float]],
    namer: ClusterNamer | None = None,
) -> dict[str, tuple[str, float, str]]:
    embedded_files = [file for file in files if file.file_id in file_vectors]
    if not embedded_files:
        return {}
    vectors = [file_vectors[file.file_id] for file in embedded_files]
    groups = {
        label: [embedded_files[index] for index in indices]
        for label, indices in enumerate(_complete_linkage_clusters(vectors, SEMANTIC_DISTANCE_THRESHOLD))
    }

    result = {}
    for label, group in groups.items():
        evidence = _cluster_terms(group, embedded_files)
        fallback = _label_from_text(" ".join(evidence), fallback=f"Cluster {label + 1:02d}")
        evidence_text = representative_evidence(group, evidence)
        subject = (namer or DeterministicClusterNamer()).name(evidence_text, fallback)
        group_indices = [embedded_files.index(file) for file in group]
        group_vectors = [vectors[index] for index in group_indices]
        centroid = _normalize(
            [sum(vector[index] for vector in group_vectors) / len(group_vectors) for index in range(len(vectors[0]))]
        )
        similarities = [sum(a * b for a, b in zip(vector, centroid, strict=True)) for vector in group_vectors]
        for file, confidence in zip(group, similarities, strict=True):
            bounded_confidence = float(max(0.2, min(0.99, confidence)))
            reason = f"Cluster evidence: {', '.join(evidence) if evidence else 'similar embeddings'}"
            result[file.file_id] = (subject, bounded_confidence, reason)
    return result


def build_plan(
    scan_result: ScanResult,
    mode: str = "fuzzy",
    file_vectors: dict[str, list[float]] | None = None,
    namer: ClusterNamer | None = None,
) -> Plan:
    ready_files = scan_result.ready_files
    semantic_labels = _semantic_labels(ready_files, file_vectors or {}, namer) if mode == "semantic" else {}
    operations = []
    root = Path(scan_result.root_path)

    for file in ready_files:
        if mode == "semantic":
            subject, confidence, reason = semantic_labels.get(
                file.file_id,
                ("Needs review", 0.2, "No current cached embedding is available for this file."),
            )
        else:
            subject, confidence, reason = fuzzy_subject(file)
        category = category_for(file)
        destination_relative = (Path("_organized") / subject / category / file.filename).as_posix()
        destination = safe_destination(root, destination_relative)
        source = root / file.relative_path
        status = "ready"
        if destination.exists() and destination.resolve() != source.resolve():
            status = "conflict"
            reason = f"Destination already exists: {destination_relative}"
        elif subject == "Needs review" or (mode == "semantic" and confidence < SEMANTIC_REVIEW_THRESHOLD):
            status = "needs-review"
        operation_id = str(uuid.uuid4())
        operations.append(
            PlanOperation(
                operation_id=operation_id,
                file_id=file.file_id,
                source_relative=file.relative_path,
                destination_relative=destination_relative,
                content_hash=file.content_hash,
                category=category,
                subject=subject,
                confidence=confidence,
                reason=reason,
                status=status,
            )
        )

    plan_seed = "\n".join(
        f"{mode}:{item.file_id}:{item.content_hash}:{item.destination_relative}:{item.status}"
        for item in operations
    )
    plan_id = f"plan-{hashlib.sha256(plan_seed.encode()).hexdigest()[:16]}"
    return Plan(plan_id=plan_id, root_path=str(root), mode=mode, operations=operations)
