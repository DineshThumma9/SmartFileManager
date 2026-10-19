import hashlib
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings.base import EmbeddingProvider
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

SUBJECT_RULES = {
    "Mathematics": {"math", "mathematics", "algebra", "calculus", "geometry", "statistics"},
    "Physics": {"physics", "mechanics", "motion", "force", "thermodynamics", "quantum"},
    "Biology": {"biology", "cell", "genetics", "anatomy", "ecology"},
    "Chemistry": {"chemistry", "chemical", "organic", "molecule", "reaction"},
    "Computer Science": {"computer", "coding", "programming", "software", "algorithm", "python", "javascript"},
    "History": {"history", "historical", "war", "empire", "ancient", "revolution"},
    "Business": {"business", "finance", "marketing", "sales", "accounting", "management"},
    "Anime": {"anime", "manga", "otaku", "animation"},
    "Cooking": {"cooking", "recipe", "kitchen", "food", "baking"},
    "Motorcycles": {"bike", "bikes", "motorcycle", "motorcycles", "motorbike", "racing"},
}


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def category_for(file: ScannedFile) -> str:
    return CATEGORY_BY_EXTENSION.get(file.extension, "Other")


def fuzzy_subject(file: ScannedFile) -> tuple[str, float, str]:
    tokens = _tokens(f"{file.relative_path} {' '.join(unit.text for unit in file.units)}")
    scores = [(subject, len(tokens & keywords), keywords) for subject, keywords in SUBJECT_RULES.items()]
    subject, score, _ = max(scores, key=lambda item: item[1])
    if score == 0:
        return "Needs review", 0.2, "No known subject terms were found."
    matched = sorted(tokens & SUBJECT_RULES[subject])
    return subject, min(0.55 + score * 0.12, 0.95), f"Matched subject terms: {', '.join(matched)}"


def _semantic_labels(files: Sequence[ScannedFile], provider: EmbeddingProvider) -> dict[str, tuple[str, float, str]]:
    texts = [f"{file.filename}\n" + "\n".join(unit.text for unit in file.units) for file in files]
    vectors = provider.embed(texts)
    if not vectors:
        return {}
    labels = DBSCAN(eps=0.28, min_samples=1, metric="cosine").fit_predict(vectors)
    groups: dict[int, list[ScannedFile]] = defaultdict(list)
    for file, label in zip(files, labels, strict=True):
        groups[int(label)].append(file)

    result = {}
    for label, group in groups.items():
        group_text = " ".join(f"{file.relative_path} {' '.join(unit.text for unit in file.units)}" for file in group)
        pseudo = group[0].model_copy(update={"relative_path": group_text})
        subject, _, _ = fuzzy_subject(pseudo)
        if subject == "Needs review":
            subject = f"Cluster {label + 1:02d}"
        centroid = vectors[labels.tolist().index(label)]
        similarities = cosine_similarity([centroid], vectors)[0]
        for index, file in enumerate(files):
            if labels[index] == label:
                score = float(max(0.2, min(0.99, similarities[index])))
                result[file.file_id] = (subject, score, f"Semantic cluster {label + 1:02d} with {len(group)} file(s).")
    return result


def build_plan(scan_result: ScanResult, mode: str = "fuzzy", provider: EmbeddingProvider | None = None) -> Plan:
    ready_files = scan_result.ready_files
    semantic_labels = _semantic_labels(ready_files, provider) if mode == "semantic" and provider else {}
    operations = []
    root = Path(scan_result.root_path)

    for file in ready_files:
        if mode == "semantic" and file.file_id in semantic_labels:
            subject, confidence, reason = semantic_labels[file.file_id]
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
        elif subject == "Needs review":
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

    plan_seed = "\n".join(f"{item.file_id}:{item.content_hash}" for item in operations)
    plan_id = f"plan-{hashlib.sha256(plan_seed.encode()).hexdigest()[:16]}"
    return Plan(plan_id=plan_id, root_path=str(root), mode=mode, operations=operations)
