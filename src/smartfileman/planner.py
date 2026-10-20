import hashlib
import math
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:  # pragma: no cover - the project dependency normally provides this
    AgglomerativeClustering = None

from .naming import ClusterNamer, DeterministicClusterNamer, representative_evidence
from .paths import safe_destination
from .readers import MEDIA_EXTENSIONS, normalize_media_filename
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
SHORT_EVIDENCE_TOKENS = {"ai", "ml", "nlp", "tv", "ui", "ux"}
SEMANTIC_DISTANCE_THRESHOLD = 0.18
SEMANTIC_REVIEW_THRESHOLD = 0.64
ProgressCallback = Callable[[str, float], None]


def category_for(file: ScannedFile) -> str:
    return CATEGORY_BY_EXTENSION.get(file.extension, "Other")


def _evidence_tokens(text: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if (len(token) > 2 or token in SHORT_EVIDENCE_TOKENS) and token not in STOP_WORDS
    ]
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]


def _all_evidence_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if (len(token) > 2 or token in SHORT_EVIDENCE_TOKENS) and token not in STOP_WORDS
    }


def _opening_text(file: ScannedFile) -> str:
    if file.extension in MEDIA_EXTENSIONS:
        return normalize_media_filename(Path(file.filename))
    for unit in file.units:
        lines = [line.strip() for line in unit.text.splitlines() if line.strip()]
        if lines:
            return " ".join(lines[:2])[:240]
    return ""


def _identity_tokens(file: ScannedFile) -> set[str]:
    return _all_evidence_tokens(_opening_text(file))


def _body_text(file: ScannedFile) -> str:
    text = _full_text(file)
    opening = " ".join(_opening_text(file).split())
    if opening and text.startswith(opening):
        return text[len(opening):].strip()
    return text


def _full_text(file: ScannedFile) -> str:
    return " ".join(" ".join(unit.text.split()) for unit in file.units).strip()


def _file_topic_terms(file: ScannedFile) -> list[str]:
    if file.extension in MEDIA_EXTENSIONS:
        return _evidence_tokens(normalize_media_filename(Path(file.filename)))
    body = _body_text(file)
    # Short files often have only a title/opening. Do not discard their only
    # useful evidence just because the identity/body split consumed it.
    if len(_all_evidence_tokens(body)) < 2:
        body = _full_text(file)
    return _evidence_tokens(body)


def _file_evidence_tokens(file: ScannedFile) -> set[str]:
    text = normalize_media_filename(Path(file.filename)) if file.extension in MEDIA_EXTENSIONS else _body_text(file)
    if len(_all_evidence_tokens(text)) < 2:
        text = _full_text(file)
    return _all_evidence_tokens(text)


def _identity_is_mixed(files: Sequence[ScannedFile], identity_by_file: dict[str, set[str]]) -> bool:
    """Flag clusters whose opening/title evidence forms separate groups."""

    if len(files) < 2 or all(file.extension in MEDIA_EXTENSIONS for file in files):
        return False
    groups: list[list[set[str]]] = []
    for file in files:
        tokens = identity_by_file[file.file_id]
        if not tokens:
            return True
        for group in groups:
            if any(
                len(tokens & other) >= 2
                or len(tokens & other) / max(1, len(tokens | other)) >= 0.4
                for other in group
            ):
                group.append(tokens)
                break
        else:
            groups.append([tokens])
    return len(groups) > 1


def _label_from_text(text: str, fallback: str = "Needs review") -> str:
    evidence = _evidence_tokens(text)
    return " - ".join(token.title() for token in evidence) if evidence else fallback


def _label_tokens(value: str) -> set[str]:
    return _all_evidence_tokens(value)


def _source_directory_label(subject: str, files: Sequence[ScannedFile]) -> bool:
    normalized_subject = " ".join(subject.lower().split())
    subject_tokens = _label_tokens(subject)
    if not subject_tokens:
        return False
    for file in files:
        parts = Path(file.relative_path).parts[:-1]
        for part in parts:
            if part in {".smartfile", "_organized"}:
                continue
            normalized_part = " ".join(part.lower().split())
            if normalized_subject == normalized_part or subject_tokens == _label_tokens(part):
                return True
    return False


def _media_metadata_label(subject: str, files: Sequence[ScannedFile]) -> bool:
    if not files or not all(file.extension in MEDIA_EXTENSIONS for file in files):
        return False
    title_tokens = [
        _all_evidence_tokens(normalize_media_filename(Path(file.filename)))
        for file in files
    ]
    shared_title_tokens = set.intersection(*title_tokens) if title_tokens else set()
    meaningful_shared_tokens = {token for token in shared_title_tokens if not token.isdigit()}
    subject_tokens = _label_tokens(subject)
    return bool(subject_tokens) and (
        not meaningful_shared_tokens
        or any(token not in shared_title_tokens for token in subject_tokens)
    )


def _safe_cluster_subject(
    subject: str,
    fallback: str,
    files: Sequence[ScannedFile],
    topic_terms: Sequence[str],
    identity_terms: Sequence[str],
) -> tuple[str, str | None]:
    if _source_directory_label(subject, files):
        return fallback, "model label matched a source directory"
    if _media_metadata_label(subject, files):
        title_tokens = [
            _all_evidence_tokens(normalize_media_filename(Path(file.filename)))
            for file in files
        ]
        shared_title_tokens = set.intersection(*title_tokens) if title_tokens else set()
        if not {token for token in shared_title_tokens if not token.isdigit()}:
            return "Needs review", "insufficient shared media-title evidence"
        return fallback, "model label used media release metadata"
    supported_tokens = set(topic_terms) | set(identity_terms)
    supported_tokens.update(
        token
        for file in files
        for token in _all_evidence_tokens(Path(file.filename).stem)
    )
    subject_tokens = _label_tokens(subject)
    overlap = subject_tokens.intersection(supported_tokens)
    if subject_tokens and (not overlap or len(overlap) / len(subject_tokens) < 0.5):
        return fallback, "model label had no shared evidence"
    return subject, None


def fuzzy_subject(file: ScannedFile) -> tuple[str, float, str]:
    """Suggest a label from file evidence without a fixed subject taxonomy."""

    identity = (
        normalize_media_filename(Path(file.filename))
        if file.extension in MEDIA_EXTENSIONS
        else file.relative_path
    )
    evidence = _evidence_tokens(f"{identity} {' '.join(unit.text for unit in file.units)}")
    if not evidence:
        return "Needs review", 0.2, "No useful topic terms were found in the file evidence."
    confidence = min(0.45 + len(evidence) * 0.12, 0.8)
    return " - ".join(token.title() for token in evidence), confidence, f"Evidence terms: {', '.join(evidence)}"


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    return 1.0 - sum(a * b for a, b in zip(left, right, strict=True))


def _complete_linkage_clusters_fallback(vectors: list[list[float]], threshold: float) -> list[list[int]]:
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


def _complete_linkage_clusters(vectors: list[list[float]], threshold: float) -> list[list[int]]:
    """Cluster vectors with complete-linkage cosine distance and stable groups."""

    if not vectors:
        return []
    if len(vectors) == 1 or threshold < 0:
        return [[index] for index in range(len(vectors))]

    # sklearn deliberately rejects zero vectors for cosine distance. Keep the
    # existing implementation for that edge case so a malformed or empty
    # embedding cannot break an otherwise usable preview.
    if AgglomerativeClustering is None or any(not any(vector) for vector in vectors):
        return _complete_linkage_clusters_fallback(vectors, threshold)

    try:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="cosine",
                linkage="complete",
                compute_full_tree=True,
            )
        except TypeError:
            # Older sklearn releases called the distance argument ``affinity``.
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                affinity="cosine",
                linkage="complete",
                compute_full_tree=True,
            )
        labels = clustering.fit_predict(vectors)
    except (TypeError, ValueError):
        # Preserve the safe behavior if an installed sklearn version has an
        # incompatible clustering API or rejects this input representation.
        return _complete_linkage_clusters_fallback(vectors, threshold)

    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def _cluster_terms(
    files: Sequence[ScannedFile],
    all_files: Sequence[ScannedFile],
    evidence_by_file: dict[str, set[str]],
    shared_first: bool = False,
    minimum_support: float = 0.0,
) -> list[str]:
    cluster_counts: dict[str, int] = defaultdict(int)
    corpus_counts: dict[str, int] = defaultdict(int)
    for file in files:
        for token in evidence_by_file[file.file_id]:
            cluster_counts[token] += 1
    for file in all_files:
        for token in evidence_by_file[file.file_id]:
            corpus_counts[token] += 1
    document_count = max(1, len(all_files))
    ranked = sorted(
        cluster_counts,
        key=lambda token: (
            -(cluster_counts[token] / len(files))
            * (1.0 if shared_first else math.log((document_count + 1) / (corpus_counts[token] + 1))),
            token,
        ),
    )
    if minimum_support:
        required_support = max(2, math.ceil(len(files) * minimum_support))
        ranked = [token for token in ranked if cluster_counts[token] >= required_support]
    return ranked[:4]


def _semantic_labels(
    files: Sequence[ScannedFile],
    file_vectors: dict[str, list[float]],
    namer: ClusterNamer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, tuple[str, float, str]]:
    embedded_files = [file for file in files if file.file_id in file_vectors]
    if not embedded_files:
        return {}
    vectors = [file_vectors[file.file_id] for file in embedded_files]
    vectors_by_file_id = {file.file_id: vector for file, vector in zip(embedded_files, vectors, strict=True)}
    topic_evidence_by_file = {file.file_id: _file_evidence_tokens(file) for file in embedded_files}
    identity_evidence_by_file = {file.file_id: _identity_tokens(file) for file in embedded_files}
    groups = {
        label: [embedded_files[index] for index in indices]
        for label, indices in enumerate(_complete_linkage_clusters(vectors, SEMANTIC_DISTANCE_THRESHOLD))
    }
    if progress_callback:
        progress_callback("Grouping files and naming clusters", 0.0)

    result = {}
    total_groups = max(1, len(groups))
    for group_index, (label, group) in enumerate(groups.items(), start=1):
        if progress_callback:
            progress_callback(
                f"Naming cluster {group_index} of {len(groups)}",
                (group_index - 1) / total_groups,
            )
        if len(group) == 1:
            topic_terms = _file_topic_terms(group[0])
            identity_terms = _evidence_tokens(_opening_text(group[0]))
        else:
            topic_terms = _cluster_terms(
                group,
                embedded_files,
                topic_evidence_by_file,
                minimum_support=0.5,
            )
            identity_terms = _cluster_terms(
                group,
                embedded_files,
                identity_evidence_by_file,
                shared_first=True,
                minimum_support=0.5,
            )
        mixed_identity = _identity_is_mixed(group, identity_evidence_by_file)
        # A mixed-identity cluster must not be named after one document title;
        # its shared topic terms are the safer semantic fallback.
        fallback_terms = topic_terms if mixed_identity else (identity_terms or topic_terms)
        fallback = _label_from_text(" ".join(fallback_terms), fallback="Needs review")
        group_vectors = [vectors_by_file_id[file.file_id] for file in group]
        centroid = _normalize(
            [sum(vector[index] for vector in group_vectors) / len(group_vectors) for index in range(len(vectors[0]))]
        )
        representative_group = sorted(
            group,
            key=lambda file: (
                -sum(a * b for a, b in zip(vectors_by_file_id[file.file_id], centroid, strict=True)),
                file.relative_path,
            ),
        )
        evidence_text = representative_evidence(representative_group, topic_terms, identity_terms)
        subject = (namer or DeterministicClusterNamer()).name(evidence_text, fallback)
        subject, naming_warning = _safe_cluster_subject(
            subject,
            fallback,
            group,
            topic_terms,
            identity_terms,
        )
        if len(group_vectors) == 1:
            similarities = [0.0]
            support_reason = "no peer support; label inferred from file evidence"
        else:
            similarities = [
                sum(
                    sum(a * b for a, b in zip(vector, peer, strict=True))
                    for peer_index, peer in enumerate(group_vectors)
                    if peer_index != index
                )
                / (len(group_vectors) - 1)
                for index, vector in enumerate(group_vectors)
            ]
            support_reason = "peer-supported cluster"
        for file, confidence in zip(group, similarities, strict=True):
            bounded_confidence = float(max(0.0, min(0.99, confidence)))
            reason = (
                f"Cluster similarity evidence: {', '.join(topic_terms) if topic_terms else 'similar embeddings'}; "
                f"{support_reason}"
            )
            if mixed_identity:
                reason += "; mixed document-identity evidence"
                if len(topic_terms) < 2:
                    reason += "; insufficient shared topic evidence for automatic organization"
            if naming_warning:
                reason += f"; {naming_warning}"
            result[file.file_id] = (subject, bounded_confidence, reason)
        if progress_callback:
            progress_callback(
                f"Naming cluster {group_index} of {len(groups)}",
                group_index / total_groups,
            )
    return result


def build_plan(
    scan_result: ScanResult,
    mode: str = "fuzzy",
    file_vectors: dict[str, list[float]] | None = None,
    namer: ClusterNamer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Plan:
    ready_files = scan_result.ready_files
    semantic_labels = (
        _semantic_labels(ready_files, file_vectors or {}, namer, progress_callback)
        if mode == "semantic"
        else {}
    )
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
        elif (
            subject == "Needs review"
            or (mode == "semantic" and confidence < SEMANTIC_REVIEW_THRESHOLD)
            or "insufficient shared topic evidence" in reason
            or "insufficient shared media-title evidence" in reason
        ):
            status = "needs-review"
        operation_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"smartfile-operation:{root}:{mode}:{file.file_id}:{file.content_hash}:{destination_relative}",
            )
        )
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

    if progress_callback:
        progress_callback("Preparing preview plan", 1.0)

    plan_seed = "\n".join(
        f"{mode}:{item.file_id}:{item.content_hash}:{item.destination_relative}:{item.status}"
        for item in operations
    )
    plan_id = f"plan-{hashlib.sha256(plan_seed.encode()).hexdigest()[:16]}"
    return Plan(plan_id=plan_id, root_path=str(root), mode=mode, operations=operations)
