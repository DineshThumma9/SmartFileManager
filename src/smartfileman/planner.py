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

from .naming import (
    ClusterNamer,
    DeterministicClusterNamer,
    representative_evidence,
    shared_filename_terms,
)
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

GRAMMAR_STOP_WORDS = {
    "about", "after", "also", "and", "are", "been", "from", "for", "have",
    "into", "more", "not", "that", "the", "their", "this", "through", "with",
    "your",
}
# "notes", "document" and "file" are ordinary words in general prose, but
# they are also parts of real document-type names ("Meeting Notes", "Design
# Document", "Lecture Notes"). They are therefore only stopwords outside the
# identity tier, where the words come from a document's own opening.
STOP_WORDS = GRAMMAR_STOP_WORDS | {
    "document", "documents", "file", "files", "notes", "page", "slide",
}
SHORT_EVIDENCE_TOKENS = {"ai", "ml", "nlp", "tv", "ui", "ux"}
# A file extension is never a useful label, and a bare number is almost always
# an amount, a count, or an identifier rather than a description. Years are
# kept because they legitimately name a document ("2024 Tax Return").
FILE_EXTENSION_TOKENS = {
    "txt", "md", "markdown", "csv", "json", "py", "yaml", "yml", "pdf",
    "docx", "pptx", "jpg", "jpeg", "png", "gif", "webp", "bmp", "svg",
    "mp4", "mov", "mkv", "webm", "avi", "mp3", "wav", "m4a",
}
# Words that appear in the boilerplate of many kinds of document. They
# describe the form of a file, not its subject, so they are never used as a
# folder name.
FUZZY_FILLER_TOKENS = {
    "amount", "total", "date", "dates", "due", "subject", "reference",
    "address", "number", "value", "price", "cost", "sum", "subtotal",
    "summary", "details", "description", "content", "contents", "text",
    "here", "attached", "please", "kind", "regards", "sincerely", "hello",
}
YEAR_TOKEN_RE = re.compile(r"^(?:19|20)\d{2}$")
# Document types that are unambiguous. When a file's opening contains one of
# these, it is that kind of document no matter what else it mentions.
STRONG_DOCUMENT_TYPE_TOKENS = {
    "resume", "cv", "curriculum", "vitae", "invoice", "receipt", "payroll",
    "timesheet", "thesis", "dissertation", "diploma", "transcript",
    "contract", "agreement", "agenda", "minutes", "ledger", "journal",
    "syllabus", "memorandum", "memo", "cover", "lease", "deed", "permit",
    "patent", "manifest", "inventory", "budget", "estimate", "quotation",
}
# Words that name a kind of document rather than a subject. A document whose
# opening contains one of these is that kind of document, whatever else it
# mentions, so the label can be the document type instead of a list of the
# tools and topics inside it.
DOCUMENT_TYPE_TOKENS = {
    "resume", "cv", "curriculum", "vitae", "cover", "memo",
    "minutes", "agenda", "report", "invoice", "receipt",
    "statement", "contract", "agreement", "proposal", "plan", "budget",
    "estimate", "quotation", "journal", "ledger",
    "log", "timesheet", "payroll", "tax", "return", "transcript", "diploma",
    "certificate", "thesis", "dissertation", "paper", "article", "blog",
    "book", "chapter", "guide", "manual", "handbook", "policy",
    "procedure", "specification", "spec", "design", "requirements",
    "architecture", "decision", "record", "adr", "changelog", "readme",
    "tutorial", "workshop", "lecture", "assignment", "exam", "question",
    "presentation", "slides", "deck", "spreadsheet", "form", "checklist",
    "newsletter", "press", "release", "announcement", "roster",
    "schedule", "calendar", "application", "questionnaire",
    "feedback", "survey", "review", "analysis", "pitch",
    "meeting", "zettel", "note", "notes",
}
# "Meeting" and "Notes" are only a document type when they appear together.
# Document names that are made of words the type vocabulary does not hold
# on its own, or that need all three of their words.
DOCUMENT_TYPE_PHRASES = {
    ("one", "on", "one"),
    ("work", "sample"),
    ("read", "me"),
    ("to", "do"),
}
# A fixed absolute cosine-distance threshold cannot separate documents that
# were produced by the same embedding model: those vectors are strongly
# anisotropic, so every document lands in a narrow band of high similarity.
# The threshold is therefore derived from the observed distance distribution
# of the corpus being organized. These bounds keep a tiny or degenerate corpus
# from producing a useless number of groups.
SEMANTIC_DISTANCE_THRESHOLD = 0.18
SEMANTIC_DISTANCE_PERCENTILE = 0.35
SEMANTIC_MIN_DISTANCE = 0.04
SEMANTIC_MAX_DISTANCE = 0.45
# Weight of the document-identity vector against the body vector when both are
# available. Identity is the stronger signal for what a document *is*; the
# body still carries the topical signal.
SEMANTIC_IDENTITY_WEIGHT = 0.7
# A raw cosine similarity is not a confidence. Embedding models are strongly
# anisotropic, so every document from one model scores around 0.9 against
# every other document, and a fixed threshold on that number cannot tell a
# trustworthy grouping from a lucky one. The score below is therefore a
# *separation* score: how much more alike a file is to its own group than it
# is to the rest of the corpus.
SEMANTIC_REVIEW_THRESHOLD = 0.35
ProgressCallback = Callable[[str, float], None]


def category_for(file: ScannedFile) -> str:
    return CATEGORY_BY_EXTENSION.get(file.extension, "Other")


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens.

    Letters and digits from any script are kept so that a non-Latin filename
    still contributes identity evidence. An ASCII-only pattern would split
    "projekt-Übersicht" into the fragments "projekt" and "bersicht".
    """

    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def _evidence_tokens(text: str) -> list[str]:
    tokens = [
        token
        for token in _tokenize(text)
        if (len(token) > 2 or token in SHORT_EVIDENCE_TOKENS) and token not in STOP_WORDS
    ]
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]


def _all_evidence_tokens(text: str) -> set[str]:
    return {
        token
        for token in _tokenize(text)
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


def _descriptive_token(token: str) -> bool:
    """Reject tokens that cannot describe what a file is.

    A bare number is an amount, a count, or an identifier. A file extension is
    an artifact of how the file was saved, not what it contains. Neither can
    be the name of a folder a person would recognize.
    """

    if token in FILE_EXTENSION_TOKENS:
        return False
    if token.isdigit():
        return False
    if YEAR_TOKEN_RE.match(token):
        # A year can legitimately appear in a label, but only alongside
        # another descriptive term, which is handled by the caller.
        return False
    if token in FUZZY_FILLER_TOKENS:
        return False
    return True


def _ordered_tokens(text: str) -> list[str]:
    """Distinct tokens in the order they appear in the text."""

    seen: dict[str, None] = {}
    for token in _tokenize(text):
        if token not in seen:
            seen[token] = None
    return list(seen)


def _looks_like_heading(text: str) -> bool:
    """Whether the first line of a file reads as a title rather than prose.

    A heading names the document ("PROFESSIONAL SUMMARY", "Meeting Notes",
    "Architecture Decision Record"), and its word order carries the meaning.
    A sentence does not: "Newton motion and force" is a bag of three words
    with no order of importance.
    """

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        if len(words) > 8:
            return False
        letters = [character for character in stripped if character.isalpha()]
        if not letters:
            return False
        if all(character.isupper() for character in letters):
            return True
        # Title Case: every word starts with a capital, and at least one
        # word is longer than a single letter so that "A Dog" does not
        # qualify.
        if all(word[:1].isupper() for word in words) and any(len(word) > 1 for word in words):
            return True
        return False
    return False


def _fuzzy_identity_tokens(file: ScannedFile) -> list[str]:
    """Terms from the part of the file that says what it *is*.

    A document usually opens with what it is: "PROFESSIONAL SUMMARY",
    "Architecture Decision Record", "Meeting notes". Those opening lines are
    a much better folder name than a word that merely appears somewhere in
    the body, and they are what makes two unrelated-looking files land in
    the same folder.

    When the opening reads as a heading, the order of its words is kept, so
    the title comes first. When it reads as prose, the terms are ranked by
    how often they occur and the word order carries no meaning.
    """

    if file.extension in MEDIA_EXTENSIONS:
        source = normalize_media_filename(Path(file.filename))
    else:
        source = _opening_text(file)
    counts: dict[str, int] = defaultdict(int)
    order: dict[str, int] = {}
    for token in _tokenize(source):
        if token not in order:
            order[token] = len(order)
        counts[token] += 1
    # The identity tier keeps words that are only stopwords in general
    # prose, because "Meeting Notes" and "Design Document" are document
    # names, not filler.
    candidates = [
        token
        for token in order
        if (len(token) > 2 or token in SHORT_EVIDENCE_TOKENS)
        and token not in GRAMMAR_STOP_WORDS
        and _descriptive_token(token)
    ]
    if _looks_like_heading(source):
        return sorted(candidates, key=lambda token: order[token])
    return sorted(candidates, key=lambda token: (-counts[token], token))


def _document_type_label(file: ScannedFile) -> list[str]:
    """The kind of document this file is, if its name or opening says so.

    A file whose opening contains "resume" is a resume even when its body is
    a list of programming languages. Naming it after the tools would produce
    a folder called "Backend - Boot - Engineer" and scatter every resume in
    the collection into a different one.
    """

    if file.extension in MEDIA_EXTENSIONS:
        sources = [(normalize_media_filename(Path(file.filename)), True)]
    else:
        # The file name is checked first because a name is a deliberate
        # label, while an opening line is just the first thing the document
        # happens to say.
        sources = [
            (Path(file.filename).stem or file.relative_path, True),
            (_opening_text(file), False),
        ]
    for source, is_filename in sources:
        tokens = _ordered_tokens(source)
        if not tokens:
            continue
        # A fixed document name is checked first: "one on one" must not be
        # read as the two adjacent type words "one on".
        for phrase in DOCUMENT_TYPE_PHRASES:
            size = len(phrase)
            for start in range(len(tokens) - size + 1):
                if tuple(tokens[start : start + size]) == phrase:
                    return list(phrase)
        # Two adjacent type words name a document better than either one
        # alone: "Meeting Notes" and "Tax Return" are specific, while
        # "Meeting" and "Tax" are not.
        for start in range(len(tokens) - 1):
            left, right = tokens[start], tokens[start + 1]
            if left in DOCUMENT_TYPE_TOKENS and right in DOCUMENT_TYPE_TOKENS:
                return [left, right]
        # The first type word wins. "Professional Summary" is a section
        # heading inside a resume, so the earliest document type in the
        # opening is the document's own type.
        for token in tokens:
            if token in STRONG_DOCUMENT_TYPE_TOKENS:
                return [token]
        if not is_filename:
            for token in tokens:
                if token in DOCUMENT_TYPE_TOKENS:
                    return [token]
    return []


def _fuzzy_filename_tokens(file: ScannedFile) -> list[str]:
    """Terms from the file's own name."""

    stem = Path(file.filename).stem or file.relative_path
    return [
        token
        for token in _ordered_tokens(stem)
        if (len(token) > 2 or token in SHORT_EVIDENCE_TOKENS)
        and token not in STOP_WORDS
        and _descriptive_token(token)
    ]


def _fuzzy_body_tokens(file: ScannedFile) -> list[str]:
    """Terms from the body of the file, in evidence order."""

    text = " ".join(unit.text for unit in file.units)
    return [
        token
        for token in _evidence_tokens(text)
        if _descriptive_token(token)
    ]


# Words that appear in file names without describing what the file holds.
PLACEHOLDER_NAME_TOKENS = {
    "untitled", "document", "file", "doc", "new", "copy", "final", "draft",
    "output", "input", "data", "temp", "tmp", "misc", "unknown", "default",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "first", "second", "third", "page", "item", "entry", "record",
}


def _is_placeholder_name(file: ScannedFile) -> bool:
    """Whether a file's name is a placeholder rather than a description.

    "one.txt", "document2.pdf" and "new file.txt" are the names that files
    get when nobody named them on purpose. Such a name carries no
    information, and it must not outrank what the document itself says.
    """

    stem = Path(file.filename).stem or ""
    tokens = _tokenize(stem)
    if not tokens:
        return True
    # A trailing number is a sequence marker: "grocery-2.txt" is the second
    # grocery list, and the word before it still describes the file. Only a
    # name made *entirely* of a number is an identifier.
    if all(token.isdigit() for token in tokens):
        return True
    return any(token in PLACEHOLDER_NAME_TOKENS for token in tokens)


def fuzzy_subject(file: ScannedFile) -> tuple[str, float, str]:
    """Suggest a label from a single file's evidence.

    This is the per-file view. :func:`_fuzzy_subjects` is the corpus-aware
    version that planning actually uses; this remains for callers that only
    have one file to work with.
    """

    identity = _fuzzy_identity_tokens(file)
    filename = _fuzzy_filename_tokens(file)
    body = _fuzzy_body_tokens(file)
    document_type = _document_type_label(file)
    chosen = document_type or identity or filename or body
    if not chosen:
        return "Needs review", 0.2, "No useful topic terms were found in the file evidence."
    terms = chosen[:3]
    confidence = min(0.45 + len(terms) * 0.12, 0.8)
    label = " - ".join(token.title() for token in terms)
    return label, confidence, f"Evidence terms: {', '.join(terms)}"


def _fuzzy_subjects(
    files: Sequence[ScannedFile],
) -> dict[str, tuple[str, float, str]]:
    """Label every file in fuzzy mode, using the whole corpus for context.

    The label is built from the file's identity evidence first, because that
    is what makes two files land in the same folder. The body is only
    consulted when the file does not say what it is, and then the terms that
    are rare across the corpus are preferred, since a word that every file
    shares describes none of them.
    """

    if not files:
        return {}
    # A file name that several files share is a deliberate label: somebody
    # named "grocery.txt" and "grocery-2.txt" that way because they belong
    # together. A name that only one file has is weaker evidence than the
    # file's own content.
    filename_sharing: dict[str, int] = defaultdict(int)
    for file in files:
        if _is_placeholder_name(file):
            continue
        for token in set(_fuzzy_filename_tokens(file)):
            filename_sharing[token] += 1
    shared_filename_tokens = {
        token for token, count in filename_sharing.items() if count > 1
    }
    # The tier a file draws its label from is decided once, so that the
    # corpus counts and the label selection cannot disagree about which
    # words a file is described by.
    tiers: dict[str, tuple[str, list[str]]] = {}
    for file in files:
        document_type = _document_type_label(file)
        if document_type:
            tiers[file.file_id] = ("document type", document_type)
            continue
        identity = _fuzzy_identity_tokens(file)
        filename = _fuzzy_filename_tokens(file)
        body = _fuzzy_body_tokens(file)
        shared_name = [
            token for token in filename if token in shared_filename_tokens
        ]
        # A heading at the top of the document names it, so it comes first.
        # A name shared with other files comes next, because it is how the
        # person who named them decided they belong together. After that
        # come the document's own opening and its body.
        if identity and _looks_like_heading(_opening_text(file)):
            tiers[file.file_id] = ("opening heading", identity)
        elif shared_name:
            tiers[file.file_id] = ("shared filename", shared_name)
        elif identity:
            tiers[file.file_id] = ("opening evidence", identity)
        else:
            tiers[file.file_id] = ("body evidence", body)
    corpus_counts: dict[str, int] = defaultdict(int)
    for file in files:
        # A term is counted once per file, and only in the tier that file
        # actually uses for its label. Counting a term that only appears in
        # the body of every file, while every file is labelled from its
        # name, would make a rare name term look common.
        for token in set(tiers[file.file_id][1]):
            corpus_counts[token] += 1
    total = len(files)
    results: dict[str, tuple[str, float, str]] = {}
    for file in files:
        source, candidates = tiers[file.file_id]
        if source == "document type":
            # The file says what kind of document it is. That is the best
            # possible folder name, and it groups files that share nothing
            # else.
            terms = candidates
        else:
            # Within a tier, a term that only this file uses beats a term
            # every file shares, because a word common to the whole
            # collection describes none of its members.
            ranked = sorted(
                candidates,
                key=lambda token: (corpus_counts[token], candidates.index(token)),
            )
            if total == 1:
                # There is nothing to compare against, so every term is
                # equally distinctive.
                terms = ranked[:3]
            elif source == "shared filename":
                # A name several files share was chosen deliberately to
                # group them, so it is kept even though every file in the
                # group carries it.
                terms = ranked[:3]
            else:
                terms = [token for token in ranked if corpus_counts[token] < total][:3]
            if not terms:
                # Every candidate term is shared by the whole corpus. When
                # the files are also the same document, the shared terms
                # describe all of them equally well and they belong
                # together. When the files differ, the shared terms describe
                # none of them in particular, so the file is left for
                # review.
                if len({_full_text(item) for item in files}) == 1:
                    terms = ranked[:3]
                    source = "shared content"
                else:
                    results[file.file_id] = (
                        "Needs review",
                        0.2,
                        "Every term in this file is shared with the rest of the collection.",
                    )
                    continue
        if not terms:
            results[file.file_id] = (
                "Needs review",
                0.2,
                "No useful topic terms were found in the file evidence.",
            )
            continue
        label = " - ".join(token.title() for token in terms)
        if len(terms) < 2:
            confidence = 0.35
        else:
            confidence = min(0.45 + len(terms) * 0.12, 0.8)
        shared = ", ".join(
            f"{token} (in {corpus_counts[token]} of {total} files)"
            for token in terms
        )
        results[file.file_id] = (label, confidence, f"Evidence terms: {shared} [{source}]")
    return results


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


def _adaptive_distance_threshold(vectors: list[list[float]]) -> float:
    """Derive a clustering distance from the corpus's own distance spread.

    A hardcoded threshold is meaningless for a fixed embedding model because
    every document vector sits in a narrow band of high cosine similarity. The
    distance that actually separates groups is the one the corpus's own
    distribution points at, so the threshold is taken as a percentile of the
    observed pairwise distances and clamped to a sane range.
    """

    if len(vectors) < 3:
        return SEMANTIC_DISTANCE_THRESHOLD
    distances: list[float] = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            distances.append(_cosine_distance(left, right))
    if not distances:
        return SEMANTIC_DISTANCE_THRESHOLD
    distances.sort()
    position = min(int(len(distances) * SEMANTIC_DISTANCE_PERCENTILE), len(distances) - 1)
    return min(max(distances[position], SEMANTIC_MIN_DISTANCE), SEMANTIC_MAX_DISTANCE)


def _separation_confidence(
    member_similarity: float,
    outside_similarity: float,
    corpus_spread: float,
) -> float:
    """Score a grouping by how much it separates from the rest of the corpus.

    The interesting question is not "how similar are these two files" but "how
    much more similar are they than they are to everything else". A resume
    group scoring 0.96 internally is convincing when the other files in the
    corpus only reach 0.84, and it is not convincing at all when the corpus
    contains files that score 0.95 against everything else.

    The margin is measured against the corpus's own spread, so the score
    means the same thing whether the corpus is tightly or loosely
    clustered.
    """

    if corpus_spread <= 0.0:
        # Every file in the corpus is equally similar to every other, so
        # the embeddings carry no information about grouping. There is
        # nothing to calibrate against, and the raw similarity is the only
        # signal left.
        return float(max(0.0, min(0.99, member_similarity)))
    margin = member_similarity - outside_similarity
    if margin <= 0.0:
        # The group is no more alike than the rest of the corpus.
        return 0.0
    # The margin is expressed as a fraction of the corpus's own spread, so
    # the score means the same thing whether the corpus is tightly or
    # loosely clustered.
    return float(max(0.0, min(0.99, margin / corpus_spread)))


def _group_separation(
    group_indices: list[int],
    vectors: list[list[float]],
    corpus_median: float = 0.0,
) -> tuple[float, float]:
    """Return the mean internal similarity and the mean similarity outward.

    A group that contains the whole corpus has nothing outside itself to be
    compared against. Comparing it to the corpus's own median similarity
    keeps the measurement honest: such a group is not separated from the
    rest of the corpus, it *is* the rest of the corpus.
    """

    if not group_indices:
        return 0.0, 0.0
    internal: list[float] = []
    for position, left_index in enumerate(group_indices):
        for right_index in group_indices[position + 1 :]:
            internal.append(1.0 - _cosine_distance(vectors[left_index], vectors[right_index]))
    outside: list[float] = []
    members = set(group_indices)
    for left_index in group_indices:
        for right_index, right in enumerate(vectors):
            if right_index not in members:
                outside.append(1.0 - _cosine_distance(vectors[left_index], right))
    mean_internal = sum(internal) / len(internal) if internal else 0.0
    mean_outside = sum(outside) / len(outside) if outside else corpus_median
    return mean_internal, mean_outside


def _corpus_similarity_profile(vectors: list[list[float]]) -> tuple[float, float]:
    """Return the corpus's median similarity and how much it varies.

    The median is the similarity a random pair of files in this corpus
    reaches, and the spread is the distance between the low and high
    percentiles of the same distribution. Both describe the corpus rather
    than the model, so a corpus of near-identical documents correctly
    reports a small spread.
    """

    similarities: list[float] = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            similarities.append(1.0 - _cosine_distance(left, right))
    if len(similarities) < 2:
        return 0.0, 0.0
    similarities.sort()
    median = similarities[len(similarities) // 2]
    low = similarities[int(len(similarities) * 0.1)]
    high = similarities[int(len(similarities) * 0.9)]
    return median, max(0.0, high - low)


def _combine_vectors(
    body_vector: list[float] | None,
    identity_vector: list[float] | None,
) -> list[float] | None:
    """Blend the body and identity vectors into one comparable vector.

    Averaging every chunk of a long document blends its distinctive opening
    into the corpus mean, which is what makes a resume look like a project
    document when both mention the same tools. The identity vector, built
    from the opening alone, keeps that signal.
    """

    if body_vector is None:
        return identity_vector
    if identity_vector is None:
        return body_vector
    if len(body_vector) != len(identity_vector):
        return body_vector
    combined = [
        SEMANTIC_IDENTITY_WEIGHT * left + (1.0 - SEMANTIC_IDENTITY_WEIGHT) * right
        for left, right in zip(identity_vector, body_vector, strict=True)
    ]
    return _normalize(combined)


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
    identity_vectors: dict[str, list[float]] | None = None,
) -> dict[str, tuple[str, float, str]]:
    embedded_files = [file for file in files if file.file_id in file_vectors]
    if not embedded_files:
        return {}
    identity_vectors = identity_vectors or {}
    body_by_file_id = {file.file_id: file_vectors[file.file_id] for file in embedded_files}
    combined_by_file_id: dict[str, list[float]] = {}
    for file in embedded_files:
        combined = _combine_vectors(
            body_by_file_id[file.file_id],
            identity_vectors.get(file.file_id),
        )
        if combined is not None:
            combined_by_file_id[file.file_id] = combined
    embedded_files = [file for file in embedded_files if file.file_id in combined_by_file_id]
    if not embedded_files:
        return {}
    vectors = [combined_by_file_id[file.file_id] for file in embedded_files]
    vectors_by_file_id = combined_by_file_id
    topic_evidence_by_file = {file.file_id: _file_evidence_tokens(file) for file in embedded_files}
    identity_evidence_by_file = {file.file_id: _identity_tokens(file) for file in embedded_files}
    corpus_median, corpus_spread = _corpus_similarity_profile(vectors)
    threshold = _adaptive_distance_threshold(vectors)
    cluster_indices = _complete_linkage_clusters(vectors, threshold)
    groups = {
        label: [embedded_files[index] for index in indices]
        for label, indices in enumerate(cluster_indices)
    }
    group_indices = {label: indices for label, indices in enumerate(cluster_indices)}
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
        # The medoid is the member closest to all other members. It is a real
        # file, so the evidence shown to the namer comes from a document that
        # actually exists in the group rather than from a synthetic centroid.
        representative_group = sorted(
            group,
            key=lambda file: (
                -sum(
                    sum(a * b for a, b in zip(vectors_by_file_id[file.file_id], peer, strict=True))
                    for peer in group_vectors
                ),
                file.relative_path,
            ),
        )
        filename_terms = shared_filename_terms(group)
        evidence_text = representative_evidence(
            representative_group, topic_terms, identity_terms, filename_terms
        )
        subject = (namer or DeterministicClusterNamer()).name(evidence_text, fallback)
        subject, naming_warning = _safe_cluster_subject(
            subject,
            fallback,
            group,
            topic_terms,
            identity_terms,
        )
        if len(group_vectors) == 1:
            support_reason = "no peer support; label inferred from file evidence"
        else:
            support_reason = "peer-supported cluster"
        # The group's own separation from the rest of the corpus is what makes
        # the grouping trustworthy. A file's raw similarity to its peers is
        # only meaningful next to how similar it is to everything else.
        mean_internal, mean_outside = _group_separation(
            group_indices[label], vectors, corpus_median
        )
        group_confidence = _separation_confidence(mean_internal, mean_outside, corpus_spread)
        for file in group:
            bounded_confidence = float(max(0.0, min(0.99, group_confidence)))
            reason = (
                f"Cluster similarity evidence: {', '.join(topic_terms) if topic_terms else 'similar embeddings'}; "
                f"{support_reason}"
            )
            if len(group_vectors) == 1:
                reason += "; no other file is similar enough to confirm this grouping"
            else:
                reason += (
                    f"; group similarity {mean_internal:.2f} against {mean_outside:.2f} for the rest of the corpus"
                    f" (corpus spread {corpus_spread:.2f})"
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


def _destination_for(file: ScannedFile, subject: str, category: str) -> str:
    return (Path("_organized") / subject / category / file.filename).as_posix()


def _disambiguate_destinations(
    files: Sequence[ScannedFile],
    subjects: dict[str, str],
    categories: dict[str, str],
) -> dict[str, str]:
    """Give every file a distinct destination path.

    Two files with the same name in different folders can be assigned the same
    subject, and then the same destination. Only one of them could ever be
    moved, so the second file would silently stay behind. Each colliding file
    is given a destination that carries its own folder name, and a numeric
    suffix is used only if that is still not enough.
    """

    destinations: dict[str, str] = {}
    used: set[str] = set()
    for file in files:
        subject = subjects[file.file_id]
        candidate = _destination_for(file, subject, categories[file.file_id])
        if candidate in used:
            stem = Path(file.filename).stem or file.filename
            suffix = Path(file.filename).suffix
            parent = Path(file.relative_path).parent.name
            if parent and parent not in {".", "..", "_organized", ".smartfile"}:
                candidate = (
                    Path("_organized")
                    / subject
                    / categories[file.file_id]
                    / f"{stem} - {parent}{suffix}"
                ).as_posix()
            attempt = 2
            while candidate in used:
                candidate = (
                    Path("_organized")
                    / subject
                    / categories[file.file_id]
                    / f"{stem} ({attempt}){suffix}"
                ).as_posix()
                attempt += 1
        used.add(candidate)
        destinations[file.file_id] = candidate
    return destinations


def build_plan(
    scan_result: ScanResult,
    mode: str = "fuzzy",
    file_vectors: dict[str, list[float]] | None = None,
    namer: ClusterNamer | None = None,
    progress_callback: ProgressCallback | None = None,
    identity_vectors: dict[str, list[float]] | None = None,
) -> Plan:
    ready_files = scan_result.ready_files
    semantic_labels = (
        _semantic_labels(ready_files, file_vectors or {}, namer, progress_callback, identity_vectors)
        if mode == "semantic"
        else {}
    )
    operations = []
    root = Path(scan_result.root_path)

    subjects: dict[str, str] = {}
    categories: dict[str, str] = {}
    reasons: dict[str, tuple[str, float, str]] = {}
    fuzzy_labels = {} if mode == "semantic" else _fuzzy_subjects(ready_files)
    for file in ready_files:
        if mode == "semantic":
            subject, confidence, reason = semantic_labels.get(
                file.file_id,
                ("Needs review", 0.2, "No current cached embedding is available for this file."),
            )
        else:
            subject, confidence, reason = fuzzy_labels[file.file_id]
        subjects[file.file_id] = subject
        categories[file.file_id] = category_for(file)
        reasons[file.file_id] = (subject, confidence, reason)

    # Two files from different folders can share a name and land in the same
    # subject. Their destinations would then collide and only one of them
    # could ever be moved, so the destinations are made distinct up front.
    destination_by_file = _disambiguate_destinations(ready_files, subjects, categories)

    for file in ready_files:
        subject, confidence, reason = reasons[file.file_id]
        category = categories[file.file_id]
        destination_relative = destination_by_file[file.file_id]
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
