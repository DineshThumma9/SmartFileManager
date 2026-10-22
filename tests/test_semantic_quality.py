"""Tests for the semantic-quality fixes: adaptive threshold, identity vectors,
purpose naming, and destination-collision detection."""

from pathlib import Path

import pytest

from smartfileman import planner
from smartfileman.naming import (
    OllamaClusterNamer,
    _name_is_supported,
    _parse_name,
    representative_evidence,
    shared_filename_terms,
)
from smartfileman.operations import OperationError, apply_plan
from smartfileman.schemas import ContentUnit, Plan, PlanOperation, ScanResult, ScannedFile


def _file(
    file_id: str,
    relative_path: str,
    text: str,
    extension: str = ".txt",
) -> ScannedFile:
    name = Path(relative_path).name
    return ScannedFile(
        file_id=file_id,
        root_path="/files",
        relative_path=relative_path,
        filename=name,
        extension=extension,
        mime_type="text/plain",
        content_hash=f"sha256:{file_id}",
        size_bytes=1,
        modified_time_ns=1,
        units=[
            ContentUnit(
                unit_type="paragraph",
                ordinal=0,
                content_hash=f"sha256:{file_id}-unit",
                text=text,
            )
        ],
    )


def test_adaptive_threshold_separates_documents_that_a_fixed_threshold_merges() -> None:
    """A fixed absolute threshold cannot separate these vectors.

    Every vector is a small perturbation of the same direction, which is what
    documents from a single embedding model look like. The threshold derived
    from the corpus distribution separates them; the old fixed value does not.
    """

    vectors = [
        [1.0, 0.0, 0.0],
        [0.99, 0.10, 0.0],
        [0.98, 0.0, 0.15],
        [0.0, 1.0, 0.0],
        [0.0, 0.99, 0.10],
        [0.0, 0.98, 0.15],
    ]

    threshold = planner._adaptive_distance_threshold(vectors)

    assert planner.SEMANTIC_MIN_DISTANCE <= threshold <= planner.SEMANTIC_MAX_DISTANCE
    groups = planner._complete_linkage_clusters(vectors, threshold)
    assert len(groups) == 2
    assert sorted(len(group) for group in groups) == [3, 3]


def test_adaptive_threshold_falls_back_for_a_tiny_corpus() -> None:
    assert (
        planner._adaptive_distance_threshold([[1.0, 0.0], [0.0, 1.0]])
        == planner.SEMANTIC_DISTANCE_THRESHOLD
    )
    assert planner._adaptive_distance_threshold([]) == planner.SEMANTIC_DISTANCE_THRESHOLD


def test_identity_vector_is_blended_more_strongly_than_the_body_vector() -> None:
    body = [1.0, 0.0]
    identity = [0.0, 1.0]

    combined = planner._combine_vectors(body, identity)

    assert combined is not None
    assert combined[1] > combined[0]
    assert combined[0] > 0.0


def test_combine_vectors_falls_back_when_only_one_signal_exists() -> None:
    body = [1.0, 0.0]
    assert planner._combine_vectors(body, None) == body
    identity = [0.0, 1.0]
    assert planner._combine_vectors(None, identity) == identity


def test_identity_evidence_separates_a_resume_from_a_project_document() -> None:
    """A resume and a project document that share a toolset must not be
    grouped together when their identities differ."""

    resume = _file("resume", "resume.txt", "PROFESSIONAL SUMMARY\nSenior Backend Engineer\nJava Spring Boot Redis Kafka")
    project = _file("project", "adr.txt", "ARCHITECTURE DECISION RECORD\nGraph ingestion service\nJava Spring Boot Redis Kafka")
    vectors = {"resume": [1.0, 0.0], "project": [0.0, 1.0]}

    labels = planner._semantic_labels(
        [resume, project],
        vectors,
        identity_vectors={"resume": [1.0, 0.0], "project": [0.0, 1.0]},
    )

    assert labels["resume"][0] != labels["project"][0]


def test_shared_filename_terms_require_more_than_one_file() -> None:
    files = [_file("a", "one/resume.txt", "x"), _file("b", "two/resume.txt", "x")]

    assert shared_filename_terms(files) == ["resume"]
    assert shared_filename_terms([files[0]]) == ["resume"]


def test_shared_filename_terms_ignore_a_unique_name() -> None:
    files = [_file("a", "one/alpha.txt", "x"), _file("b", "two/beta.txt", "x")]

    assert shared_filename_terms(files) == []


def test_a_purpose_label_is_not_rejected_for_being_a_purpose() -> None:
    """A label such as "Resume" is correct even though the shared topic terms
    only list tools."""

    evidence = (
        "Shared topic/tool terms: boot, engineer, redis, spring\n"
        "Shared identity/title terms: professional, summary\n"
        "Shared filename terms: resume\n"
    )

    assert _name_is_supported("Resume", evidence) is True
    assert _name_is_supported("Professional Summary", evidence) is True
    assert _name_is_supported("Database Migration", evidence) is False


def test_evidence_includes_shared_filename_terms() -> None:
    evidence = representative_evidence(
        [_file("a", "one/resume.txt", "PROFESSIONAL SUMMARY")],
        ["react"],
        ["summary"],
        ["resume"],
    )

    assert "Shared filename terms: resume" in evidence


def test_naming_parser_accepts_a_fenced_json_response() -> None:
    assert _parse_name('```json\n{"name": "Resume"}\n```') == "Resume"


def test_naming_parser_rejects_a_response_without_a_name() -> None:
    with pytest.raises(ValueError):
        _parse_name("not json")
    with pytest.raises(ValueError):
        _parse_name("[]")
    with pytest.raises(ValueError):
        _parse_name('{"other": "value"}')


def test_naming_failure_is_logged_and_falls_back(tmp_path: Path, caplog) -> None:
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test", cache_path=tmp_path / "cache.json")

    class BrokenClient:
        def chat(self, **kwargs):
            raise TimeoutError("model timed out")

    namer.client = BrokenClient()

    with caplog.at_level("WARNING"):
        assert namer.name("Shared topic/tool terms: resume", "Resume Fallback") == "Resume Fallback"

    assert any("naming failed" in record.message.lower() for record in caplog.records)


def test_a_failed_naming_call_is_not_cached(tmp_path: Path) -> None:
    """A transient failure must not be persisted as a successful name."""

    cache_path = tmp_path / "cache.json"
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test", cache_path=cache_path)

    class BrokenClient:
        def chat(self, **kwargs):
            raise TimeoutError("model timed out")

    namer.client = BrokenClient()
    namer.name("Shared topic/tool terms: resume", "Resume Fallback")

    assert not cache_path.exists()


def test_two_files_with_the_same_name_get_distinct_destinations() -> None:
    """Both files must be movable, so they cannot share a destination."""

    files = [
        _file("a", "one/notes.txt", "alpha beta gamma"),
        _file("b", "two/notes.txt", "alpha beta gamma"),
    ]

    plan = planner.build_plan(ScanResult(root_path="/files", files=files), mode="fuzzy")

    destinations = [operation.destination_relative for operation in plan.operations]
    assert len(destinations) == len(set(destinations)), "the two files must not share a destination"
    assert all(operation.status == "ready" for operation in plan.operations)


def test_three_files_with_the_same_name_and_folder_still_get_distinct_destinations() -> None:
    files = [
        _file("a", "one/notes.txt", "alpha beta gamma"),
        _file("b", "one/notes.txt", "alpha beta gamma"),
        _file("c", "one/notes.txt", "alpha beta gamma"),
    ]

    plan = planner.build_plan(ScanResult(root_path="/files", files=files), mode="fuzzy")

    destinations = [operation.destination_relative for operation in plan.operations]
    assert len(destinations) == len(set(destinations))


def test_apply_refuses_a_batch_whose_second_source_changed(tmp_path: Path) -> None:
    """No file may be moved when a later file in the same batch is stale."""

    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    from smartfileman.scanner import content_hash

    operations = [
        PlanOperation(
            operation_id="op-one",
            file_id="one",
            source_relative="one.txt",
            destination_relative="_organized/S/Documents/one.txt",
            content_hash=content_hash(first),
            category="Documents",
            subject="S",
            confidence=0.9,
            reason="t",
            status="ready",
        ),
        PlanOperation(
            operation_id="op-two",
            file_id="two",
            source_relative="two.txt",
            destination_relative="_organized/S/Documents/two.txt",
            content_hash="sha256:" + ("0" * 64),
            category="Documents",
            subject="S",
            confidence=0.9,
            reason="t",
            status="ready",
        ),
    ]
    plan = Plan(root_path=str(tmp_path), mode="fuzzy", plan_id="test", operations=operations)

    with pytest.raises(OperationError, match="Source changed"):
        apply_plan(plan, tmp_path / ".smartfile", {"op-one", "op-two"}, approved=True)

    assert first.exists(), "the first file must not be moved when the batch fails"
    assert not (tmp_path / "_organized" / "S" / "Documents" / "one.txt").exists()


def test_non_latin_and_diacritic_names_still_contribute_evidence() -> None:
    assert "übersicht" in planner._all_evidence_tokens("projekt-Übersicht.pdf")
    assert planner._all_evidence_tokens("简历.pdf")


def test_a_group_that_does_not_stand_out_from_the_corpus_is_not_confident() -> None:
    """Confidence must measure separation, not raw similarity.

    Every vector here is nearly the same, so the group is not really more
    alike than the rest of the corpus. A confidence based on the raw cosine
    similarity would call this group highly confident.
    """

    vectors = [
        [1.0, 0.0],
        [0.999, 0.044],
        [0.998, 0.063],
        [0.999, 0.045],
    ]
    files = [
        _file(f"f{index}", f"file{index}.txt", f"document number {index}")
        for index in range(len(vectors))
    ]
    file_vectors = {file.file_id: vector for file, vector in zip(files, vectors, strict=True)}

    labels = planner._semantic_labels(files, file_vectors)

    assert all(confidence < planner.SEMANTIC_REVIEW_THRESHOLD for _, confidence, _ in labels.values())


def test_a_well_separated_group_is_confident() -> None:
    """A group that is clearly apart from the corpus earns a high confidence."""

    vectors = [
        [1.0, 0.0],
        [0.999, 0.044],
        [0.0, 1.0],
        [0.044, 0.999],
    ]
    files = [
        _file(f"f{index}", f"file{index}.txt", f"document number {index}")
        for index in range(len(vectors))
    ]
    file_vectors = {file.file_id: vector for file, vector in zip(files, vectors, strict=True)}

    labels = planner._semantic_labels(files, file_vectors)

    assert labels["f0"][1] >= planner.SEMANTIC_REVIEW_THRESHOLD
    assert labels["f2"][1] >= planner.SEMANTIC_REVIEW_THRESHOLD


def test_a_lone_file_is_never_confident() -> None:
    """One file has no peers, so nothing confirms the grouping."""

    file = _file("only", "only.txt", "some content")
    labels = planner._semantic_labels([file], {"only": [1.0, 0.0]})

    assert labels["only"][1] < planner.SEMANTIC_REVIEW_THRESHOLD
    assert "no other file is similar" in labels["only"][2]


def test_separation_confidence_is_bounded() -> None:
    assert planner._separation_confidence(0.9, 0.9, 0.2) == 0.0
    assert 0.0 < planner._separation_confidence(0.95, 0.5, 0.2) <= 0.99
    # A corpus with no spread to compare against carries no information, so
    # the raw similarity is all that is left.
    assert planner._separation_confidence(1.0, 0.0, 0.0) == 0.99


def test_a_bare_number_cannot_name_a_folder() -> None:
    """The amount on an invoice is not what the invoice is."""

    file = _file("invoice", "invoice-1200.txt", "Invoice total amount 1200 due March")

    subject, _, _ = planner.fuzzy_subject(file)

    assert "1200" not in subject
    assert "Amount" not in subject


def test_a_document_type_beats_the_tools_it_mentions() -> None:
    """A resume is a resume, even when its body is a list of technologies."""

    file = _file(
        "resume",
        "resume.txt",
        "PROFESSIONAL SUMMARY\nSenior Backend Engineer\nJava Spring Boot Redis Kafka",
    )

    subject, _, _ = planner.fuzzy_subject(file)

    assert subject == "Resume"


def test_a_document_type_in_the_file_name_is_used() -> None:
    file = _file("inv", "invoice.txt", "Amount 1200 due March")

    subject, _, _ = planner.fuzzy_subject(file)

    assert subject == "Invoice"


def test_fuzzy_mode_groups_files_by_document_type() -> None:
    """Two files that say what they are belong in the same folder."""

    files = [
        _file("a", "one.txt", "Meeting Notes\n2024-01-05\nAgenda and action items"),
        _file("b", "two.txt", "Meeting Notes\n2024-02-07\nAgenda and action items"),
    ]

    plan = planner.build_plan(ScanResult(root_path="/files", files=files), mode="fuzzy")

    assert len({operation.subject for operation in plan.operations}) == 1
    assert plan.operations[0].subject == "Meeting - Notes"


def test_fuzzy_mode_uses_a_file_name_shared_by_several_files() -> None:
    """A name several files share is how they were meant to be grouped."""

    files = [
        _file("a", "grocery.txt", "Shopping list milk eggs bread coffee"),
        _file("b", "grocery-2.txt", "Shopping list milk eggs pasta tea"),
    ]

    plan = planner.build_plan(ScanResult(root_path="/files", files=files), mode="fuzzy")

    assert {operation.subject for operation in plan.operations} == {"Grocery"}


def test_a_placeholder_file_name_does_not_outrank_the_content() -> None:
    """A name that says nothing must not decide the label."""

    file = _file("a", "document1.txt", "Newton motion and force")

    subject, _, _ = planner.fuzzy_subject(file)

    assert subject == "Force - Motion - Newton"


def test_fuzzy_mode_prefers_a_term_that_is_rare_in_the_corpus() -> None:
    """A word every file shares describes none of them."""

    shared = ["alpha", "beta", "gamma", "delta", "epsilon"]
    files = [
        _file("a", "one.txt", " ".join(shared + ["zebra"])),
        _file("b", "two.txt", " ".join(shared + ["zebra"])),
        _file("c", "three.txt", " ".join(shared)),
    ]

    plan = planner.build_plan(ScanResult(root_path="/files", files=files), mode="fuzzy")

    subjects = {operation.file_id: operation.subject for operation in plan.operations}
    assert subjects["a"] == "Zebra"
    assert subjects["b"] == "Zebra"
    # The third file has nothing distinctive of its own.
    assert subjects["c"] == "Needs review"


def test_scanner_reads_each_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scan must not read every file twice.

    The extraction cache key is the file's digest, and the file used to be
    hashed a second time to confirm it had not changed during extraction.
    """

    import smartfileman.scanner as scanner

    (tmp_path / "one.txt").write_text("distinct content one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("distinct content two", encoding="utf-8")

    calls = 0
    original = scanner.content_hash

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(scanner, "content_hash", counted)
    scanner.scan(tmp_path, extraction_cache_path=tmp_path / "cache.json")

    assert calls == 2
