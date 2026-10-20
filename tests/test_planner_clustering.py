import math

import pytest

from smartfileman import planner
from smartfileman.schemas import ContentUnit, ScanResult, ScannedFile


def test_complete_linkage_uses_sklearn_with_stable_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    real_clustering = planner.AgglomerativeClustering
    calls: list[dict[str, object]] = []

    class RecordingClustering:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)
            self.delegate = real_clustering(**kwargs)

        def fit_predict(self, vectors: list[list[float]]) -> object:
            return self.delegate.fit_predict(vectors)

    monkeypatch.setattr(planner, "AgglomerativeClustering", RecordingClustering)

    groups = planner._complete_linkage_clusters(
        [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]],
        0.18,
    )

    assert groups == [[0, 1], [2]]
    assert calls == [
        {
            "n_clusters": None,
            "distance_threshold": 0.18,
            "metric": "cosine",
            "linkage": "complete",
            "compute_full_tree": True,
        }
    ]


def test_complete_linkage_falls_back_to_legacy_affinity_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class LegacyClustering:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)
            if "metric" in kwargs:
                raise TypeError("metric is not supported")

        def fit_predict(self, vectors: list[list[float]]) -> list[int]:
            return [0, 0, 1]

    monkeypatch.setattr(planner, "AgglomerativeClustering", LegacyClustering)

    assert planner._complete_linkage_clusters(
        [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]],
        0.18,
    ) == [[0, 1], [2]]
    assert [call.get("metric", call.get("affinity")) for call in calls] == ["cosine", "cosine"]
    assert "affinity" not in calls[0]
    assert calls[1]["affinity"] == "cosine"


@pytest.mark.parametrize(
    ("vectors", "threshold", "expected"),
    [
        ([], 0.18, []),
        ([[1.0, 0.0]], 0.18, [[0]]),
        ([[0.0, 0.0], [1.0, 0.0]], 0.18, [[0], [1]]),
        ([[1.0, 0.0], [0.8, 0.6]], 0.2, [[0, 1]]),
        ([[1.0, 0.0], [0.8, 0.6]], 0.199, [[0], [1]]),
    ],
)
def test_complete_linkage_preserves_edge_and_threshold_behavior(
    vectors: list[list[float]],
    threshold: float,
    expected: list[list[int]],
) -> None:
    assert planner._complete_linkage_clusters(vectors, threshold) == expected


def test_complete_linkage_does_not_chain_through_a_bridge_vector() -> None:
    bridge = [0.95, math.sqrt(1 - 0.95**2)]

    vectors = [[1.0, 0.0], bridge, [0.8, 0.6]]

    assert planner._complete_linkage_clusters(vectors, 0.18) == [[0, 1], [2]]


def test_isolated_semantic_file_is_not_presented_as_high_confidence() -> None:
    file = ScannedFile(
        file_id="isolated",
        root_path="/files",
        relative_path="resume.pdf",
        filename="resume.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        content_hash="sha256:isolated",
        size_bytes=1,
        modified_time_ns=1,
        units=[
            ContentUnit(
                unit_type="page",
                ordinal=0,
                content_hash="sha256:unit",
                text="software engineer resume",
            )
        ],
    )

    labels = planner._semantic_labels([file], {"isolated": [1.0, 0.0]})

    assert labels["isolated"][0] == "Engineer - Resume - Software"
    assert labels["isolated"][1] == 0.0
    assert "no peer support" in labels["isolated"][2]


def test_mixed_document_identity_cluster_keeps_label_when_topics_are_shared() -> None:
    def file(file_id: str, opening: str, body: str) -> ScannedFile:
        return ScannedFile(
            file_id=file_id,
            root_path="/files",
            relative_path=f"{file_id}.txt",
            filename=f"{file_id}.txt",
            extension=".txt",
            mime_type="text/plain",
            content_hash=f"sha256:{file_id}",
            size_bytes=1,
            modified_time_ns=1,
            units=[
                ContentUnit(
                    unit_type="paragraph",
                    ordinal=0,
                    content_hash=f"sha256:{file_id}",
                    text=f"{opening}\n{body}",
                )
            ],
        )

    files = [
        file("resume-a", "Professional Summary\nAlex Morgan", "React Neo4j Python"),
        file("resume-b", "Professional Summary\nJordan Lee", "React Neo4j Go"),
        file("project-a", "Architecture Decision Record\nGraph ingestion service", "React Neo4j pipeline"),
        file("project-b", "Implementation Notes\nGraph pipeline deployment", "React Neo4j pipeline"),
    ]
    plan = planner.build_plan(
        ScanResult(root_path="/files", files=files),
        mode="semantic",
        file_vectors={file.file_id: [1.0, 0.0] for file in files},
    )

    assert all(operation.status == "ready" for operation in plan.operations)
    assert all(operation.subject != "Needs review" for operation in plan.operations)
    assert all("mixed document-identity evidence" in operation.reason for operation in plan.operations)
    assert all("insufficient shared topic evidence" not in operation.reason for operation in plan.operations)


def test_media_title_cluster_is_not_rejected_as_mixed_document_identity() -> None:
    def file(file_id: str, name: str) -> ScannedFile:
        return ScannedFile(
            file_id=file_id,
            root_path="/files",
            relative_path=name,
            filename=name,
            extension=".mkv",
            mime_type="video/x-matroska",
            content_hash=f"sha256:{file_id}",
            size_bytes=1,
            modified_time_ns=1,
            units=[
                ContentUnit(
                    unit_type="media",
                    ordinal=0,
                    content_hash=f"sha256:{file_id}",
                    text=name,
                )
            ],
        )

    files = [
        file("episode-14", "ReZERO - S02E14 - Straight Bet.mkv"),
        file("episode-15", "ReZERO - S02E15 - Otto Suwen.mkv"),
    ]
    labels = planner._semantic_labels(
        files,
        {file.file_id: [1.0, 0.0] for file in files},
    )

    assert labels["episode-14"][0] == "Rezero - Tv"
    assert labels["episode-15"][0] == "Rezero - Tv"
    assert "mixed document-identity evidence" not in labels["episode-14"][2]


def test_cluster_namer_cannot_reuse_unsupported_source_label() -> None:
    file = ScannedFile(
        file_id="resume",
        root_path="/files",
        relative_path="DineshThumma.pdf",
        filename="DineshThumma.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        content_hash="sha256:resume",
        size_bytes=1,
        modified_time_ns=1,
        units=[
            ContentUnit(
                unit_type="page",
                ordinal=0,
                content_hash="sha256:resume-unit",
                text="Professional Summary\nBackend engineer with Java Spring Boot experience and a GitHub portfolio",
            )
        ],
    )

    class BadNamer:
        def name(self, evidence: str, fallback: str) -> str:
            return "2027 - Com - Github"

    labels = planner._semantic_labels([file], {"resume": [1.0, 0.0]}, namer=BadNamer())

    assert labels["resume"][0] != "2027 - Com - Github"
    assert "no shared evidence" in labels["resume"][2]


def test_cluster_namer_cannot_use_media_release_metadata_as_title() -> None:
    def file(file_id: str, filename: str) -> ScannedFile:
        return ScannedFile(
            file_id=file_id,
            root_path="/files",
            relative_path=f"Telegram Desktop/{filename}",
            filename=filename,
            extension=".mkv",
            mime_type="video/x-matroska",
            content_hash=f"sha256:{file_id}",
            size_bytes=1,
            modified_time_ns=1,
            units=[
                ContentUnit(
                    unit_type="media",
                    ordinal=0,
                    content_hash=f"sha256:{file_id}",
                    text=filename,
                )
            ],
        )

    files = [
        file("dc", "DC.2026.1080p.WEB-DL.Hindi.5.1.mkv"),
        file("love", "Love.Of.Your.Life.2026.1080p.WEB-DL.Hindi.5.1.mkv"),
    ]

    class BadNamer:
        def name(self, evidence: str, fallback: str) -> str:
            return "2026 - English - Hindi"

    labels = planner._semantic_labels(
        files,
        {"dc": [1.0, 0.0], "love": [1.0, 0.0]},
        namer=BadNamer(),
    )

    assert labels["dc"][0] == "Needs review"
    assert "insufficient shared media-title evidence" in labels["dc"][2]


def test_coherent_cluster_sends_identity_and_topic_evidence_to_namer() -> None:
    def file(file_id: str, person: str, skills: str) -> ScannedFile:
        return ScannedFile(
            file_id=file_id,
            root_path="/files",
            relative_path=f"{file_id}.txt",
            filename=f"{file_id}.txt",
            extension=".txt",
            mime_type="text/plain",
            content_hash=f"sha256:{file_id}",
            size_bytes=1,
            modified_time_ns=1,
            units=[
                ContentUnit(
                    unit_type="paragraph",
                    ordinal=0,
                    content_hash=f"sha256:{file_id}",
                    text=f"Professional Summary\n{person}\nSkills: {skills}",
                )
            ],
        )

    class RecordingNamer:
        evidence = ""

        def name(self, evidence: str, fallback: str) -> str:
            self.evidence = evidence
            return "Resume Materials"

    namer = RecordingNamer()
    files = [
        file("resume-a", "Alex Morgan", "React Neo4j Python"),
        file("resume-b", "Jordan Lee", "React Neo4j Go"),
    ]
    labels = planner._semantic_labels(
        files,
        {file.file_id: [1.0, 0.0] for file in files},
        namer=namer,
    )

    assert labels["resume-a"][0] == "Resume Materials"
    assert labels["resume-b"][0] == "Resume Materials"
    assert "Shared identity/title terms: professional, summary" in namer.evidence
    assert "Shared topic/tool terms:" in namer.evidence
    assert "neo4j" in namer.evidence
    assert "react" in namer.evidence
