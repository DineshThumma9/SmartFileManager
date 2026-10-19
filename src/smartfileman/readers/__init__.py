import hashlib
import mimetypes
import re
from collections.abc import Callable
from pathlib import Path

from docx import Document
from pypdf import PdfReader
from pptx import Presentation

from ..schemas import ContentUnit

MAX_CHUNK_CHARS = 1400

def extension_for(path: Path) -> str:
    return path.suffix.lower()


def mime_type_for(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _unit(unit_type: str, ordinal: int, text: str, locator: dict[str, str | int] | None = None) -> ContentUnit:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return ContentUnit(
        unit_type=unit_type,
        ordinal=ordinal,
        content_hash=f"sha256:{digest}",
        text=normalized,
        locator=locator or {},
        extractor_version="2",
    )


def _text_units(text: str) -> list[ContentUnit]:
    paragraphs = [part.strip() for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n") if part.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    return [_unit("paragraph", index, paragraph) for index, paragraph in enumerate(paragraphs)]


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text without silently truncating it."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return [""]
    if len(normalized) <= max_chars:
        return [normalized]

    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", normalized) if piece.strip()]
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(piece), max_chars):
                chunks.append(piece[start : start + max_chars])
            continue
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def _chunk_units(units: list[ContentUnit], max_chars: int) -> list[ContentUnit]:
    chunked: list[ContentUnit] = []
    for unit in units:
        parts = _split_text(unit.text, max_chars)
        for chunk_index, part in enumerate(parts):
            locator = dict(unit.locator)
            locator["chunk_index"] = chunk_index
            locator["chunk_count"] = len(parts)
            chunked.append(
                _unit(unit.unit_type, len(chunked), part, locator)
            )
    return chunked


def _read_text(path: Path) -> list[ContentUnit]:
    return _text_units(path.read_text(encoding="utf-8", errors="replace"))


def _read_pdf(path: Path) -> list[ContentUnit]:
    return [
        _unit("page", index, page.extract_text() or "", {"page": index + 1})
        for index, page in enumerate(PdfReader(str(path)).pages)
    ]


def _read_docx(path: Path) -> list[ContentUnit]:
    return [
        _unit("paragraph", index, paragraph.text, {"paragraph": index + 1})
        for index, paragraph in enumerate(Document(str(path)).paragraphs)
        if paragraph.text.strip()
    ]


def _read_pptx(path: Path) -> list[ContentUnit]:
    units = []
    for index, slide in enumerate(Presentation(str(path)).slides):
        texts = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
        units.append(_unit("slide", index, "\n".join(texts), {"slide": index + 1}))
    return units


def _read_media(path: Path) -> list[ContentUnit]:
    return [_unit("media", 0, path.stem, {"extension": extension_for(path)})]


Reader = Callable[[Path], list[ContentUnit]]
READERS: dict[str, Reader] = {
    **{extension: _read_text for extension in {".txt", ".md", ".markdown", ".csv", ".json", ".py", ".yaml", ".yml"}},
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".pptx": _read_pptx,
    **{
        extension: _read_media
        for extension in {
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg",
            ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".wav", ".m4a",
        }
    },
}

SUPPORTED_EXTENSIONS = frozenset(READERS)


def extract_units(path: Path, max_chunk_chars: int = MAX_CHUNK_CHARS) -> list[ContentUnit]:
    """Select the reader from the extension registry and safely chunk output."""

    reader = READERS.get(extension_for(path))
    return _chunk_units(reader(path), max_chunk_chars) if reader else []
