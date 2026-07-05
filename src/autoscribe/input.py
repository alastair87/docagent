from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autoscribe.errors import InputError
from autoscribe.models import Document, DocumentMetadata


def load_document(path: Path, metadata_path: Path | None = None) -> Document:
    if not path.exists():
        raise InputError(f"Input file does not exist: {path}", ["Provide an existing .txt, .md, or .json file."])

    text, embedded_metadata = _load_text(path)
    text = normalise_text(text)
    if not text:
        raise InputError("Input text is empty after normalisation.", ["Provide non-empty document text."])

    metadata_data = embedded_metadata
    if metadata_path:
        if not metadata_path.exists():
            raise InputError(f"Metadata file does not exist: {metadata_path}")
        metadata_data.update(json.loads(metadata_path.read_text(encoding="utf-8")))

    input_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Document(
        document_id=f"doc_{input_hash[:12]}",
        text=text,
        input_hash=input_hash,
        metadata=DocumentMetadata(**metadata_data),
    )


def normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _load_text(path: Path) -> tuple[str, dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8"), {}
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InputError("JSON input must be an object containing a text field.")
        text = payload.get("text")
        if not isinstance(text, str):
            raise InputError(
                "JSON input must contain a string field named 'text'.",
                ["Add a top-level text field or provide a .txt/.md file."],
            )
        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise InputError("JSON metadata field must be an object when provided.")
        return text, metadata
    raise InputError("Unsupported input file type.", ["Use .txt, .md, or .json input."])
