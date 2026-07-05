from pathlib import Path

import pytest

from autoscribe.chunking import chunk_document
from autoscribe.errors import InputError
from autoscribe.input import load_document


def test_load_text_document(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("Hello\r\nworld\n", encoding="utf-8")

    document = load_document(path)

    assert document.text == "Hello\nworld"
    assert document.document_id.startswith("doc_")


def test_empty_input_fails(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(InputError):
        load_document(path)


def test_json_input_reads_text_and_metadata(tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_text('{"text": "Policy text", "metadata": {"title": "A title"}}', encoding="utf-8")

    document = load_document(path)

    assert document.text == "Policy text"
    assert document.metadata.title == "A title"


def test_chunk_ids_are_stable_and_ordered(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("Paragraph one.\n\n" + ("word " * 500), encoding="utf-8")
    document = load_document(path)

    chunks = chunk_document(document, chunk_size_tokens=80, overlap_tokens=10)

    assert len(chunks) > 1
    assert chunks[0].chunk_id.endswith("_chunk_0001")
    assert chunks[1].sequence == 2
    assert chunks[0].char_start < chunks[0].char_end
