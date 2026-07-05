from __future__ import annotations

import json

from autoscribe.agents import classify_document
from autoscribe.models import AppConfig, Document, DocumentMetadata


def test_classify_document_accepts_string_secondary_types():
    backend = StaticBackend(
        {
            "document_type": "phd_thesis",
            "confidence": 0.95,
            "secondary_types": ["academic_paper"],
            "reasoning_summary": "Looks like a thesis.",
            "detected_features": ["chapter structure"],
            "recommended_pipeline": "academic_summarization",
        }
    )
    document = Document(
        document_id="doc_test",
        text="This PhD thesis has chapters.",
        input_hash="abc",
        metadata=DocumentMetadata(),
    )

    result = classify_document(backend, AppConfig(), document, document.text)

    assert result.document_type == "phd_thesis"
    assert result.secondary_types[0].document_type == "academic_paper"
    assert result.secondary_types[0].confidence == 0.0


class StaticBackend:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def complete(self, *args, **kwargs) -> str:
        return json.dumps(self.payload)
