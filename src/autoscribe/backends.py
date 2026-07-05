from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from autoscribe.errors import BackendError
from autoscribe.models import AppConfig, DocumentType


GENERIC_JSON_GRAMMAR = r'''
root ::= object
value ::= object | array | string | number | ("true" | "false" | "null")
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))*
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws ::= [ \t\n\r]*
'''.strip()


class LLMBackend(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        model: str | None = None,
        timeout: int | None = None,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, config: AppConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise BackendError("The openai package is required for the OpenAI-compatible backend.") from exc
        kwargs: dict[str, Any] = {}
        if config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url
        self.client = OpenAI(**kwargs)
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        model: str | None = None,
        timeout: int | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout or self.config.timeout_seconds,
        }
        if schema is not None:
            self._apply_json_decoding(request)

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                content = _extract_message_content(response)
                if not content:
                    raise BackendError("LLM returned an empty response.")
                return content
            except Exception as exc:  # pragma: no cover - exercised in integration use
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise BackendError(f"LLM request failed: {last_error}")

    def _apply_json_decoding(self, request: dict[str, Any]) -> None:
        mode = self.config.json_decoding_mode
        if mode in {"grammar", "both"}:
            request.setdefault("extra_body", {})["grammar"] = self.config.json_grammar or GENERIC_JSON_GRAMMAR
        if mode in {"response_format", "both"}:
            request["response_format"] = {"type": "json_object"}


class MockBackend(LLMBackend):
    def complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        model: str | None = None,
        timeout: int | None = None,
    ) -> str:
        prompt = "\n".join(message.get("content", "") for message in messages).lower()
        if "classify the document" in prompt:
            return json.dumps(_mock_classification(prompt))
        if "structure mapper" in prompt:
            return json.dumps(
                {
                    "summary": "Detected the major visible sections, speakers, or topical blocks in the supplied chunks.",
                    "structured_data": {"sections": _mock_sections(prompt)},
                    "confidence": 0.74,
                    "warnings": [],
                }
            )
        if "critique" in prompt or "caveat" in prompt:
            return json.dumps(
                {
                    "summary": "The analysis should be treated as provisional. Claims are source claims unless independently verified.",
                    "structured_data": {
                        "uncertainties": [
                            "LLM classification is probabilistic.",
                            "Chunk-level provenance does not prove factual accuracy.",
                        ]
                    },
                    "confidence": 0.78,
                    "warnings": ["Review source passages for high-stakes decisions."],
                }
            )
        return json.dumps(
            {
                "summary": "Type-aware summary generated from the available chunks, preserving major topics and caveats.",
                "structured_data": {
                    "key_points": [
                        "The document contains multiple substantive points.",
                        "Important claims should be checked against source chunks.",
                    ]
                },
                "confidence": 0.72,
                "warnings": [],
            }
        )


def make_backend(config: AppConfig) -> LLMBackend:
    if config.llm_backend == "mock":
        return MockBackend()
    return OpenAICompatibleBackend(config)


def _extract_message_content(response: Any) -> str | None:
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if content:
        return content
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return reasoning_content
    return None


def parse_json_response(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendError("LLM returned invalid JSON.", ["Retry the run.", "Use a model with JSON-mode support."]) from exc
    if not isinstance(value, dict):
        raise BackendError("LLM JSON response must be an object.")
    return value


def _mock_classification(prompt: str) -> dict[str, Any]:
    sample = prompt.split("document sample:", 1)[-1]
    if any(term in sample for term in ["programme", "podcast", "host:", "guest:", "broadcast", "radio"]):
        return _classification(DocumentType.radio_transcript, 0.87, ["speakers", "broadcast register"], "broadcast_transcript_pipeline")
    if any(term in sample for term in ["phd", "thesis", "chapter", "methodology", "literature review"]):
        return _classification(DocumentType.phd_thesis, 0.88, ["chapters", "methodology", "academic register"], "academic_thesis_pipeline")
    if any(term in sample for term in ["policy", "stakeholder", "intervention", "implementation", "department"]):
        return _classification(DocumentType.policy_document, 0.84, ["policy problem", "proposed measures"], "policy_analysis_pipeline")
    if any(term in sample for term in ["legal", "regulation", "statutory", "jurisdiction", "rights", "duties"]):
        return _classification(DocumentType.legal_or_regulatory_document, 0.76, ["rights", "duties"], "generic_document_pipeline")
    return _classification(DocumentType.unknown, 0.42, ["insufficient distinctive features"], "generic_document_pipeline")


def _classification(document_type: DocumentType, confidence: float, features: list[str], pipeline: str) -> dict[str, Any]:
    return {
        "document_type": document_type.value,
        "confidence": confidence,
        "secondary_types": [],
        "reasoning_summary": "Mock classification based on visible lexical and structural features.",
        "detected_features": features,
        "recommended_pipeline": pipeline,
    }


def _mock_sections(prompt: str) -> list[dict[str, str]]:
    if "speaker" in prompt or "host:" in prompt:
        return [{"title": "Opening discussion"}, {"title": "Main interview or discussion"}, {"title": "Closing remarks"}]
    if "chapter" in prompt:
        return [{"title": "Abstract/Introduction"}, {"title": "Methodology"}, {"title": "Findings/Conclusion"}]
    return [{"title": "Opening"}, {"title": "Main body"}, {"title": "Conclusion or closing material"}]
