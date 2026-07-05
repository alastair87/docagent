from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from autoscribe.backends import LLMBackend, parse_json_response
from autoscribe.errors import BackendError, ClassificationError
from autoscribe.models import AgentInput, AgentOutput, AppConfig, ClassificationResult, Document, DocumentType, SourceReference
from autoscribe.prompts import CLASSIFIER_SYSTEM_PROMPT, agent_system_prompt, agent_user_prompt, classifier_user_prompt
from autoscribe.registry import DOCUMENT_TYPE_TO_PIPELINE


def classify_document(
    backend: LLMBackend,
    config: AppConfig,
    document: Document,
    chunks_text: str,
    override_type: str | None = None,
) -> ClassificationResult:
    raw = backend.complete(
        [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": classifier_user_prompt(chunks_text, document.metadata.model_dump(mode="json"))},
        ],
        schema={},
        temperature=config.temperature,
        model=config.classification_model or config.model,
        timeout=config.timeout_seconds,
    )
    try:
        data = parse_json_response(raw)
        result = ClassificationResult(**data)
    except (BackendError, ValidationError, ValueError) as exc:
        raise ClassificationError(
            "The system could not parse a valid classification result.",
            ["Use the generic pipeline.", "Manually select a document type.", "Retry with a JSON-capable model."],
        ) from exc

    if override_type:
        result.warnings.append(f"Document type manually overridden from {result.document_type.value} to {override_type}.")
        result.document_type = DocumentType(override_type)
        result.recommended_pipeline = DOCUMENT_TYPE_TO_PIPELINE.get(result.document_type, "generic_document_pipeline")
        result.verified = True
    else:
        result.verified = result.confidence >= 0.80
        if 0.50 <= result.confidence < 0.80:
            result.warnings.append("Classification is uncertain; selected pipeline should be reviewed.")
        elif result.confidence < 0.50:
            result.warnings.append("Classification confidence is low; using generic exploratory pipeline.")
    return result


class AgentRunner:
    def __init__(self, backend: LLMBackend, config: AppConfig):
        self.backend = backend
        self.config = config

    def run_agent(self, agent_id: str, agent_input: AgentInput, pipeline) -> AgentOutput:
        context = _build_context(agent_input)
        raw = self.backend.complete(
            [
                {"role": "system", "content": agent_system_prompt(agent_id, pipeline)},
                {"role": "user", "content": agent_user_prompt(agent_id, context)},
            ],
            schema={},
            temperature=self.config.temperature,
            model=self.config.summarisation_model or self.config.model,
            timeout=self.config.timeout_seconds,
        )
        try:
            data = parse_json_response(raw)
        except BackendError as exc:
            raise BackendError(f"Agent {agent_id} returned invalid JSON.") from exc

        references = [
            SourceReference(chunk_id=chunk.chunk_id, char_start=chunk.char_start, char_end=chunk.char_end)
            for chunk in agent_input.chunks[:3]
        ]
        return AgentOutput(
            agent_id=agent_id,
            output_type=agent_id,
            summary=str(data.get("summary", "")),
            structured_data=_object_or_empty(data.get("structured_data")),
            source_references=references,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            warnings=[str(item) for item in data.get("warnings", []) if item],
        )


def _build_context(agent_input: AgentInput) -> str:
    chunk_excerpt = "\n\n".join(
        f"[{chunk.chunk_id} chars={chunk.char_start}-{chunk.char_end}]\n{chunk.text[:2500]}" for chunk in agent_input.chunks[:6]
    )
    return f"""document_id: {agent_input.document_id}
document_type: {agent_input.document_type.value}
user_goal: {agent_input.user_goal or ""}
instructions: {agent_input.instructions}

prior_outputs:
{agent_input.prior_outputs}

chunks:
{chunk_excerpt}
"""


def _object_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
