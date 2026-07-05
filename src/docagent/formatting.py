from __future__ import annotations

from typing import Any

from docagent.models import AgentOutput, AppConfig, ClassificationResult, Document, PipelineDefinition, PipelineRun
from docagent.prompts import AGENT_PROMPT_VERSION, CLASSIFIER_PROMPT_VERSION


ADVICE_BOUNDARY_TYPES = {
    "legal_or_regulatory_document",
    "administrative_correspondence",
}


def build_outputs(
    document: Document,
    classification: ClassificationResult,
    pipeline: PipelineDefinition,
    run: PipelineRun,
    agent_outputs: list[AgentOutput],
    config: AppConfig,
) -> tuple[str, dict[str, Any]]:
    warnings = list(classification.warnings)
    for output in agent_outputs:
        warnings.extend(output.warnings)

    sections = _section_content(pipeline, agent_outputs)
    markdown = _markdown(document, classification, pipeline, run, agent_outputs, sections, warnings)
    json_output = {
        "document_id": document.document_id,
        "run_id": run.run_id,
        "document_type": classification.document_type.value,
        "classification_confidence": classification.confidence,
        "classification_verified": classification.verified,
        "pipeline_used": pipeline.pipeline_id,
        "summary": {
            "short": _first_summary(agent_outputs),
            "detailed": "\n\n".join(_unique_nonempty(output.summary for output in agent_outputs)),
        },
        "sections": sections,
        "key_claims": _collect_key_points(agent_outputs),
        "evidence": [],
        "limitations": sections.get("Limitations", []),
        "warnings": warnings,
        "source_references": [ref.model_dump(mode="json") for ref in _unique_source_references(agent_outputs)],
        "metadata": document.metadata.model_dump(mode="json"),
        "reproducibility": {
            "input_hash": document.input_hash,
            "model": config.model,
            "classification_model": config.classification_model or config.model,
            "summarisation_model": config.summarisation_model or config.model,
            "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
            "agent_prompt_version": AGENT_PROMPT_VERSION,
            "pipeline_version": pipeline.version,
            "config": config.model_dump(mode="json"),
        },
    }
    return markdown, json_output


def _markdown(
    document: Document,
    classification: ClassificationResult,
    pipeline: PipelineDefinition,
    run: PipelineRun,
    agent_outputs: list[AgentOutput],
    sections: dict[str, list[str]],
    warnings: list[str],
) -> str:
    lines = [
        f"# {pipeline.output_title}",
        "",
        f"- Document ID: `{document.document_id}`",
        f"- Run ID: `{run.run_id}`",
        f"- Document type: `{classification.document_type.value}`",
        f"- Classification confidence: `{classification.confidence:.2f}`",
        f"- Classification verified: `{classification.verified}`",
        f"- Processing pipeline: `{pipeline.pipeline_id}`",
        f"- Agent sequence: `{', '.join(run.agent_sequence)}`",
        "",
    ]
    if document.metadata.title:
        lines.extend([f"Source title: {document.metadata.title}", ""])

    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
        lines.append("")

    if classification.document_type.value in ADVICE_BOUNDARY_TYPES or _contains_high_stakes_feature(classification):
        lines.extend(
            [
                "## Professional Advice Boundary",
                "",
                "This summary is an analytical aid. It is not a substitute for advice from a qualified professional.",
                "",
            ]
        )

    for section in pipeline.required_sections:
        lines.extend([f"## {section}", ""])
        values = sections.get(section) or ["No specific information extracted in the MVP pass."]
        lines.extend(f"- {value}" for value in values)
        lines.append("")

    lines.extend(["## Source References", ""])
    refs = _unique_source_references(agent_outputs)
    if refs:
        for ref in refs[:20]:
            suffix = f" chars {ref.char_start}-{ref.char_end}" if ref.char_start is not None else ""
            lines.append(f"- `{ref.chunk_id}`{suffix}")
    else:
        lines.append("- No source references available.")

    lines.extend(["", "## Suggested Next Actions", "", "- Review source chunks for any high-stakes or surprising claims.", "- Rerun with a manual document type or pipeline if the classification is uncertain."])
    return "\n".join(lines).strip() + "\n"


def _section_content(pipeline: PipelineDefinition, outputs: list[AgentOutput]) -> dict[str, list[str]]:
    joined = _unique_nonempty(output.summary for output in outputs)
    key_points = _collect_key_points(outputs)
    uncertainties = _collect_uncertainties(outputs)
    sections: dict[str, list[str]] = {}
    for section in pipeline.required_sections:
        lower = section.lower()
        section_values = _collect_section_values(section, outputs)
        if section_values:
            sections[section] = section_values
        elif "uncert" in lower or "ambigu" in lower or "critique" in lower or "risk" in lower or "limitation" in lower:
            sections[section] = uncertainties or ["No specific caveats extracted for this section."]
        elif "claim" in lower or "finding" in lower or "topic" in lower or "question" in lower:
            sections[section] = key_points or ["No section-specific claims extracted."]
        elif "overall" in lower or "assessment" in lower:
            sections[section] = joined[:2] or ["No summary available."]
        else:
            sections[section] = ["No specific information extracted for this section."]
    return sections


def _collect_key_points(outputs: list[AgentOutput]) -> list[str]:
    points: list[str] = []
    for output in outputs:
        raw = output.structured_data.get("key_points")
        if isinstance(raw, list):
            points.extend(str(item) for item in raw if item)
    return points


def _collect_uncertainties(outputs: list[AgentOutput]) -> list[str]:
    items: list[str] = []
    for output in outputs:
        raw = output.structured_data.get("uncertainties")
        if isinstance(raw, list):
            items.extend(str(item) for item in raw if item)
    return items


def _collect_section_values(section: str, outputs: list[AgentOutput]) -> list[str]:
    values: list[str] = []
    wanted = _normalise_key(section)
    for output in outputs:
        structured = output.structured_data
        for key, value in structured.items():
            if _normalise_key(str(key)) == wanted:
                values.extend(_coerce_section_values(value))

        sections = structured.get("sections")
        if isinstance(sections, dict):
            for key, value in sections.items():
                if _normalise_key(str(key)) == wanted:
                    values.extend(_coerce_section_values(value))
        elif isinstance(sections, list):
            for item in sections:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or item.get("section") or item.get("heading")
                if title and _normalise_key(str(title)) == wanted:
                    values.extend(_coerce_section_values(item.get("content") or item.get("summary") or item.get("value")))

    return _unique_nonempty(values)


def _coerce_section_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                content = item.get("content") or item.get("summary") or item.get("value") or item.get("text")
                if content is not None:
                    values.append(str(content))
            elif item is not None:
                values.append(str(item))
        return values
    if isinstance(value, dict):
        content = value.get("content") or value.get("summary") or value.get("value") or value.get("text")
        return [str(content)] if content is not None else []
    return [str(value)]


def _normalise_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _unique_nonempty(values) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _unique_source_references(outputs: list[AgentOutput]):
    refs = []
    seen = set()
    for output in outputs:
        for ref in output.source_references:
            key = (ref.chunk_id, ref.char_start, ref.char_end, ref.section, ref.speaker, ref.page)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    return refs


def _first_summary(outputs: list[AgentOutput]) -> str:
    for output in outputs:
        if output.summary:
            return output.summary
    return ""


def _contains_high_stakes_feature(classification: ClassificationResult) -> bool:
    text = " ".join([classification.document_type.value, *classification.detected_features]).lower()
    return any(term in text for term in ["legal", "regulatory", "benefit", "welfare", "medical", "financial", "safety"])
