from __future__ import annotations

from typing import Any

from autoscribe.models import AgentOutput, AppConfig, ClassificationResult, Document, PipelineDefinition, PipelineRun
from autoscribe.prompts import AGENT_PROMPT_VERSION, CLASSIFIER_PROMPT_VERSION


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
            "detailed": "\n\n".join(output.summary for output in agent_outputs if output.summary),
        },
        "sections": sections,
        "key_claims": _collect_key_points(agent_outputs),
        "evidence": [],
        "limitations": sections.get("Limitations", []),
        "warnings": warnings,
        "source_references": [
            ref.model_dump(mode="json") for output in agent_outputs for ref in output.source_references
        ],
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
    refs = [ref for output in agent_outputs for ref in output.source_references]
    if refs:
        for ref in refs[:20]:
            suffix = f" chars {ref.char_start}-{ref.char_end}" if ref.char_start is not None else ""
            lines.append(f"- `{ref.chunk_id}`{suffix}")
    else:
        lines.append("- No source references available.")

    lines.extend(["", "## Suggested Next Actions", "", "- Review source chunks for any high-stakes or surprising claims.", "- Rerun with a manual document type or pipeline if the classification is uncertain."])
    return "\n".join(lines).strip() + "\n"


def _section_content(pipeline: PipelineDefinition, outputs: list[AgentOutput]) -> dict[str, list[str]]:
    joined = [output.summary for output in outputs if output.summary]
    key_points = _collect_key_points(outputs)
    uncertainties = _collect_uncertainties(outputs)
    sections: dict[str, list[str]] = {}
    for section in pipeline.required_sections:
        lower = section.lower()
        if "uncert" in lower or "ambigu" in lower or "critique" in lower or "risk" in lower:
            sections[section] = uncertainties or joined[:2]
        elif "claim" in lower or "finding" in lower or "topic" in lower or "question" in lower:
            sections[section] = key_points or joined[:2]
        else:
            sections[section] = joined[:2] or ["No summary available."]
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


def _first_summary(outputs: list[AgentOutput]) -> str:
    for output in outputs:
        if output.summary:
            return output.summary
    return ""


def _contains_high_stakes_feature(classification: ClassificationResult) -> bool:
    text = " ".join([classification.document_type.value, *classification.detected_features]).lower()
    return any(term in text for term in ["legal", "regulatory", "benefit", "welfare", "medical", "financial", "safety"])
