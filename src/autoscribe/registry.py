from __future__ import annotations

from autoscribe.models import DocumentType, PipelineDefinition


PIPELINES: dict[str, PipelineDefinition] = {
    "broadcast_transcript_pipeline": PipelineDefinition(
        pipeline_id="broadcast_transcript_pipeline",
        document_types=[DocumentType.radio_transcript, DocumentType.podcast_transcript],
        agents=["structure_mapper", "broadcast_summary", "critique_caveat", "output_formatter"],
        output_title="Broadcast Transcript Summary",
        required_sections=[
            "Programme Type",
            "Participants",
            "Main Topics",
            "Segment-by-Segment Summary",
            "Key Claims",
            "Notable Quotes or Phrases",
            "Framing and Tone",
            "Factual Claims Requiring Verification",
            "Overall Summary",
        ],
    ),
    "academic_thesis_pipeline": PipelineDefinition(
        pipeline_id="academic_thesis_pipeline",
        document_types=[DocumentType.phd_thesis],
        agents=["structure_mapper", "thesis_summary", "critique_caveat", "output_formatter"],
        output_title="PhD Thesis Summary",
        required_sections=[
            "Bibliographic Metadata",
            "Thesis Topic",
            "Research Questions",
            "Central Argument",
            "Chapter-by-Chapter Summary",
            "Methodology",
            "Evidence Base",
            "Main Findings",
            "Original Contribution",
            "Limitations",
            "Theoretical Framework",
            "Practical Relevance",
            "Strongest Critique",
            "Overall Assessment",
        ],
    ),
    "policy_analysis_pipeline": PipelineDefinition(
        pipeline_id="policy_analysis_pipeline",
        document_types=[DocumentType.policy_document],
        agents=["structure_mapper", "policy_summary", "critique_caveat", "output_formatter"],
        output_title="Policy Document Analysis",
        required_sections=[
            "Policy Area",
            "Problem Definition",
            "Proposed Measures",
            "Target Groups",
            "Assumptions",
            "Incentives",
            "Enforcement Mechanisms",
            "Likely Behavioural Effects",
            "Distributional Effects",
            "Implementation Risks",
            "Strongest Challenge to the Framing",
            "Practical Implications",
        ],
    ),
    "generic_document_pipeline": PipelineDefinition(
        pipeline_id="generic_document_pipeline",
        document_types=[DocumentType.unknown, DocumentType.mixed_document],
        agents=["structure_mapper", "generic_summary", "critique_caveat", "output_formatter"],
        output_title="General Document Summary",
        required_sections=[
            "Probable Document Type",
            "Main Topics",
            "Structure",
            "Key Claims",
            "Important Details",
            "Uncertainties",
            "Suggested Better Classification",
            "Overall Summary",
        ],
    ),
}

DOCUMENT_TYPE_TO_PIPELINE: dict[DocumentType, str] = {
    DocumentType.radio_transcript: "broadcast_transcript_pipeline",
    DocumentType.podcast_transcript: "broadcast_transcript_pipeline",
    DocumentType.phd_thesis: "academic_thesis_pipeline",
    DocumentType.policy_document: "policy_analysis_pipeline",
    DocumentType.unknown: "generic_document_pipeline",
    DocumentType.mixed_document: "generic_document_pipeline",
}


def select_pipeline(document_type: DocumentType, confidence: float, override: str | None = None) -> PipelineDefinition:
    if override:
        return get_pipeline(override)
    if confidence < 0.50:
        return PIPELINES["generic_document_pipeline"]
    return PIPELINES[DOCUMENT_TYPE_TO_PIPELINE.get(document_type, "generic_document_pipeline")]


def get_pipeline(pipeline_id: str) -> PipelineDefinition:
    try:
        return PIPELINES[pipeline_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline: {pipeline_id}") from exc
