from __future__ import annotations

from autoscribe.models import PipelineDefinition


CLASSIFIER_PROMPT_VERSION = "0.1.0"
AGENT_PROMPT_VERSION = "0.1.0"


CLASSIFIER_SYSTEM_PROMPT = """You classify documents for an adaptive summarisation system.
Return only JSON. Be conservative. Do not claim certainty unless the text strongly supports it.
Allowed document_type values are: radio_transcript, podcast_transcript, interview_transcript,
meeting_transcript, phd_thesis, academic_paper, policy_document, legal_or_regulatory_document,
book_chapter, essay_or_article, technical_documentation, administrative_correspondence,
personal_notes, mixed_document, unknown.
"""


def classifier_user_prompt(sample: str, metadata: dict[str, object]) -> str:
    return f"""Classify the document.

Return JSON with:
document_type, confidence, secondary_types, reasoning_summary, detected_features, recommended_pipeline.

Metadata:
{metadata}

Document sample:
{sample}
"""


def agent_system_prompt(agent_id: str, pipeline: PipelineDefinition) -> str:
    return f"""You are the {agent_id} agent in {pipeline.pipeline_id}.
Return only JSON with fields: summary, structured_data, confidence, warnings.
Preserve caveats. Distinguish source claims from verified facts. Use chunk IDs when referring to evidence.
The final output for this pipeline must support these sections: {pipeline.required_sections}.
"""


def agent_user_prompt(agent_id: str, context: str) -> str:
    role = {
        "structure_mapper": "Structure Mapper: find visible structure, sections, speakers, headings, or topical blocks.",
        "broadcast_summary": "Broadcast Summary: summarise spoken material without treating repetition as written structure.",
        "thesis_summary": "Thesis Summary: extract research questions, argument, method, findings, contribution, limitations, and relevance.",
        "policy_summary": "Policy Summary: extract problem framing, measures, stakeholders, assumptions, incentives, risks, and impacts.",
        "generic_summary": "Generic Summary: summarise probable type, topics, key claims, important details, uncertainty, and reclassification suggestions.",
        "critique_caveat": "Critique and Caveat: identify uncertainties, weak support, missing context, and professional-advice boundaries.",
        "output_formatter": "Output Formatter: consolidate prior outputs into final structured section content.",
    }.get(agent_id, f"{agent_id}: complete the requested pipeline step.")
    return f"""{role}

Context:
{context}
"""
