from __future__ import annotations

from autoscribe.formatting import _section_content, build_outputs
from autoscribe.models import AgentOutput, AppConfig, ClassificationResult, Document, PipelineDefinition, PipelineRun, SourceReference


def test_section_content_does_not_repeat_same_summary_across_all_sections():
    pipeline = PipelineDefinition(
        pipeline_id="academic_thesis_pipeline",
        document_types=["phd_thesis"],
        agents=["thesis_summary", "output_formatter"],
        output_title="PhD Thesis Summary",
        required_sections=["Thesis Topic", "Methodology", "Overall Assessment"],
    )
    outputs = [
        AgentOutput(
            agent_id="thesis_summary",
            output_type="summary",
            summary="Repeated generic thesis summary.",
            structured_data={},
            confidence=0.8,
        ),
        AgentOutput(
            agent_id="output_formatter",
            output_type="summary",
            summary="Repeated generic thesis summary.",
            structured_data={},
            confidence=0.8,
        ),
    ]

    sections = _section_content(pipeline, outputs)

    assert sections["Thesis Topic"] == ["No section-specific claims extracted."]
    assert sections["Methodology"] == ["No specific information extracted for this section."]
    assert sections["Overall Assessment"] == ["Repeated generic thesis summary."]


def test_section_content_prefers_matching_structured_data():
    pipeline = PipelineDefinition(
        pipeline_id="academic_thesis_pipeline",
        document_types=["phd_thesis"],
        agents=["output_formatter"],
        output_title="PhD Thesis Summary",
        required_sections=["Thesis Topic", "Methodology"],
    )
    outputs = [
        AgentOutput(
            agent_id="output_formatter",
            output_type="summary",
            summary="Generic summary.",
            structured_data={
                "sections": {
                    "Thesis Topic": "Preceptorship for newly qualified nurses.",
                    "Methodology": ["Action Research", "Reflective appraisal"],
                }
            },
            confidence=0.8,
        )
    ]

    sections = _section_content(pipeline, outputs)

    assert sections["Thesis Topic"] == ["Preceptorship for newly qualified nurses."]
    assert sections["Methodology"] == ["Action Research", "Reflective appraisal"]


def test_build_outputs_deduplicates_source_references():
    document = Document(document_id="doc_test", text="text", input_hash="abc")
    classification = ClassificationResult(document_type="phd_thesis", confidence=0.9, recommended_pipeline="academic_thesis_pipeline")
    pipeline = PipelineDefinition(
        pipeline_id="academic_thesis_pipeline",
        document_types=["phd_thesis"],
        agents=["thesis_summary", "output_formatter"],
        output_title="PhD Thesis Summary",
        required_sections=["Overall Assessment"],
    )
    run = PipelineRun(
        run_id="run_test",
        document_id=document.document_id,
        pipeline_id=pipeline.pipeline_id,
        agent_sequence=pipeline.agents,
        status="completed",
    )
    ref = SourceReference(chunk_id="chunk_1", char_start=0, char_end=10)
    outputs = [
        AgentOutput(agent_id="a", output_type="summary", summary="Summary.", source_references=[ref], confidence=0.8),
        AgentOutput(agent_id="b", output_type="summary", summary="Summary.", source_references=[ref], confidence=0.8),
    ]

    markdown, json_output = build_outputs(document, classification, pipeline, run, outputs, AppConfig())

    assert markdown.count("`chunk_1` chars 0-10") == 1
    assert len(json_output["source_references"]) == 1
