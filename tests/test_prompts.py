from __future__ import annotations

from docagent.models import DocumentType, PipelineDefinition
from docagent.prompts import CLASSIFIER_SYSTEM_PROMPT, agent_system_prompt, agent_user_prompt, classifier_user_prompt


def test_classifier_system_prompt_loads_markdown_asset():
    assert CLASSIFIER_SYSTEM_PROMPT.startswith("You classify documents")
    assert "Allowed document_type values" in CLASSIFIER_SYSTEM_PROMPT


def test_classifier_user_prompt_renders_metadata_and_sample():
    prompt = classifier_user_prompt("Chapter 1 text", {"title": "Example thesis"})

    assert "Classify the document." in prompt
    assert "{'title': 'Example thesis'}" in prompt
    assert "Chapter 1 text" in prompt


def test_agent_system_prompt_renders_pipeline_values():
    pipeline = PipelineDefinition(
        pipeline_id="test_pipeline",
        document_types=[DocumentType.unknown],
        agents=["structure_mapper"],
        output_title="Test Output",
        required_sections=["Summary", "Limitations"],
    )

    prompt = agent_system_prompt("structure_mapper", pipeline)

    assert "You are the structure_mapper agent in test_pipeline." in prompt
    assert "['Summary', 'Limitations']" in prompt


def test_agent_user_prompt_uses_agent_markdown_template():
    prompt = agent_user_prompt("structure_mapper", "chunk text")

    assert prompt.startswith("Structure Mapper:")
    assert "Context:\nchunk text" in prompt


def test_agent_user_prompt_uses_default_template_for_unknown_agent():
    prompt = agent_user_prompt("custom_agent", "context text")

    assert prompt.startswith("custom_agent: complete the requested pipeline step.")
    assert "Context:\ncontext text" in prompt
