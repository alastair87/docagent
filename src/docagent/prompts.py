from __future__ import annotations

from importlib import resources

from docagent.models import PipelineDefinition


CLASSIFIER_PROMPT_VERSION = "0.1.0"
AGENT_PROMPT_VERSION = "0.1.0"


_TEMPLATES = resources.files("docagent.prompt_templates")


def _read_template(path: str) -> str:
    return _TEMPLATES.joinpath(path).read_text(encoding="utf-8").rstrip() + "\n"


def _render_template(path: str, **values: object) -> str:
    return _read_template(path).format(**values)


CLASSIFIER_SYSTEM_PROMPT = _read_template("classifier_system.md")


def classifier_user_prompt(sample: str, metadata: dict[str, object]) -> str:
    return _render_template("classifier_user.md", sample=sample, metadata=metadata)


def agent_system_prompt(agent_id: str, pipeline: PipelineDefinition) -> str:
    return _render_template(
        "agent_system.md",
        agent_id=agent_id,
        pipeline_id=pipeline.pipeline_id,
        required_sections=pipeline.required_sections,
    )


def agent_user_prompt(agent_id: str, context: str) -> str:
    path = f"agents/{agent_id}.md"
    if not _TEMPLATES.joinpath(path).is_file():
        path = "agents/default.md"
    return _render_template(path, agent_id=agent_id, context=context)
