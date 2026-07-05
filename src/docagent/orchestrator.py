from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from docagent.agents import AgentRunner, classify_document
from docagent.backends import LLMBackend
from docagent.chunking import chunk_document
from docagent.formatting import build_outputs
from docagent.models import AgentInput, AppConfig, DocumentType, PipelineRun
from docagent.registry import select_pipeline
from docagent.storage import RunStore


class Orchestrator:
    def __init__(self, backend: LLMBackend, config: AppConfig, store: RunStore):
        self.backend = backend
        self.config = config
        self.store = store
        self.agent_runner = AgentRunner(backend, config)

    def summarise(
        self,
        document,
        document_type_override: str | None = None,
        pipeline_override: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[PipelineRun, str, dict[str, object]]:
        _progress(progress, "Chunking document")
        chunks = chunk_document(document, self.config.chunk_size_tokens, self.config.chunk_overlap_tokens)
        _progress(progress, f"Created {len(chunks)} chunk(s)")
        _progress(progress, "Storing document and chunks")
        self.store.save_document(document)
        self.store.save_chunks(document.document_id, chunks)

        sample = "\n\n".join(chunk.text for chunk in chunks[:4])
        _progress(progress, "Classifying document")
        classification = classify_document(self.backend, self.config, document, sample, document_type_override)
        pipeline = select_pipeline(classification.document_type, classification.confidence, pipeline_override)
        _progress(
            progress,
            f"Selected {pipeline.pipeline_id} for {classification.document_type.value} "
            f"at confidence {classification.confidence:.2f}",
        )
        run = PipelineRun(
            run_id=f"run_{uuid4().hex[:12]}",
            document_id=document.document_id,
            pipeline_id=pipeline.pipeline_id,
            agent_sequence=pipeline.agents,
            status="running",
            warnings=list(classification.warnings),
        )
        _progress(progress, f"Created run {run.run_id}")
        self.store.create_run_dir(run.run_id)
        self.store.save_classification(run.run_id, classification)
        self.store.save_pipeline(run)

        prior_outputs = {}
        agent_outputs = []
        for index, agent_id in enumerate(pipeline.agents, start=1):
            _progress(progress, f"Starting agent {index}/{len(pipeline.agents)}: {agent_id}")
            agent_input = AgentInput(
                document_id=document.document_id,
                document_type=classification.document_type if classification.confidence >= 0.50 else DocumentType.unknown,
                input_text=sample,
                chunks=chunks,
                prior_outputs=prior_outputs,
                user_goal=document.metadata.user_goal,
                instructions=document.metadata.special_instructions or "",
            )
            output = self.agent_runner.run_agent(agent_id, agent_input, pipeline)
            agent_outputs.append(output)
            prior_outputs[agent_id] = output.model_dump(mode="json")
            self.store.save_agent_output(run.run_id, index, output)
            _progress(progress, f"Completed agent {index}/{len(pipeline.agents)}: {agent_id}")

        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc).isoformat()
        self.store.save_pipeline(run)

        _progress(progress, "Writing final Markdown and JSON outputs")
        markdown, json_output = build_outputs(document, classification, pipeline, run, agent_outputs, self.config)
        self.store.save_final(run.run_id, markdown, json_output)
        _progress(progress, f"Completed run {run.run_id}")
        return run, markdown, json_output

    def classify_only(self, document, document_type_override: str | None = None):
        chunks = chunk_document(document, self.config.chunk_size_tokens, self.config.chunk_overlap_tokens)
        sample = "\n\n".join(chunk.text for chunk in chunks[:4])
        return classify_document(self.backend, self.config, document, sample, document_type_override)


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
