from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autoscribe.models import AgentOutput, Chunk, ClassificationResult, Document, PipelineRun


class RunStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents_dir = data_dir / "documents"
        self.chunks_dir = data_dir / "chunks"
        self.runs_dir = data_dir / "runs"
        self.exports_dir = data_dir / "exports"
        for path in [self.documents_dir, self.chunks_dir, self.runs_dir, self.exports_dir, data_dir / "configs"]:
            path.mkdir(parents=True, exist_ok=True)

    def save_document(self, document: Document) -> Path:
        path = self.documents_dir / f"{document.document_id}.json"
        _write_json(path, document)
        return path

    def save_chunks(self, document_id: str, chunks: list[Chunk]) -> Path:
        path = self.chunks_dir / f"{document_id}.json"
        _write_json(path, chunks)
        return path

    def create_run_dir(self, run_id: str) -> Path:
        run_dir = self.runs_dir / run_id
        (run_dir / "agent_outputs").mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_classification(self, run_id: str, classification: ClassificationResult) -> Path:
        path = self.create_run_dir(run_id) / "classification.json"
        _write_json(path, classification)
        return path

    def save_pipeline(self, run: PipelineRun) -> Path:
        path = self.create_run_dir(run.run_id) / "pipeline.json"
        _write_json(path, run)
        return path

    def save_agent_output(self, run_id: str, index: int, output: AgentOutput) -> Path:
        path = self.create_run_dir(run_id) / "agent_outputs" / f"{index:02d}_{output.agent_id}.json"
        _write_json(path, output)
        return path

    def save_final(self, run_id: str, markdown: str, json_output: dict[str, Any]) -> tuple[Path, Path]:
        run_dir = self.create_run_dir(run_id)
        md_path = run_dir / "final.md"
        json_path = run_dir / "final.json"
        md_path.write_text(markdown, encoding="utf-8")
        _write_json(json_path, json_output)
        return md_path, json_path

    def export(self, run_id: str, output_format: str) -> Path:
        run_dir = self.runs_dir / run_id
        source = run_dir / ("final.md" if output_format == "markdown" else "final.json")
        if not source.exists():
            raise FileNotFoundError(f"No {output_format} export found for run {run_id}")
        destination = self.exports_dir / f"{run_id}.{source.suffix.lstrip('.')}"
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return destination


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    elif isinstance(value, list):
        payload = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in value]
    else:
        payload = value
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
