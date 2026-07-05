import json
from pathlib import Path

from typer.testing import CliRunner

from autoscribe.backends import MockBackend
from autoscribe.config import load_config
from autoscribe.input import load_document
from autoscribe.orchestrator import Orchestrator
from autoscribe.registry import select_pipeline
from autoscribe.storage import RunStore
from autoscribe.cli import app


def test_low_confidence_selects_generic_pipeline():
    pipeline = select_pipeline(document_type="unknown", confidence=0.42)

    assert pipeline.pipeline_id == "generic_document_pipeline"


def test_broadcast_sample_runs_end_to_end(tmp_path: Path):
    run = _summarise_sample("broadcast.txt", tmp_path)

    assert run.pipeline_id == "broadcast_transcript_pipeline"
    assert (tmp_path / "runs" / run.run_id / "classification.json").exists()
    assert (tmp_path / "runs" / run.run_id / "final.md").exists()
    assert "Broadcast Transcript Summary" in (tmp_path / "runs" / run.run_id / "final.md").read_text(encoding="utf-8")


def test_thesis_sample_runs_end_to_end(tmp_path: Path):
    run = _summarise_sample("thesis_excerpt.txt", tmp_path)

    assert run.pipeline_id == "academic_thesis_pipeline"
    assert "PhD Thesis Summary" in (tmp_path / "runs" / run.run_id / "final.md").read_text(encoding="utf-8")


def test_policy_sample_runs_end_to_end(tmp_path: Path):
    run = _summarise_sample("policy.txt", tmp_path)

    assert run.pipeline_id == "policy_analysis_pipeline"
    final_json = json.loads((tmp_path / "runs" / run.run_id / "final.json").read_text(encoding="utf-8"))
    assert final_json["pipeline_used"] == "policy_analysis_pipeline"
    assert final_json["source_references"]


def test_cli_summarise_outputs_paths(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "summarise",
            "samples/broadcast.txt",
            "--backend",
            "mock",
            "--data-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"].startswith("run_")
    assert Path(payload["markdown"]).exists()


def test_manual_type_override_routes_to_thesis(tmp_path: Path):
    run = _summarise_sample("broadcast.txt", tmp_path, document_type_override="phd_thesis")

    assert run.pipeline_id == "academic_thesis_pipeline"


def _summarise_sample(name: str, tmp_path: Path, document_type_override: str | None = None):
    config = load_config(overrides={"llm_backend": "mock", "data_dir": str(tmp_path)})
    orchestrator = Orchestrator(MockBackend(), config, RunStore(tmp_path))
    document = load_document(Path("samples") / name)
    run, _, _ = orchestrator.summarise(document, document_type_override=document_type_override)
    return run
