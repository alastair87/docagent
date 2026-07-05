import json
from pathlib import Path

from typer.testing import CliRunner

from docagent.backends import MockBackend
from docagent.config import load_config
from docagent.input import load_document
from docagent.orchestrator import Orchestrator
from docagent.registry import select_pipeline
from docagent.storage import RunStore
from docagent.cli import app


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


def test_cli_summarise_verbose_writes_progress_to_stderr(tmp_path: Path):
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
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_id"].startswith("run_")
    assert "[docagent] Loading input file samples/broadcast.txt" in result.stderr
    assert "[docagent] Starting agent 1/4: structure_mapper" in result.stderr
    assert "[docagent] Completed run" in result.stderr


def test_cli_classify_verbose_keeps_stdout_json(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "classify",
            "samples/policy.txt",
            "--backend",
            "mock",
            "--data-dir",
            str(tmp_path),
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["document_type"] == "policy_document"
    assert "[docagent] Chunking and classifying document" in result.stderr


def test_cli_export_verbose_keeps_stdout_path(tmp_path: Path):
    run = _summarise_sample("broadcast.txt", tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export",
            run.run_id,
            "--format",
            "markdown",
            "--data-dir",
            str(tmp_path),
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip().endswith(f"{run.run_id}.md")
    assert "[docagent] Resolving markdown export" in result.stderr


def test_orchestrator_progress_callback_receives_agent_events(tmp_path: Path):
    config = load_config(overrides={"llm_backend": "mock", "data_dir": str(tmp_path)})
    orchestrator = Orchestrator(MockBackend(), config, RunStore(tmp_path))
    document = load_document(Path("samples") / "broadcast.txt")
    events: list[str] = []

    orchestrator.summarise(document, progress=events.append)

    assert "Chunking document" in events
    assert "Starting agent 1/4: structure_mapper" in events
    assert "Completed agent 4/4: output_formatter" in events


def test_manual_type_override_routes_to_thesis(tmp_path: Path):
    run = _summarise_sample("broadcast.txt", tmp_path, document_type_override="phd_thesis")

    assert run.pipeline_id == "academic_thesis_pipeline"


def _summarise_sample(name: str, tmp_path: Path, document_type_override: str | None = None):
    config = load_config(overrides={"llm_backend": "mock", "data_dir": str(tmp_path)})
    orchestrator = Orchestrator(MockBackend(), config, RunStore(tmp_path))
    document = load_document(Path("samples") / name)
    run, _, _ = orchestrator.summarise(document, document_type_override=document_type_override)
    return run
