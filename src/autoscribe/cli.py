from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from autoscribe.backends import make_backend
from autoscribe.config import load_config
from autoscribe.errors import AutoscribeError
from autoscribe.input import load_document
from autoscribe.orchestrator import Orchestrator
from autoscribe.storage import RunStore

app = typer.Typer(help="Classify and summarise documents with type-aware LLM pipelines.")


def _orchestrator(
    backend_name: str | None,
    model: str | None,
    data_dir: Path | None,
    config_path: Path | None,
) -> Orchestrator:
    config = load_config(
        config_path,
        {
            "llm_backend": backend_name,
            "model": model,
            "data_dir": str(data_dir) if data_dir else None,
        },
    )
    backend = make_backend(config)
    return Orchestrator(backend, config, RunStore(Path(config.data_dir)))


@app.command()
def ingest(
    input_file: Path,
    metadata: Optional[Path] = typer.Option(None, "--metadata", help="Optional metadata JSON file."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Storage directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config JSON file."),
) -> None:
    """Load, normalise, and store a document without running LLM agents."""
    try:
        cfg = load_config(config, {"data_dir": str(data_dir) if data_dir else None})
        store = RunStore(Path(cfg.data_dir))
        document = load_document(input_file, metadata)
        path = store.save_document(document)
        typer.echo(json.dumps({"document_id": document.document_id, "path": str(path)}, indent=2))
    except AutoscribeError as exc:
        _fail(exc)


@app.command()
def classify(
    input_file: Path,
    metadata: Optional[Path] = typer.Option(None, "--metadata", help="Optional metadata JSON file."),
    backend: Optional[str] = typer.Option(None, "--backend", help="mock or openai-compatible."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name."),
    document_type: Optional[str] = typer.Option(None, "--type", help="Manual document type override."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Storage directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config JSON file."),
) -> None:
    """Classify a document and print structured classification JSON."""
    try:
        orchestrator = _orchestrator(backend, model, data_dir, config)
        document = load_document(input_file, metadata)
        result = orchestrator.classify_only(document, document_type)
        typer.echo(result.model_dump_json(indent=2))
    except AutoscribeError as exc:
        _fail(exc)
    except ValueError as exc:
        _fail_message(str(exc))


@app.command()
def summarise(
    input_file: Path,
    metadata: Optional[Path] = typer.Option(None, "--metadata", help="Optional metadata JSON file."),
    backend: Optional[str] = typer.Option(None, "--backend", help="mock or openai-compatible."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name."),
    document_type: Optional[str] = typer.Option(None, "--type", help="Manual document type override."),
    pipeline: Optional[str] = typer.Option(None, "--pipeline", help="Manual pipeline override."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Storage directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config JSON file."),
) -> None:
    """Run the full classify, pipeline, agent, and export flow."""
    try:
        orchestrator = _orchestrator(backend, model, data_dir, config)
        document = load_document(input_file, metadata)
        run, _, _ = orchestrator.summarise(document, document_type, pipeline)
        run_dir = Path(orchestrator.config.data_dir) / "runs" / run.run_id
        typer.echo(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "document_id": run.document_id,
                    "pipeline": run.pipeline_id,
                    "markdown": str(run_dir / "final.md"),
                    "json": str(run_dir / "final.json"),
                },
                indent=2,
            )
        )
    except AutoscribeError as exc:
        _fail(exc)
    except ValueError as exc:
        _fail_message(str(exc))


@app.command()
def export(
    run_id: str,
    output_format: str = typer.Option("markdown", "--format", help="markdown or json."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Storage directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config JSON file."),
) -> None:
    """Copy a completed run export to data/exports."""
    try:
        if output_format not in {"markdown", "json"}:
            raise ValueError("--format must be markdown or json")
        cfg = load_config(config, {"data_dir": str(data_dir) if data_dir else None})
        destination = RunStore(Path(cfg.data_dir)).export(run_id, output_format)
        typer.echo(str(destination))
    except AutoscribeError as exc:
        _fail(exc)
    except (ValueError, FileNotFoundError) as exc:
        _fail_message(str(exc))


def _fail(exc: AutoscribeError) -> None:
    typer.echo(json.dumps(exc.to_dict(), indent=2), err=True)
    raise typer.Exit(code=1)


def _fail_message(message: str) -> None:
    typer.echo(json.dumps({"error": "command_error", "message": message}, indent=2), err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
