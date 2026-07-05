# autoscribe

LLM-based document type classification and adaptive summarisation pipeline.

## Quick start

```bash
python -m pip install -e ".[dev]"
docagent summarise samples/broadcast.txt --backend mock
```

By default, outputs are written under `data/runs/<run_id>/`.

For a real OpenAI-compatible backend:

```bash
export OPENAI_API_KEY=...
docagent summarise input.txt --backend openai-compatible --model gpt-4.1-mini
```
