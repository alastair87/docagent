# DocAgent

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

## Forced JSON decoding

Structured classifier and agent calls can ask OpenAI-compatible servers to constrain
generation to JSON. The default is grammar mode, which is useful for llama.cpp-style
servers:

```bash
export OPENAI_API_KEY=local-not-used
export DOCAGENT_BACKEND=openai-compatible
export DOCAGENT_BASE_URL=http://127.0.0.1:8033/v1
export DOCAGENT_MODEL=/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
export DOCAGENT_JSON_DECODING_MODE=grammar

docagent classify samples/policy.txt
```

Other modes are available for different servers:

```bash
export DOCAGENT_JSON_DECODING_MODE=response_format  # OpenAI-style JSON object mode
export DOCAGENT_JSON_DECODING_MODE=both             # send grammar and response_format
export DOCAGENT_JSON_DECODING_MODE=off              # prompt-only JSON guidance
```
