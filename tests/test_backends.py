from __future__ import annotations

from pathlib import Path

from docagent.backends import GENERIC_JSON_GRAMMAR, OpenAICompatibleBackend, _extract_message_content
from docagent.config import load_config
from docagent.models import AppConfig


def test_grammar_mode_sends_grammar_only():
    request = _complete_request(AppConfig(json_decoding_mode="grammar"))

    assert request["extra_body"]["grammar"] == GENERIC_JSON_GRAMMAR
    assert "response_format" not in request


def test_response_format_mode_sends_response_format_only():
    request = _complete_request(AppConfig(json_decoding_mode="response_format"))

    assert request["response_format"] == {"type": "json_object"}
    assert "extra_body" not in request


def test_both_mode_sends_grammar_and_response_format():
    request = _complete_request(AppConfig(json_decoding_mode="both", json_grammar="root ::= \"{}\""))

    assert request["extra_body"]["grammar"] == 'root ::= "{}"'
    assert request["response_format"] == {"type": "json_object"}


def test_off_mode_sends_no_json_decoding_controls():
    request = _complete_request(AppConfig(json_decoding_mode="off"))

    assert "extra_body" not in request
    assert "response_format" not in request


def test_no_schema_sends_no_json_decoding_controls():
    request = _complete_request(AppConfig(json_decoding_mode="grammar"), schema=None)

    assert "extra_body" not in request
    assert "response_format" not in request


def test_json_decoding_mode_loads_from_environment(monkeypatch):
    monkeypatch.setenv("DOCAGENT_JSON_DECODING_MODE", "both")
    monkeypatch.setenv("DOCAGENT_JSON_GRAMMAR", 'root ::= "{}"')

    config = load_config()

    assert config.json_decoding_mode == "both"
    assert config.json_grammar == 'root ::= "{}"'


def test_json_decoding_mode_loads_from_config_file(tmp_path: Path):
    config_path = tmp_path / "docagent.json"
    config_path.write_text('{"json_decoding_mode": "response_format"}', encoding="utf-8")

    config = load_config(config_path)

    assert config.json_decoding_mode == "response_format"


def test_extract_message_content_falls_back_to_reasoning_content():
    response = FakeResponse(content="", reasoning_content="{\"ok\": true}")

    assert _extract_message_content(response) == "{\"ok\": true}"


def _complete_request(config: AppConfig, schema: dict[str, object] | None = {}) -> dict[str, object]:
    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.config = config
    backend.client = FakeClient()

    backend.complete(
        [{"role": "user", "content": "Return JSON."}],
        schema=schema,
        temperature=0.0,
        model="local-model",
        timeout=3,
    )

    return backend.client.chat.completions.last_request


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeCompletions:
    def __init__(self):
        self.last_request: dict[str, object] = {}

    def create(self, **request):
        self.last_request = request
        return FakeResponse()


class FakeResponse:
    def __init__(self, content: str = "{\"ok\": true}", reasoning_content: str | None = None):
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "message": type(
                        "Message",
                        (),
                        {
                            "content": content,
                            "reasoning_content": reasoning_content,
                        },
                    )()
                },
            )()
        ]
