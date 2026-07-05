from __future__ import annotations


class DocAgentError(Exception):
    code = "docagent_error"

    def __init__(self, message: str, suggested_actions: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.suggested_actions = suggested_actions or []

    def to_dict(self) -> dict[str, object]:
        return {
            "error": self.code,
            "message": self.message,
            "suggested_actions": self.suggested_actions,
        }


class InputError(DocAgentError):
    code = "input_error"


class ClassificationError(DocAgentError):
    code = "classification_error"


class BackendError(DocAgentError):
    code = "llm_backend_error"


class PipelineError(DocAgentError):
    code = "pipeline_error"
