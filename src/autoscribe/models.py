from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    radio_transcript = "radio_transcript"
    podcast_transcript = "podcast_transcript"
    interview_transcript = "interview_transcript"
    meeting_transcript = "meeting_transcript"
    phd_thesis = "phd_thesis"
    academic_paper = "academic_paper"
    policy_document = "policy_document"
    legal_or_regulatory_document = "legal_or_regulatory_document"
    book_chapter = "book_chapter"
    essay_or_article = "essay_or_article"
    technical_documentation = "technical_documentation"
    administrative_correspondence = "administrative_correspondence"
    personal_notes = "personal_notes"
    mixed_document = "mixed_document"
    unknown = "unknown"


class DocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    source: str | None = None
    date: str | None = None
    known_document_type: str | None = None
    user_goal: str | None = None
    preferred_output_depth: Literal["brief", "standard", "detailed"] = "standard"
    special_instructions: str | None = None


class Document(BaseModel):
    document_id: str
    text: str
    input_hash: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    sequence: int
    text: str
    char_start: int
    char_end: int


class SecondaryType(BaseModel):
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_types: list[SecondaryType] = Field(default_factory=list)
    reasoning_summary: str = ""
    detected_features: list[str] = Field(default_factory=list)
    recommended_pipeline: str = "generic_document_pipeline"
    verified: bool = False
    warnings: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    chunk_id: str
    char_start: int | None = None
    char_end: int | None = None
    section: str | None = None
    speaker: str | None = None
    page: int | None = None


class AgentInput(BaseModel):
    document_id: str
    document_type: DocumentType
    input_text: str
    chunks: list[Chunk]
    prior_outputs: dict[str, Any] = Field(default_factory=dict)
    user_goal: str | None = None
    instructions: str = ""


class AgentOutput(BaseModel):
    agent_id: str
    output_type: str
    summary: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    source_references: list[SourceReference] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class PipelineDefinition(BaseModel):
    pipeline_id: str
    document_types: list[DocumentType]
    agents: list[str]
    output_title: str
    required_sections: list[str]
    version: str = "0.1.0"


class PipelineRun(BaseModel):
    run_id: str
    document_id: str
    pipeline_id: str
    agent_sequence: list[str]
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    llm_backend: Literal["mock", "openai-compatible"] = "mock"
    model: str = "mock-model"
    classification_model: str | None = None
    summarisation_model: str | None = None
    openai_base_url: str | None = None
    max_context_tokens: int = 32768
    chunk_size_tokens: int = 3000
    chunk_overlap_tokens: int = 250
    temperature: float = 0.2
    output_format: list[Literal["markdown", "json"]] = Field(default_factory=lambda: ["markdown", "json"])
    enable_provenance: bool = True
    enable_user_goal_agent: bool = True
    data_dir: str = "data"
    timeout_seconds: int = 120
    max_retries: int = 2
