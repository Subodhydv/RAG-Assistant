"""
Shared dataclasses / pydantic models used across ingestion, retrieval,
and the API layer. Keeping these in one place avoids the "which module
owns this shape" confusion that creeps into RAG codebases.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """One Whisper segment: a timestamped slice of speech."""
    start: float
    end: float
    text: str


class VideoTranscript(BaseModel):
    video_id: str
    source_filename: str
    language: str
    segments: List[TranscriptSegment]


class Chunk(BaseModel):
    """A retrieval unit: several transcript segments merged together,
    small enough to embed well, large enough to keep context intact."""
    chunk_id: str
    video_id: str
    source_filename: str
    start: float
    end: float
    text: str


class RetrievedChunk(Chunk):
    score: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = None
    video_id: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single video, if set."
    )


class Citation(BaseModel):
    video_id: str
    source_filename: str
    start: float
    end: float
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class IngestResponse(BaseModel):
    video_id: str
    num_segments: int
    num_chunks: int
