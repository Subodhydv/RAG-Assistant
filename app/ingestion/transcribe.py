"""
Stage 1 of the pipeline: video -> mp3 -> timestamped transcript.

Mirrors what the course does ("Converting Videos to mp3 for Whisper",
"Video to Text with Timestamps Using Whisper") but uses faster-whisper
instead of openai-whisper, since it's CTranslate2-backed and runs
noticeably faster on CPU-only machines — worth mentioning in an
interview as a deliberate optimization over the "default" approach.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

from app.config import settings, get_session_paths
from app.models import TranscriptSegment, VideoTranscript

_model = None  # lazy-loaded singleton, avoids reloading weights per call


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def video_to_audio(video_path: Path, out_dir: Path | None = None) -> Path:
    """Extract mono 16kHz mp3 audio from a video file using ffmpeg.

    If ffmpeg is unavailable or fails, returns original file path for Whisper.
    """
    out_dir = out_dir or settings.audio_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / f"{video_path.stem}.mp3"

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path
    except Exception:
        pass

    return video_path


def transcribe_audio(audio_path: Path, video_id: str | None = None) -> VideoTranscript:
    """Run Whisper on an audio file and return a structured transcript
    with per-segment timestamps preserved (needed later so citations
    can point a learner back to the exact moment in the video)."""
    video_id = video_id or uuid.uuid4().hex[:12]
    model = _get_model()

    segments_iter, info = model.transcribe(str(audio_path), vad_filter=True)
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
        if s.text.strip()
    ]

    return VideoTranscript(
        video_id=video_id,
        source_filename=audio_path.name,
        language=info.language,
        segments=segments,
    )


import hashlib


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 content hash of a video/audio file for deduplication."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_existing_transcript_by_hash(file_hash: str, session_id: str | None = None) -> VideoTranscript | None:
    """Check if a transcript with the same content SHA-256 hash already exists."""
    sp = get_session_paths(session_id)
    search_dirs = [sp.transcripts_dir, settings.transcripts_dir]
    for d in search_dirs:
        if not d.exists():
            continue
        for json_file in d.glob("*.json"):
            try:
                t = VideoTranscript.model_validate_json(json_file.read_text(encoding="utf-8"))
                if getattr(t, "content_hash", None) == file_hash:
                    return t
            except Exception:
                continue
    return None


def save_transcript(transcript: VideoTranscript, session_id: str | None = None) -> Path:
    sp = get_session_paths(session_id)
    out_path = sp.transcripts_dir / f"{transcript.video_id}.json"
    out_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_transcript(video_id: str, session_id: str | None = None) -> VideoTranscript | None:
    sp = get_session_paths(session_id)
    path = sp.transcripts_dir / f"{video_id}.json"
    if not path.exists():
        default_path = settings.transcripts_dir / f"{video_id}.json"
        if default_path.exists():
            path = default_path
    if not path.exists():
        return None
    return VideoTranscript.model_validate_json(path.read_text(encoding="utf-8"))


def ingest_video(
    video_path: Path, video_id: str | None = None, session_id: str | None = None
) -> VideoTranscript:
    """End-to-end: video file on disk -> saved transcript JSON with SHA-256 deduplication."""
    sp = get_session_paths(session_id)

    # Compute content hash to prevent redundant Whisper re-transcription
    file_hash = compute_file_sha256(video_path)
    existing = find_existing_transcript_by_hash(file_hash, session_id=session_id)
    if existing:
        return existing

    audio_path = video_to_audio(video_path, out_dir=sp.audio_dir)
    transcript = transcribe_audio(audio_path, video_id=video_id)
    transcript.content_hash = file_hash
    save_transcript(transcript, session_id=session_id)
    return transcript
