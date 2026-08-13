"""
Unit tests for SHA-256 deduplication and transcript loading.
"""
import tempfile
from pathlib import Path
import pytest

from app.ingestion.transcribe import compute_file_sha256, find_existing_transcript_by_hash, save_transcript
from app.models import VideoTranscript, TranscriptSegment


def test_compute_file_sha256():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp.write("Sample video audio content for hashing test")
        tmp_path = Path(tmp.name)

    try:
        h1 = compute_file_sha256(tmp_path)
        h2 = compute_file_sha256(tmp_path)
        assert len(h1) == 64  # SHA-256 hex string length
        assert h1 == h2
    finally:
        tmp_path.unlink(missing_ok=True)


def test_find_existing_transcript_by_hash(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        fake_sp = type("SessionPaths", (), {
            "transcripts_dir": tmp_path,
            "raw_videos_dir": tmp_path,
            "audio_dir": tmp_path,
            "index_dir": tmp_path
        })

        import app.ingestion.transcribe as transcribe_mod
        monkeypatch.setattr(transcribe_mod, "get_session_paths", lambda sess=None: fake_sp)

        transcript = VideoTranscript(
            video_id="vid_hash_1",
            source_filename="lecture.mp4",
            language="en",
            segments=[TranscriptSegment(start=0, end=10, text="Hello world")],
            content_hash="abc123hashvalue"
        )
        save_transcript(transcript, session_id="test_sess")

        found = find_existing_transcript_by_hash("abc123hashvalue", session_id="test_sess")
        assert found is not None
        assert found.video_id == "vid_hash_1"

        not_found = find_existing_transcript_by_hash("nonexistenthash", session_id="test_sess")
        assert not_found is None
