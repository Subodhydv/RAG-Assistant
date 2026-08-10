"""
Chunking is the one stage with zero external dependencies (no Whisper
model, no embedding model, no API key), so it's exhaustively unit
tested here. Run with: pytest tests/test_chunking.py -v
"""
from app.config import settings
from app.ingestion.chunking import chunk_transcript
from app.models import TranscriptSegment, VideoTranscript


def make_transcript(sentences, seg_len=2.0):
    """Build a synthetic transcript where each sentence is one segment."""
    segments = []
    t = 0.0
    for s in sentences:
        segments.append(TranscriptSegment(start=t, end=t + seg_len, text=s))
        t += seg_len
    return VideoTranscript(
        video_id="test_vid",
        source_filename="test.mp3",
        language="en",
        segments=segments,
    )


def test_empty_transcript_yields_no_chunks():
    transcript = make_transcript([])
    assert chunk_transcript(transcript) == []


def test_short_transcript_yields_single_chunk():
    transcript = make_transcript(["Hello there.", "This is a short lecture."])
    chunks = chunk_transcript(transcript)
    assert len(chunks) == 1
    assert "Hello there." in chunks[0].text
    assert chunks[0].video_id == "test_vid"


def test_long_transcript_splits_into_multiple_chunks():
    settings.chunk_char_len = 100
    settings.chunk_overlap = 20
    sentences = [f"This is sentence number {i} in the lecture." for i in range(30)]
    transcript = make_transcript(sentences)

    chunks = chunk_transcript(transcript)

    assert len(chunks) > 1
    # every chunk should respect (roughly) the configured length
    for c in chunks:
        assert len(c.text) <= settings.chunk_char_len + 60  # allow sentence backoff slack


def test_chunks_preserve_monotonic_timestamps():
    settings.chunk_char_len = 120
    settings.chunk_overlap = 20
    sentences = [f"Segment {i} explains a small concept clearly." for i in range(20)]
    transcript = make_transcript(sentences)

    chunks = chunk_transcript(transcript)

    for c in chunks:
        assert c.start <= c.end
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)


def test_chunk_ids_are_unique_and_ordered():
    settings.chunk_char_len = 100
    settings.chunk_overlap = 10
    sentences = [f"Point {i}." for i in range(15)]
    transcript = make_transcript(sentences)

    chunks = chunk_transcript(transcript)
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_overlap_causes_forward_progress_always():
    """Regression guard: a pathological config shouldn't cause an
    infinite loop (this was a real bug class during development)."""
    settings.chunk_char_len = 10
    settings.chunk_overlap = 10_000  # absurdly large overlap
    sentences = [f"Word{i}." for i in range(50)]
    transcript = make_transcript(sentences)

    chunks = chunk_transcript(transcript)  # must terminate
    assert len(chunks) > 0
