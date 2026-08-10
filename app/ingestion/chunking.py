"""
Stage 2: turn a list of short Whisper segments (often just a few words
each) into merged chunks that are good retrieval units.

Naive approach (chunk every N segments) breaks sentences mid-thought.
Instead this walks the segment stream and accumulates text until it
hits chunk_char_len, backing off to the nearest sentence boundary,
then starts the next chunk `chunk_overlap` characters earlier so
retrieval doesn't lose context that straddles a chunk edge.

This is pure text logic with no external dependencies, so it's the
easiest part of the pipeline to unit test exhaustively.
"""
from __future__ import annotations

import re
from typing import List

from app.config import settings
from app.models import Chunk, TranscriptSegment, VideoTranscript

_SENTENCE_END = re.compile(r"[.!?]\s")


def _find_backoff_point(text: str, target_len: int) -> int:
    """Given text longer than target_len, find the last sentence-ending
    punctuation at or before target_len. Falls back to target_len itself
    (a hard character cut) if no sentence boundary is found."""
    window = text[:target_len]
    matches = list(_SENTENCE_END.finditer(window))
    if matches:
        return matches[-1].end()
    return target_len


def chunk_transcript(transcript: VideoTranscript) -> List[Chunk]:
    segments = transcript.segments
    if not segments:
        return []

    chunks: List[Chunk] = []
    chunk_idx = 0
    i = 0

    while i < len(segments):
        buf_text: List[str] = []
        buf_segments: List[TranscriptSegment] = []
        cur_len = 0
        start_i = i

        # accumulate segments until we hit the target length
        while i < len(segments) and cur_len < settings.chunk_char_len:
            seg = segments[i]
            buf_text.append(seg.text)
            buf_segments.append(seg)
            cur_len += len(seg.text) + 1
            i += 1

        text = " ".join(buf_text).strip()

        if len(text) > settings.chunk_char_len:
            cut = _find_backoff_point(text, settings.chunk_char_len)
            text = text[:cut].strip()

        chunk = Chunk(
            chunk_id=f"{transcript.video_id}_{chunk_idx:04d}",
            video_id=transcript.video_id,
            source_filename=transcript.source_filename,
            start=buf_segments[0].start,
            end=buf_segments[-1].end,
            text=text,
        )
        chunks.append(chunk)
        chunk_idx += 1

        # If we've consumed every segment, there's no more content ahead
        # to justify an overlapping tail chunk — stop here. Otherwise,
        # step back by roughly chunk_overlap characters worth of segments
        # so the next chunk re-includes trailing context.
        if i >= len(segments):
            break

        overlap_chars = 0
        back = i - 1
        while back > start_i and overlap_chars < settings.chunk_overlap:
            overlap_chars += len(segments[back].text)
            back -= 1
        i = max(back + 1, start_i + 1)  # always make forward progress

    return chunks
