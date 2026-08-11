"""
FastAPI entrypoint for the RAG Teaching Assistant.

Endpoints:
  POST /ingest   - upload a lecture video, runs transcription + chunking
                   + embedding, returns how many chunks were indexed
  POST /ask      - ask a question, get an answer grounded in the
                   indexed lecture content, with timestamped citations
  GET  /health   - liveness check

Run with:
  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings, BASE_DIR, get_session_paths
from app.generation.answer import generate_answer
from app.generation.quiz import generate_quiz_questions
from app.ingestion.chunking import chunk_transcript
from app.ingestion.transcribe import ingest_video, load_transcript
from app.models import AskRequest, AskResponse, Citation, IngestResponse, QuizRequest, QuizResponse
from app.retrieval.vector_store import get_store


def get_session_id(x_session_id: Optional[str] = Header(None, alias="X-Session-ID")) -> str:
    if not x_session_id or not x_session_id.strip():
        return "default"
    return "".join(c for c in x_session_id if c.isalnum() or c in ("-", "_"))[:32]


app = FastAPI(
    title="RAG Teaching Assistant",
    description="Ask questions about lecture videos; answers are grounded "
    "in retrieved transcript excerpts with timestamp citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
settings.ensure_dirs()
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "whisper_model": settings.whisper_model_size,
        "embedding_model": settings.embedding_model_name,
        "session_isolated": True,
    }


@app.get("/videos")
def list_videos(session_id: str = Depends(get_session_id)) -> List[Dict[str, Any]]:
    """List all ingested videos for the current private session."""
    videos = []
    store = get_store(session_id)
    sp = get_session_paths(session_id)

    if not sp.transcripts_dir.exists():
        return []

    for t_file in sorted(sp.transcripts_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            transcript = load_transcript(t_file.stem, session_id=session_id)
            chunks_count = sum(1 for c in store._meta if c.video_id == transcript.video_id)
            
            # Find matching raw video file if it exists
            video_files = list(sp.raw_videos_dir.glob(f"{transcript.video_id}.*"))
            media_url = f"/media/{video_files[0].name}" if video_files else None

            videos.append({
                "video_id": transcript.video_id,
                "source_filename": transcript.source_filename,
                "language": transcript.language,
                "num_segments": len(transcript.segments),
                "num_chunks": chunks_count,
                "media_url": media_url,
            })
        except Exception:
            continue
    return videos


@app.get("/videos/{video_id}/transcript")
def get_video_transcript(video_id: str, session_id: str = Depends(get_session_id)):
    """Retrieve full transcript segments for a video in the current private session."""
    try:
        transcript = load_transcript(video_id, session_id=session_id)
        return transcript
    except FileNotFoundError:
        raise HTTPException(404, f"Transcript for video '{video_id}' not found")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), session_id: str = Depends(get_session_id)):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    sp = get_session_paths(session_id)
    video_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename).suffix or ".mp4"
    dest = sp.raw_videos_dir / f"{video_id}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        transcript = ingest_video(dest, video_id=video_id, session_id=session_id)
    except RuntimeError as e:
        raise HTTPException(500, f"Transcription failed: {e}") from e

    chunks = chunk_transcript(transcript)
    get_store(session_id).add_chunks(chunks)

    return IngestResponse(
        video_id=transcript.video_id,
        num_segments=len(transcript.segments),
        num_chunks=len(chunks),
    )


from datetime import datetime
from fastapi import Response


class ExportNotesRequest(BaseModel):
    title: Optional[str] = "Lecture Study Guide"
    messages: List[Dict[str, Any]]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, session_id: str = Depends(get_session_id)):
    store = get_store(session_id)
    top_k = req.top_k or settings.top_k
    retrieved = store.search(req.question, top_k=top_k, video_id=req.video_id)

    answer = generate_answer(req.question, retrieved)

    citations = [
        Citation(
            video_id=c.video_id,
            source_filename=c.source_filename,
            start=c.start,
            end=c.end,
            score=c.score,
        )
        for c in retrieved
    ]
    return AskResponse(answer=answer, citations=citations)


@app.post("/export-notes")
def export_notes(req: ExportNotesRequest):
    lines = [
        f"# {req.title or 'Lecture Study Guide'}",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} via RAG Teaching Assistant*\n",
        "---\n",
    ]
    for i, msg in enumerate(req.messages, start=1):
        q = msg.get("question", "")
        a = msg.get("answer", "")
        citations = msg.get("citations", [])

        lines.append(f"### Q{i}: {q}\n")
        lines.append(f"{a}\n")

        if citations:
            lines.append("**Timestamp Citations**:")
            for c in citations:
                fn = c.get("source_filename", "video")
                st = c.get("start", 0)
                en = c.get("end", 0)
                m1, s1 = int(st // 60), int(st % 60)
                m2, s2 = int(en // 60), int(en % 60)
                lines.append(f"- `{fn}` @ {m1}:{s1:02d} - {m2}:{s2:02d}")
            lines.append("")

        lines.append("---\n")

    content = "\n".join(lines)
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="lecture_study_notes.md"'},
    )


@app.post("/quiz", response_model=QuizResponse)
def generate_quiz(req: QuizRequest, session_id: str = Depends(get_session_id)):
    paths = get_session_paths(session_id)
    transcripts_dir = paths.transcripts_dir

    selected_transcript = None

    try:
        if req.video_id:
            selected_transcript = load_transcript(req.video_id, session_id=session_id)
        else:
            # Pick the first transcript in session transcripts dir, or default
            files = list(transcripts_dir.glob("*.json"))
            if files:
                video_id = files[0].stem
                selected_transcript = load_transcript(video_id, session_id=session_id)
            else:
                global_files = list(settings.transcripts_dir.glob("*.json"))
                if global_files:
                    video_id = global_files[0].stem
                    selected_transcript = load_transcript(video_id, session_id=session_id)
    except Exception:
        selected_transcript = None

    if not selected_transcript:
        raise HTTPException(
            status_code=404,
            detail="No ingested lecture video transcripts found in your session. Please upload a video in the 'Ingest Lecture' tab first!"
        )

    num_q = req.num_questions or 5
    return generate_quiz_questions(selected_transcript, num_questions=num_q)



# Mount media files for video playback
app.mount("/media", StaticFiles(directory=str(settings.raw_videos_dir)), name="media")

# Mount Web UI static assets at root
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


