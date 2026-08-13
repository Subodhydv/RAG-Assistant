"""
FastAPI entrypoint for the RAG Teaching Assistant.

Production Features:
  - HMAC Signed Session Security
  - Async Background Ingestion Pipeline (Non-blocking /ingest & progress polling)
  - YouTube URL Ingestion via yt-dlp
  - Rate Limiting via SlowAPI
  - Real-Time Streaming Answers via Server-Sent Events (SSE)
  - Multi-Turn Conversation History
  - Video & Session Cleanup Endpoints
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Header, Response, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse

from app.config import settings, BASE_DIR, DATA_DIR, get_session_paths
from app.auth import get_current_session, verify_session_token, get_current_user, create_access_token, verify_credentials
from app.generation.answer import generate_answer, generate_answer_stream
from app.generation.quiz import generate_quiz_questions
from app.ingestion.chunking import chunk_transcript
from app.ingestion.transcribe import ingest_video, load_transcript
from app.models import AskRequest, AskResponse, Citation, IngestResponse, QuizRequest, QuizResponse
from app.retrieval.vector_store import get_store

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RAG Teaching Assistant",
    description="Ask questions about lecture videos; answers are grounded "
    "in retrieved transcript excerpts with timestamp citations.",
    version="2.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# In-memory status store for background ingestion tasks
ingest_tasks: Dict[str, Dict[str, Any]] = {}


class LoginRequest(BaseModel):
    username: str
    password: str


class YouTubeIngestRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL to transcribe and index")


class ExportNotesRequest(BaseModel):
    title: Optional[str] = "Lecture Study Guide"
    messages: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "whisper_model": settings.whisper_model_size,
        "embedding_model": settings.embedding_model_name,
        "session_isolated": True,
        "async_ingest_supported": True,
        "youtube_ingest_supported": True,
    }


@app.post("/login")
def login(req: LoginRequest):
    if not verify_credentials(req.username, req.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": req.username, "role": "admin"})
    return {"access_token": token, "token_type": "bearer", "username": req.username}


@app.get("/videos")
def list_videos(session_id: str = Depends(get_current_session)) -> List[Dict[str, Any]]:
    """List all ingested videos for the current private session."""
    videos = []
    store = get_store(session_id)
    sp = get_session_paths(session_id)

    if not sp.transcripts_dir.exists():
        return []

    for t_file in sorted(sp.transcripts_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            transcript = load_transcript(t_file.stem, session_id=session_id)
            if not transcript:
                continue
            chunks_count = sum(1 for c in store._meta if c.video_id == transcript.video_id)
            media_url = f"/videos/{transcript.video_id}/stream"

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


@app.get("/videos/{video_id}/stream")
def stream_video(video_id: str, session_id: str = Depends(get_current_session)):
    """Stream media file for authorized session video playback."""
    sp = get_session_paths(session_id)
    files = list(sp.raw_videos_dir.glob(f"{video_id}.*"))
    if not files:
        files = list(settings.raw_videos_dir.glob(f"{video_id}.*"))
    if not files:
        for sess in (DATA_DIR / "sessions").glob("*"):
            match = list((sess / "raw_videos").glob(f"{video_id}.*"))
            if match:
                files = match
                break
    if not files or not files[0].exists():
        raise HTTPException(404, f"Video stream for ID '{video_id}' not found")
    return FileResponse(files[0])


@app.get("/videos/{video_id}/transcript")
def get_video_transcript(video_id: str, session_id: str = Depends(get_current_session)):
    """Retrieve full transcript segments for a video in the current private session."""
    transcript = load_transcript(video_id, session_id=session_id)
    if not transcript:
        raise HTTPException(404, f"Transcript for video '{video_id}' not found")
    return transcript


@app.delete("/videos/{video_id}")
def delete_video(video_id: str, session_id: str = Depends(get_current_session)):
    """Delete a video and its indexed vector chunks from the current session."""
    sp = get_session_paths(session_id)
    transcript_path = sp.transcripts_dir / f"{video_id}.json"

    if transcript_path.exists():
        transcript_path.unlink()

    for media_file in sp.raw_videos_dir.glob(f"{video_id}.*"):
        media_file.unlink()

    # Remove chunks from vector store
    store = get_store(session_id)
    store._meta = [c for c in store._meta if c.video_id != video_id]
    store._persist()

    return {"status": "deleted", "video_id": video_id}


def _run_background_ingest(task_id: str, dest_path: Path, video_id: str, session_id: str):
    """Background task worker for video transcription, chunking, and FAISS indexing."""
    try:
        ingest_tasks[task_id]["status"] = "transcribing"
        ingest_tasks[task_id]["message"] = "Extracting audio & transcribing speech with Whisper..."
        transcript = ingest_video(dest_path, video_id=video_id, session_id=session_id)

        ingest_tasks[task_id]["status"] = "indexing"
        ingest_tasks[task_id]["message"] = "Generating embeddings & indexing into FAISS vector store..."
        chunks = chunk_transcript(transcript)
        get_store(session_id).add_chunks(chunks)

        ingest_tasks[task_id]["status"] = "completed"
        ingest_tasks[task_id]["message"] = "Ingestion complete!"
        ingest_tasks[task_id]["result"] = {
            "video_id": transcript.video_id,
            "num_segments": len(transcript.segments),
            "num_chunks": len(chunks),
        }
    except Exception as e:
        ingest_tasks[task_id]["status"] = "failed"
        ingest_tasks[task_id]["message"] = f"Ingestion failed: {e}"


@app.post("/ingest")
@limiter.limit("10/hour")
def ingest(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Depends(get_current_session)
):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    allowed_exts = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".aac")
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_exts)}")

    sp = get_session_paths(session_id)
    video_id = uuid.uuid4().hex[:12]
    dest = sp.raw_videos_dir / f"{video_id}{ext}"

    try:
        file.file.seek(0)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # File size validation
        size_mb = dest.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, f"File size ({size_mb:.1f}MB) exceeds limit of {settings.max_upload_size_mb}MB")

        task_id = f"task_{uuid.uuid4().hex[:8]}"
        ingest_tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "message": "Queued for processing...",
            "created_at": datetime.now().isoformat(),
        }

        # Launch background task for async ingestion
        background_tasks.add_task(_run_background_ingest, task_id, dest, video_id, session_id)

        return {
            "task_id": task_id,
            "status": "queued",
            "video_id": video_id,
            "message": "File uploaded successfully. Background ingestion task started."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}") from e


@app.post("/ingest-youtube")
@limiter.limit("10/hour")
def ingest_youtube(
    request: Request,
    bg_req: YouTubeIngestRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(get_current_session)
):
    """Ingest lecture directly from YouTube URL using yt-dlp."""
    url = bg_req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Invalid URL format")

    sp = get_session_paths(session_id)
    video_id = uuid.uuid4().hex[:12]
    audio_dest = sp.audio_dir / f"{video_id}.mp3"

    task_id = f"yt_task_{uuid.uuid4().hex[:8]}"
    ingest_tasks[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "message": "Queued YouTube audio extraction...",
        "created_at": datetime.now().isoformat(),
    }

    def _run_youtube_ingest():
        try:
            ingest_tasks[task_id]["status"] = "downloading"
            ingest_tasks[task_id]["message"] = "Downloading audio from YouTube using yt-dlp..."

            out_template = sp.audio_dir / f"{video_id}.%(ext)s"
            cmd = [
                "yt-dlp", "--extract-audio", "--audio-format", "mp3",
                "--no-playlist", "-o", str(out_template), url
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)

            generated_files = list(sp.audio_dir.glob(f"{video_id}.*"))
            if res.returncode != 0 or not generated_files:
                raise RuntimeError(f"yt-dlp failed: {res.stderr[-500:]}")

            audio_file = generated_files[0]
            _run_background_ingest(task_id, audio_file, video_id, session_id)
        except Exception as e:
            ingest_tasks[task_id]["status"] = "failed"
            ingest_tasks[task_id]["message"] = f"YouTube ingestion failed: {e}"

    background_tasks.add_task(_run_youtube_ingest)
    return {"task_id": task_id, "status": "queued", "video_id": video_id}


@app.get("/ingest/{task_id}/status")
def get_ingest_status(task_id: str):
    """Poll progress status of an async video ingestion task."""
    if task_id not in ingest_tasks:
        raise HTTPException(404, f"Ingestion task '{task_id}' not found")
    return ingest_tasks[task_id]


@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
def ask(request: Request, req: AskRequest, session_id: str = Depends(get_current_session)):
    store = get_store(session_id)
    top_k = req.top_k or settings.top_k
    retrieved = store.search(req.question, top_k=top_k, video_id=req.video_id)

    answer = generate_answer(req.question, retrieved, history=req.conversation_history)

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


@app.post("/ask/stream")
@limiter.limit("30/minute")
def ask_stream(request: Request, req: AskRequest, session_id: str = Depends(get_current_session)):
    """Server-Sent Events (SSE) endpoint for real-time streaming answer generation."""
    store = get_store(session_id)
    top_k = req.top_k or settings.top_k
    retrieved = store.search(req.question, top_k=top_k, video_id=req.video_id)

    def event_generator():
        for chunk in generate_answer_stream(req.question, retrieved, history=req.conversation_history):
            yield {"event": "message", "data": chunk}

    return EventSourceResponse(event_generator())


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
@limiter.limit("10/minute")
def generate_quiz(request: Request, req: QuizRequest, session_id: str = Depends(get_current_session)):
    paths = get_session_paths(session_id)
    transcripts_dir = paths.transcripts_dir

    selected_transcript = None

    try:
        if req.video_id:
            selected_transcript = load_transcript(req.video_id, session_id=session_id)
        else:
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
