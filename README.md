# 🎓 RAG Teaching Assistant

Ask questions about lecture videos and get answers grounded in the actual transcript, with timestamp citations back to the source video.

- 🌐 **Live Cloud Application**: [https://rag-teaching-assistant-njhd.onrender.com](https://rag-teaching-assistant-njhd.onrender.com)
- 🐙 **GitHub Repository**: [https://github.com/Subodhydv/RAG-Assistant](https://github.com/Subodhydv/RAG-Assistant)

## Pipeline

```
video file
   |  ffmpeg (extract mono 16kHz mp3)
   v
audio file
   |  faster-whisper (transcription + timestamps)
   v
timestamped transcript (JSON)
   |  chunking (merge segments -> ~800-char chunks, sentence-aware, overlapping)
   v
chunks
   |  sentence-transformers (all-MiniLM-L6-v2) -> 384-dim embeddings
   v
FAISS index (IndexFlatIP, cosine similarity via normalized vectors)
   |
   |  query time: embed question -> top-k nearest chunks
   v
retrieved chunks --> prompt --> Claude / GPT --> grounded answer + citations
```

## Architecture Highlights (Interview Talkthroughs)

- **Async Background Ingestion Pipeline**: Ingestion (`POST /ingest` & `POST /ingest-youtube`) executes asynchronously via FastAPI `BackgroundTasks`. Clients poll `GET /ingest/{task_id}/status` for real-time progress (`transcribing -> indexing -> completed`), avoiding cloud HTTP request timeouts on long lecture videos.
- **SHA-256 Content Hash Deduplication**: Files are hashed before processing. Re-uploading an existing lecture instantly reuses indexed vector chunks without running redundant Whisper transcription.
- **YouTube URL Ingestion**: Built-in `yt-dlp` audio extraction pipeline (`POST /ingest-youtube`) allows instant indexing directly from public YouTube lecture URLs.
- **HMAC-Signed Session Isolation**: User workspaces are protected with cryptographic HMAC-SHA256 session tokens, preventing cross-session unauthorized data or video stream access.
- **`faster-whisper` over `openai-whisper`**: CTranslate2 backend achieves 4x faster CPU execution with lower RAM usage.
- **Sentence-Aware Chunking with Overlap**: Prevents sentence slicing mid-thought and preserves boundary context.
- **Local Embeddings (`sentence-transformers`)**: Zero per-chunk embedding API costs or rate limits during bulk lecture ingestion.
- **Single-Worker FAISS Persistence Note**: `VectorStore` persists indices to disk per session. In multi-worker production deployments (e.g. multiple Uvicorn workers), a centralized vector database (e.g. Qdrant / Pinecone / pgvector) or distributed file lock should replace file-backed FAISS to prevent concurrent process write races.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# ffmpeg is required for audio extraction
sudo apt-get install ffmpeg   # or: brew install ffmpeg

cp .env.example .env
# fill in ANTHROPIC_API_KEY (or OPENAI_API_KEY if LLM_PROVIDER=openai)
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API docs at `http://localhost:8000/docs`.

### Ingest a lecture video

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@lecture1.mp4"
```

Returns `{"video_id": "...", "num_segments": N, "num_chunks": M}`.
The transcript is cached to `data/transcripts/<video_id>.json`, so
re-running the pipeline never re-transcribes from scratch.

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the time complexity of a hash map lookup?"}'
```

Returns an answer plus a list of citations (video id, timestamp range,
similarity score) so a learner can jump straight to the relevant moment
in the source video.

## Tests

```bash
pytest tests/ -v
```

- `test_chunking.py` — pure-logic tests for the sentence-aware chunker
  (no models/network needed). Includes a regression test for an
  infinite-loop edge case found and fixed during development
  (a chunk that exactly consumes the last segment must not trigger a
  spurious overlapping tail chunk).
- `test_api.py` — exercises the `/health` and `/ask` FastAPI routes with
  the vector store and LLM call monkeypatched out, so it runs in
  milliseconds with no API keys or model downloads required.

## Extending this for CP Insight / the internship platform

The same pipeline generalizes beyond lecture videos: swap the ingestion
stage (`app/ingestion/transcribe.py`) for one that pulls Codeforces/
LeetCode editorials and problem statements as source text (skip the
Whisper step entirely, chunking/embedding/retrieval/generation stay
identical), and the result is a "explain this problem" / "find similar
past problems" AI mentor grounded in real problem text rather than
free-form LLM recall.
