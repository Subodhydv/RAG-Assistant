"""
Central configuration for the RAG Teaching Assistant.

All tunables live here so the rest of the codebase never hardcodes
paths, model names, or chunking parameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env", override=True)


@dataclass
class Settings:
    # --- storage paths -----------------------------------------------
    raw_videos_dir: Path = DATA_DIR / "raw_videos"
    audio_dir: Path = DATA_DIR / "audio"
    transcripts_dir: Path = DATA_DIR / "transcripts"
    index_dir: Path = DATA_DIR / "index"

    # --- transcription (Whisper) --------------------------------------
    # "tiny" / "base" / "small" / "medium" / "large-v3"
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "tiny")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # --- chunking -------------------------------------------------------
    chunk_char_len: int = int(os.getenv("CHUNK_CHAR_LEN", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- embeddings -----------------------------------------------------
    # Runs locally, no API key needed. 384-dim, fast, good enough for
    # lecture-transcript style retrieval.
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # --- retrieval --------------------------------------------------------
    top_k: int = int(os.getenv("TOP_K", "5"))

    # --- generation (LLM) -------------------------------------------------
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_answer_tokens: int = int(os.getenv("MAX_ANSWER_TOKENS", "700"))

    @property
    def llm_provider(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("LLM_PROVIDER", "anthropic")

    @property
    def anthropic_api_key(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("ANTHROPIC_API_KEY", "").strip()

    @property
    def openai_api_key(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("OPENAI_API_KEY", "").strip()

    @property
    def gemini_api_key(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def gemini_model(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))

    @property
    def session_secret_key(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        key = os.getenv("SESSION_SECRET_KEY", "").strip()
        if not key:
            # Fallback securely derived from JWT secret or project seed
            key = os.getenv("JWT_SECRET_KEY", "rag-assistant-secure-hmac-seed-2026").strip()
        return key

    @property
    def auth_enabled(self) -> bool:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")

    @property
    def auth_username(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("AUTH_USERNAME", "admin").strip()

    @property
    def auth_password(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("AUTH_PASSWORD", "admin123").strip()

    @property
    def jwt_secret_key(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("JWT_SECRET_KEY", "rag-assistant-secret-jwt-key-2026").strip()

    @property
    def anthropic_model(self) -> str:
        load_dotenv(BASE_DIR / ".env", override=True)
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()

    def ensure_dirs(self) -> None:
        for d in (self.raw_videos_dir, self.audio_dir, self.transcripts_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionPaths:
    raw_videos_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    index_dir: Path

    def ensure_dirs(self) -> None:
        for d in (self.raw_videos_dir, self.audio_dir, self.transcripts_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_session_paths(session_id: str | None = None) -> SessionPaths:
    if not session_id or session_id.strip() in ("", "default"):
        sp = SessionPaths(
            raw_videos_dir=DATA_DIR / "raw_videos",
            audio_dir=DATA_DIR / "audio",
            transcripts_dir=DATA_DIR / "transcripts",
            index_dir=DATA_DIR / "index",
        )
    else:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))[:32] or "default"
        sess_dir = DATA_DIR / "sessions" / safe_id
        sp = SessionPaths(
            raw_videos_dir=sess_dir / "raw_videos",
            audio_dir=sess_dir / "audio",
            transcripts_dir=sess_dir / "transcripts",
            index_dir=sess_dir / "index",
        )
    sp.ensure_dirs()
    return sp


settings = Settings()
settings.ensure_dirs()
