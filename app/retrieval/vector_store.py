"""
Stage 3: embed chunks and store them in a FAISS index for fast
nearest-neighbour search.

Design choices worth being able to explain in an interview:
  - sentence-transformers/all-MiniLM-L6-v2 runs locally (no per-call
    API cost, no rate limits during bulk ingestion of many lecture
    hours) and is fast enough for CPU.
  - FAISS IndexFlatIP (inner product) over L2-normalized vectors gives
    exact cosine-similarity search. For this project's scale
    (hundreds-thousands of chunks) exact search is fast enough that
    an approximate index (IVF/HNSW) would be premature optimization.
  - Index + metadata are persisted separately: FAISS only stores
    vectors, so a parallel JSON-lines file maps row -> Chunk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import settings, get_session_paths
from app.models import Chunk, RetrievedChunk

_embedder = None
_stores: dict[str, VectorStore] = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(settings.embedding_model_name)
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


class VectorStore:
    """Thin wrapper around a FAISS flat index + a sidecar metadata file."""

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or settings.index_dir
        self.index_path = self.index_dir / "chunks.faiss"
        self.meta_path = self.index_dir / "chunks_meta.jsonl"
        self._index = None
        self._meta: List[Chunk] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        import faiss

        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._meta = [
                Chunk.model_validate_json(line)
                for line in self.meta_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            self._index = faiss.IndexFlatIP(settings.embedding_dim)
            self._meta = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        vecs = embed_texts([c.text for c in chunks])
        self._index.add(vecs)
        self._meta.extend(chunks)
        self._persist()

    def _persist(self) -> None:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for c in self._meta:
                f.write(c.model_dump_json() + "\n")

    def search(
        self, query: str, top_k: int = 5, video_id: Optional[str] = None
    ) -> List[RetrievedChunk]:
        if self._index.ntotal == 0:
            return []

        q_vec = embed_texts([query])
        # over-fetch when filtering by video_id, since FAISS itself
        # doesn't know about that filter
        fetch_k = top_k * 5 if video_id else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, idxs = self._index.search(q_vec, fetch_k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = self._meta[idx]
            if video_id and chunk.video_id != video_id:
                continue
            results.append(RetrievedChunk(**chunk.model_dump(), score=float(score)))
            if len(results) >= top_k:
                break
        return results


def get_store(session_id: str | None = None) -> VectorStore:
    global _stores
    key = session_id or "default"
    if key not in _stores:
        sp = get_session_paths(session_id)
        _stores[key] = VectorStore(index_dir=sp.index_dir)
    return _stores[key]
