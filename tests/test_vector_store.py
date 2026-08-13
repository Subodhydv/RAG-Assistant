"""
Unit tests for VectorStore FAISS index management, persistence, and similarity search.
"""
import tempfile
from pathlib import Path
import pytest

from app.models import Chunk
from app.retrieval.vector_store import VectorStore


def test_vector_store_add_and_search(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Monkeypatch embed_texts to avoid loading real SentenceTransformer model in unit tests
        import numpy as np

        def fake_embed(texts):
            # Return dummy 384-dim normalized vectors
            vecs = np.zeros((len(texts), 384), dtype="float32")
            for i in range(len(texts)):
                vecs[i, 0] = 1.0  # Normalized along first dimension
            return vecs

        import app.retrieval.vector_store as store_module
        monkeypatch.setattr(store_module, "embed_texts", fake_embed)

        store = VectorStore(index_dir=tmp_path)
        assert store._index.ntotal == 0

        chunks = [
            Chunk(
                chunk_id="v1_001",
                video_id="v1",
                source_filename="lecture1.mp4",
                start=0.0,
                end=15.0,
                text="Binary search trees maintain sorted elements."
            ),
            Chunk(
                chunk_id="v1_002",
                video_id="v1",
                source_filename="lecture1.mp4",
                start=15.0,
                end=30.0,
                text="Hash tables provide O(1) average time complexity."
            )
        ]

        store.add_chunks(chunks)
        assert store._index.ntotal == 2

        # Verify persistence on disk
        assert (tmp_path / "chunks.faiss").exists()
        assert (tmp_path / "chunks_meta.jsonl").exists()

        # Test search
        results = store.search("binary search", top_k=2)
        assert len(results) == 2
        assert results[0].video_id == "v1"

        # Reload store from persisted files
        store_reloaded = VectorStore(index_dir=tmp_path)
        assert store_reloaded._index.ntotal == 2
