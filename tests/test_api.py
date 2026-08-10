"""
Exercises the /ask endpoint end-to-end without touching FAISS,
sentence-transformers, or any LLM API — those are monkeypatched with
lightweight stubs so this test runs in milliseconds and needs no
network access or API keys. Run with: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

from app.models import RetrievedChunk


@pytest.fixture
def client(monkeypatch):
    import app.main as main_module

    fake_chunk = RetrievedChunk(
        chunk_id="vid1_0000",
        video_id="vid1",
        source_filename="lecture1.mp3",
        start=12.5,
        end=45.0,
        text="A hash map gives average O(1) lookup by trading space for time.",
        score=0.87,
    )

    class FakeStore:
        def search(self, question, top_k=5, video_id=None):
            return [fake_chunk]

    monkeypatch.setattr(main_module, "get_store", lambda session_id=None: FakeStore())
    monkeypatch.setattr(
        main_module,
        "generate_answer",
        lambda question, chunks: f"Stubbed answer using {len(chunks)} excerpt(s).",
    )

    return TestClient(main_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["session_isolated"] is True


def test_ask_returns_answer_and_citations(client):
    headers = {"X-Session-ID": "sess_test123"}
    resp = client.post("/ask", json={"question": "How does a hash map work?"}, headers=headers)
    assert resp.status_code == 200

    body = resp.json()
    assert "Stubbed answer" in body["answer"]
    assert len(body["citations"]) == 1

    citation = body["citations"][0]
    assert citation["video_id"] == "vid1"
    assert citation["start"] == 12.5
    assert citation["score"] == 0.87


def test_ask_rejects_too_short_question(client):
    headers = {"X-Session-ID": "sess_test123"}
    resp = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert resp.status_code == 422  # pydantic min_length=3 validation
