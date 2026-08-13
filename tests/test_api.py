"""
Exercises API endpoints end-to-end with signed headers and mocked dependencies.
Run with: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

from app.auth import sign_session_id
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
        lambda question, chunks, history=None: f"Stubbed answer using {len(chunks)} excerpt(s).",
    )

    return TestClient(main_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["session_isolated"] is True
    assert resp.json()["async_ingest_supported"] is True


def test_ask_returns_answer_and_citations(client):
    headers = {"X-Session-ID": sign_session_id("sess_test123")}
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
    headers = {"X-Session-ID": sign_session_id("sess_test123")}
    resp = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert resp.status_code == 422  # pydantic min_length=3 validation


def test_export_notes(client):
    payload = {
        "title": "Data Structures Study Notes",
        "messages": [
            {
                "question": "What is hash map lookup complexity?",
                "answer": "Average O(1) time complexity.",
                "citations": [{"source_filename": "ds.mp3", "start": 10.0, "end": 25.0}],
            }
        ],
    }
    resp = client.post("/export-notes", json=payload)
    assert resp.status_code == 200
    assert "Data Structures Study Notes" in resp.text
    assert "What is hash map lookup complexity?" in resp.text
    assert "Average O(1) time complexity." in resp.text


def test_quiz_endpoint(client, monkeypatch):
    from app.models import VideoTranscript, TranscriptSegment, QuizResponse, QuizQuestion
    import app.main as main_module

    fake_transcript = VideoTranscript(
        video_id="vid_test",
        source_filename="test_lecture.mp4",
        language="en",
        segments=[
            TranscriptSegment(start=10.0, end=20.0, text="Binary search trees maintain sorted keys."),
            TranscriptSegment(start=25.0, end=40.0, text="Hash tables provide fast constant time access.")
        ]
    )

    fake_quiz_response = QuizResponse(
        video_id="vid_test",
        title="Quiz: test_lecture.mp4",
        questions=[
            QuizQuestion(
                id=1,
                question="What maintains sorted keys?",
                options=["BST", "Hash table", "Queue", "Stack"],
                correct_answer="BST",
                explanation="At 10:00 BSTs maintain sorted keys.",
                timestamp="10:00"
            ),
            QuizQuestion(
                id=2,
                question="What provides constant time access?",
                options=["Hash table", "Array", "Linked List", "Tree"],
                correct_answer="Hash table",
                explanation="At 25:00 Hash tables provide O(1) access.",
                timestamp="25:00"
            )
        ]
    )

    monkeypatch.setattr(main_module, "load_transcript", lambda video_id=None, session_id=None: fake_transcript)
    monkeypatch.setattr(main_module, "generate_quiz_questions", lambda transcript, num_questions=5: fake_quiz_response)

    headers = {"X-Session-ID": sign_session_id("sess_test123")}
    resp = client.post("/quiz", json={"num_questions": 2}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "questions" in data
    assert len(data["questions"]) == 2
    assert "options" in data["questions"][0]
    assert "correct_answer" in data["questions"][0]
