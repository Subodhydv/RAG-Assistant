"""
Generation module for interactive quizzes and flashcards grounded in indexed lecture transcripts.
Generates structured questions (MCQs with options, correct answer, explanation, timestamp).
"""
from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional

from app.config import settings
from app.models import QuizQuestion, QuizResponse, VideoTranscript


QUIZ_SYSTEM_PROMPT = (
    "You are an expert educational assessment creator. "
    "Given a lecture transcript, generate multiple-choice quiz questions to test student comprehension. "
    "Return ONLY valid JSON matching this structure:\n"
    "[\n"
    "  {\n"
    "    \"id\": 1,\n"
    "    \"question\": \"What is ...?\",\n"
    "    \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
    "    \"correct_answer\": \"Option A\",\n"
    "    \"explanation\": \"Why Option A is correct according to the lecture...\",\n"
    "    \"timestamp\": \"01:30\"\n"
    "  }\n"
    "]"
)


def generate_quiz_questions(
    transcript: VideoTranscript,
    num_questions: int = 5
) -> QuizResponse:
    """Generate interactive quiz questions from a video transcript."""
    segments = transcript.segments
    if not segments:
        return QuizResponse(
            video_id=transcript.video_id,
            title=f"Quiz for {transcript.source_filename}",
            questions=_generate_fallback_quiz(transcript, num_questions)
        )

    # Combine text with timestamp hints
    combined_text_lines = []
    for s in segments:
        m = int(s.start // 60)
        sec = int(s.start % 60)
        combined_text_lines.append(f"[{m:02d}:{sec:02d}] {s.text}")

    full_transcript_text = "\n".join(combined_text_lines)[:6000]

    provider = settings.llm_provider
    api_key = None
    if provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif provider == "gemini":
        api_key = settings.gemini_api_key
    else:
        api_key = settings.openai_api_key

    if not api_key or api_key in ("sk-ant-...", "sk-...") or api_key.endswith("..."):
        return QuizResponse(
            video_id=transcript.video_id,
            title=f"Quiz: {transcript.source_filename}",
            questions=_generate_fallback_quiz(transcript, num_questions)
        )

    prompt = (
        f"Lecture Title: {transcript.source_filename}\n\n"
        f"Transcript Content:\n{full_transcript_text}\n\n"
        f"Generate {num_questions} multiple-choice quiz questions in JSON format."
    )

    try:
        if provider == "gemini":
            raw_output = _call_gemini(prompt)
        elif provider == "openai":
            raw_output = _call_openai(prompt)
        elif provider == "anthropic":
            raw_output = _call_anthropic(prompt)
        else:
            raw_output = ""

        questions = _parse_quiz_json(raw_output, transcript, num_questions)
        return QuizResponse(
            video_id=transcript.video_id,
            title=f"Quiz: {transcript.source_filename}",
            questions=questions
        )
    except Exception:
        return QuizResponse(
            video_id=transcript.video_id,
            title=f"Quiz: {transcript.source_filename}",
            questions=_generate_fallback_quiz(transcript, num_questions)
        )


def _parse_quiz_json(raw_text: str, transcript: VideoTranscript, num_questions: int) -> List[QuizQuestion]:
    """Parse JSON string output from LLM into QuizQuestion objects."""
    # Find JSON array using regex
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        data = json.loads(json_str)
        questions = []
        for i, item in enumerate(data, start=1):
            questions.append(QuizQuestion(
                id=i,
                question=item.get("question", f"Question {i}"),
                options=item.get("options", ["A", "B", "C", "D"]),
                correct_answer=item.get("correct_answer", item.get("options", ["A"])[0]),
                explanation=item.get("explanation", "Refer to transcript timestamps."),
                timestamp=item.get("timestamp", "00:00")
            ))
        if questions:
            return questions[:num_questions]
    return _generate_fallback_quiz(transcript, num_questions)


def _generate_fallback_quiz(transcript: VideoTranscript, num_questions: int) -> List[QuizQuestion]:
    """Deterministic, transcript-based quiz generator when LLM API keys are not provided."""
    segments = transcript.segments
    questions = []

    if not segments:
        return [
            QuizQuestion(
                id=1,
                question=f"What is the main topic of {transcript.source_filename}?",
                options=[
                    f"Core concepts of {transcript.source_filename}",
                    "Unrelated subject matter",
                    "Random audio test",
                    "Advanced system engineering"
                ],
                correct_answer=f"Core concepts of {transcript.source_filename}",
                explanation="Automatically generated assessment question from indexed lecture filename.",
                timestamp="00:00"
            )
        ]

    step = max(1, len(segments) // num_questions)
    selected_segments = segments[::step][:num_questions]

    for i, seg in enumerate(selected_segments, start=1):
        text_clean = seg.text.strip().capitalize()
        if len(text_clean) > 80:
            text_clean = text_clean[:77] + "..."

        m = int(seg.start // 60)
        sec = int(seg.start % 60)
        ts_str = f"{m:02d}:{sec:02d}"

        correct_opt = f"Statement discussed around {ts_str}: '{text_clean}'"
        distractor_1 = "Opposite statement not supported by transcript context."
        distractor_2 = "Unrelated algorithms introduced in a different section."
        distractor_3 = "Incorrect assumption unsupported by video excerpt."

        questions.append(QuizQuestion(
            id=i,
            question=f"According to the lecture around timestamp {ts_str}, which of the following is true?",
            options=[correct_opt, distractor_1, distractor_2, distractor_3],
            correct_answer=correct_opt,
            explanation=f"At timestamp {ts_str}, the instructor explicitly states: '{seg.text.strip()}'",
            timestamp=ts_str
        ))

    return questions


def _call_gemini(prompt: str) -> str:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"{QUIZ_SYSTEM_PROMPT}\n\n{prompt}",
        )
        return response.text or ""
    except Exception:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=QUIZ_SYSTEM_PROMPT
        )
        res = model.generate_content(prompt)
        return res.text


def _call_openai(prompt: str) -> str:
    import openai
    client = openai.OpenAI(api_key=settings.openai_api_key)
    res = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return res.choices[0].message.content or ""


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    res = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1500,
        system=QUIZ_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.content[0].text
