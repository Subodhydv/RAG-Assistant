"""
Stage 4: take retrieved chunks + the user's question, build a grounded
prompt, and call an LLM for the final answer.

Provider is swappable via LLM_PROVIDER env var (anthropic | openai) —
this is the piece the course's "Bonus Video 2: Replacing Local LLM
with GPT-5" is about, generalized so swapping providers doesn't
require touching the retrieval code at all.
"""
from __future__ import annotations

from typing import List

from app.config import settings
from app.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You are an intelligent teaching assistant powered by Google Gemini. "
    "Instructions for answering student questions:\n"
    "1. If relevant lecture transcript excerpts are provided in the context, answer the question grounded in those excerpts and reference the video timestamps when appropriate.\n"
    "2. If no relevant lecture excerpts are provided, OR if the student asks a question outside/beyond the scope of the current lecture context, DO NOT say you cannot answer. Instead, provide a clear, accurate, and direct answer based on your general AI knowledge.\n"
    "3. When answering from general knowledge outside the provided lecture excerpts, start your answer with '[General AI Knowledge]: ' so the student knows it was not found in the current lecture transcripts."
)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        ts = f"{int(c.start // 60)}:{int(c.start % 60):02d}"
        blocks.append(f"[Excerpt {i} | video={c.video_id} @ {ts}]\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: List[RetrievedChunk], history: List[dict] | None = None) -> str:
    parts = []
    if history:
        history_lines = ["Previous Conversation History:"]
        for turn in history[-4:]:  # Include up to 4 recent turns
            q = turn.get("question", "")
            a = turn.get("answer", "")
            if q:
                history_lines.append(f"Student: {q}")
            if a:
                history_lines.append(f"Assistant: {a}")
        parts.append("\n".join(history_lines) + "\n")

    if chunks:
        context = _format_context(chunks)
        parts.append(
            f"Lecture Excerpts Context:\n{context}\n\n"
            f"Student Question: {question}\n\n"
            "Instructions: Answer using the provided lecture excerpts if relevant. If the question goes beyond or is outside the excerpts context, provide a full, accurate answer using your general AI knowledge starting with '[General AI Knowledge]: '."
        )
    else:
        parts.append(
            f"Student Question: {question}\n\n"
            "Instructions: No matching lecture video excerpts were found in the vector store for this question. Provide a complete, accurate, and helpful answer to the student's question using your AI knowledge starting with '[General AI Knowledge]: '."
        )

    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[RetrievedChunk], history: List[dict] | None = None) -> str:
    # Check configured provider API key
    provider = settings.llm_provider
    if provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif provider == "gemini":
        api_key = settings.gemini_api_key
    else:
        api_key = settings.openai_api_key

    # Automatic fallback to Gemini if configured provider API key is not present
    if not api_key or api_key in ("sk-ant-...", "sk-...") or api_key.endswith("..."):
        if settings.gemini_api_key and not settings.gemini_api_key.startswith("sk-"):
            provider = "gemini"
            api_key = settings.gemini_api_key

    # If still no valid key, format fallback excerpt summary
    if not api_key or api_key in ("sk-ant-...", "sk-...") or api_key.endswith("..."):
        if not chunks:
            return (
                "No relevant excerpts were found in the indexed lecture videos for your question. "
                "Please try uploading a lecture video in the **Ingest Lecture** tab or rephrasing your search."
            )
        return _format_fallback_excerpt_answer(chunks)

    prompt = build_prompt(question, chunks, history=history)

    try:
        if provider == "gemini":
            return _call_gemini(prompt)
        elif provider == "openai":
            return _call_openai(prompt)
        elif provider == "anthropic":
            return _call_anthropic(prompt)
        else:
            return _call_gemini(prompt)
    except Exception as e:
        if settings.gemini_api_key and provider != "gemini":
            try:
                return _call_gemini(prompt)
            except Exception:
                pass
        if not chunks:
            return f"[General AI Knowledge Fallback Error]: {e}"
        return _format_fallback_excerpt_answer(chunks, error_msg=str(e))


def generate_answer_stream(question: str, chunks: List[RetrievedChunk], history: List[dict] | None = None):
    """Generator yielding text chunks for Server-Sent Events (SSE) streaming."""
    full_answer = generate_answer(question, chunks, history=history)
    # Stream in word chunks for smooth SSE rendering
    words = full_answer.split(" ")
    for i in range(0, len(words), 3):
        chunk_text = " ".join(words[i:i+3]) + " "
        yield chunk_text


def _format_fallback_excerpt_answer(chunks: List[RetrievedChunk], error_msg: str | None = None) -> str:
    lines = ["Based on the retrieved lecture transcript excerpts:\n"]
    for i, c in enumerate(chunks, start=1):
        m, s = int(c.start // 60), int(c.start % 60)
        ts = f"{m}:{s:02d}"
        lines.append(f"**Excerpt {i}** ({c.source_filename} @ {ts}):")
        lines.append(f"> \"{c.text}\"\n")

    if error_msg:
        lines.append(f"*(LLM API Call notice: {error_msg}. Returning direct transcript excerpts above. Configure a valid ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in your .env file for AI synthesis.)*")
    else:
        lines.append("*(Note: To enable AI synthesis with Gemini, Claude, or GPT, add your `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` to your `.env` file.)*")

    return "\n".join(lines)


def _call_anthropic(prompt: str) -> str:
    import anthropic

    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.max_answer_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=settings.max_answer_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_gemini(prompt: str) -> str:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        )
        return response.text or ""
    except Exception:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(f"{SYSTEM_PROMPT}\n\n{prompt}")
        return response.text or ""


