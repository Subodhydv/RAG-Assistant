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
    "You are an intelligent teaching assistant. "
    "Primary Instruction: Answer the student's question using the provided lecture transcript excerpts. "
    "If the provided lecture excerpts do NOT contain enough information to answer the question, "
    "provide a short, accurate, and direct answer based on your general AI knowledge. "
    "When using general knowledge outside the provided lecture excerpts, prefix your answer with "
    "'[General AI Knowledge]: ' so the student knows it was not found in the current lecture transcripts."
)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        ts = f"{int(c.start // 60)}:{int(c.start % 60):02d}"
        blocks.append(f"[Excerpt {i} | video={c.video_id} @ {ts}]\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context = _format_context(chunks) if chunks else "(no matching lecture excerpts found in vector store)"
    return (
        f"Lecture Excerpts Context:\n{context}\n\n"
        f"Student Question: {question}\n\n"
        "Instructions: Answer using the excerpts if relevant. If context is missing/insufficient, give a short, correct answer based on general knowledge starting with '[General AI Knowledge]: '."
    )


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    # Check if API key is configured for the selected provider
    provider = settings.llm_provider
    if provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif provider == "gemini":
        api_key = settings.gemini_api_key
    else:
        api_key = settings.openai_api_key
    
    # If no key or placeholder key is set, use direct transcript excerpt summary fallback
    if not api_key or api_key in ("sk-ant-...", "sk-...") or api_key.endswith("..."):
        if not chunks:
            return (
                "No relevant excerpts were found in the indexed lecture videos for your question. "
                "Please try uploading a lecture video in the **Ingest Lecture** tab or rephrasing your search."
            )
        return _format_fallback_excerpt_answer(chunks)

    prompt = build_prompt(question, chunks)

    try:
        if provider == "anthropic":
            return _call_anthropic(prompt)
        elif provider == "openai":
            return _call_openai(prompt)
        elif provider == "gemini":
            return _call_gemini(prompt)
        else:
            return _format_fallback_excerpt_answer(chunks)
    except Exception as e:
        if not chunks:
            return f"[General AI Knowledge Fallback Error]: {e}"
        return _format_fallback_excerpt_answer(chunks, error_msg=str(e))


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


