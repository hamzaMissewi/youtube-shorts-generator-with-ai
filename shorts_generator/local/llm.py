"""Local LLM backend — OpenAI or Gemini, selected by LLM_PROVIDER."""
from ..config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    return response.text or ""


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "auto").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    if provider == "auto":
        if GEMINI_API_KEY:
            try:
                return call_gemini_llm(prompt)
            except Exception as exc:
                print(
                    f"[llm] Gemini attempt failed; falling back to OpenAI: {exc}",
                    flush=True,
                )
        if OPENAI_API_KEY:
            return call_openai_llm(prompt)
        raise RuntimeError(
            "LLM_PROVIDER=auto selected but no Gemini or OpenAI API key is configured. "
            "Set LLM_PROVIDER=auto/openai/gemini and provide the corresponding key."
        )
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', or 'auto'."
    )
