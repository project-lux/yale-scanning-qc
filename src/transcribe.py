"""
Transcription and page-type classification backends.

Three interchangeable transcription backends, selected in Settings:

- sglang: an OpenAI-compatible SGLang server on an adjacent machine
  (POST {base_url}/v1/chat/completions with an inline base64 image).
- vertex: Google Vertex AI via the google-genai SDK
  (project cultural-heritage-gemini, model gemini-3.1-flash-lite).
- kraken: local OCR — kraken segmentation + the PP-OCRv6 line recognizer
  (see :mod:`src.ocr`). No network, no prompt; returns per-line polygons.
- nara: local VLM — the NARA-tuned Qwen 0.8B via transformers on MPS
  (see :mod:`src.nara`). Whole-page text, no polygons.

Page-type classification still needs one of the LLM backends or a trained
local model; kraken/nara only read text.

Auth for Vertex uses Application Default Credentials
(``gcloud auth application-default login`` on the QC host).

Besides transcription, the configured backend doubles as a page-type
classifier while no locally trained page-type model is active — its
labels bootstrap the training data for that model.
"""

import base64
from pathlib import Path

import requests

from .config import PAGE_TYPE_DESCRIPTIONS, all_page_types

TRANSCRIPTION_PROMPT = (
    "Transcribe all handwritten text in this scanned page exactly as written. "
    "Preserve line breaks. If a word is illegible, write [illegible]. "
    "Return only the transcription with no commentary."
)


def page_type_prompt(page_types: list[str]) -> str:
    """Classification prompt over the current class list (built-in + custom)."""
    lines = []
    for pt in page_types:
        desc = PAGE_TYPE_DESCRIPTIONS.get(pt)
        lines.append(f"- {pt} ({desc})" if desc else f"- {pt}")
    return (
        "Classify this scanned page. Answer with exactly one of these labels "
        "and nothing else:\n" + "\n".join(lines)
    )


def _generate_sglang(jpeg_path: Path, prompt: str, settings: dict,
                     max_tokens: int = 4096) -> str:
    """One image + prompt round trip against an OpenAI-compatible SGLang server."""
    base_url = settings.get("sglang_base_url", "http://localhost:30000").rstrip("/")
    model = settings.get("sglang_model", "default")
    headers = {"Content-Type": "application/json"}
    api_key = settings.get("sglang_api_key", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    b64 = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                }},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                         headers=headers, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _generate_vertex(jpeg_path: Path, prompt: str, settings: dict,
                     max_tokens: int = 4096) -> str:
    """One image + prompt round trip against Vertex AI Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=settings.get("vertex_project", "cultural-heritage-gemini"),
        location=settings.get("vertex_location", "global"),
    )
    response = client.models.generate_content(
        model=settings.get("vertex_model", "gemini-3.1-flash-lite"),
        contents=[
            types.Part.from_bytes(data=jpeg_path.read_bytes(), mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return (response.text or "").strip()


BACKENDS = {
    "sglang": _generate_sglang,
    "vertex": _generate_vertex,
}

# Backends that can answer an arbitrary prompt about a page (and can
# therefore double as the page-type classifier). kraken only reads text.
LLM_BACKENDS = ("sglang", "vertex")


def transcribe(jpeg_path: Path, settings: dict) -> tuple[str | None, str, dict | None]:
    """
    Transcribe using the configured backend.

    Returns (text, backend_name, geometry); (None, "", None) when the
    backend is 'off'. *geometry* is the per-line polygon layout from local
    OCR (see :func:`src.ocr.transcribe_kraken`) — None for LLM backends,
    which return plain text only.
    Raises on backend errors so the pipeline can record them per scan.
    """
    backend = settings.get("transcription_backend", "off")
    if backend == "kraken":
        from .ocr import transcribe_kraken

        text, geometry = transcribe_kraken(jpeg_path, settings)
        return text, backend, geometry
    if backend == "nara":
        from .nara import transcribe_nara

        return transcribe_nara(jpeg_path, settings), backend, None
    fn = BACKENDS.get(backend)
    if fn is None:
        return None, "", None
    return fn(jpeg_path, TRANSCRIPTION_PROMPT, settings), backend, None


def classify_page_llm(jpeg_path: Path, settings: dict,
                      page_types: list[str] | None = None) -> tuple[str | None, str]:
    """
    Classify the page type with the configured backend.

    Returns (page_type, backend_name) with page_type from *page_types*
    (defaults to the full configured list including custom classes), or
    (None, "") when the backend is 'off' or the answer is unusable.
    Raises on backend errors.
    """
    backend = settings.get("transcription_backend", "off")
    fn = BACKENDS.get(backend) if backend in LLM_BACKENDS else None
    if fn is None:
        return None, ""

    if page_types is None:
        page_types = all_page_types(settings)

    answer = fn(jpeg_path, page_type_prompt(page_types), settings, max_tokens=32).lower()
    # Longest label first so e.g. a custom 'handwriting_margin' isn't
    # swallowed by the built-in 'handwriting' substring.
    for pt in sorted(page_types, key=len, reverse=True):
        if pt in answer:
            return pt, backend
    return None, backend
