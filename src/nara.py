"""
Local VLM transcription with the NARA-tuned Qwen model
(wjbmattingly/nara-qwen-3.5-0.8b-ep1 by default).

A 0.8B Qwen3.5 fine-tune for transcribing handwritten historical documents
(National Archives pension files, cursive). Whole-page image-to-text: no
line polygons, just the transcription.

Two run modes, chosen by the nara_server_url setting:
- empty: load the model locally through transformers (device_map="auto"
  puts it on MPS on the QC Mac);
- "spark:<port>": vLLM serving the model on the DGX Spark, reached through
  the app's SSH tunnel (the box only exposes SSH). Any http(s) URL is used
  directly, so a future non-tunneled vLLM works too.

Two hard requirements from the model card:
- the prompt must be the repo's prompt.txt, verbatim;
- enable_thinking=False, or the model reasons instead of transcribing.
"""

import threading
from pathlib import Path

DEFAULT_NARA_MODEL = "wjbmattingly/nara-qwen-3.5-0.8b-ep1"
# --served-model-name used when vLLM hosts the model on the DGX.
SERVED_MODEL_NAME = "nara-qwen"

_lock = threading.Lock()
_engine: dict | None = None  # {"model_id", "model", "processor", "prompt"}
_prompt_cache: dict[str, str] = {}


def _prompt_for(model_id: str) -> str:
    """The repo's prompt.txt, verbatim (local dir or hub download)."""
    if model_id not in _prompt_cache:
        local = Path(model_id).expanduser()
        if local.is_dir():
            _prompt_cache[model_id] = (local / "prompt.txt").read_text()
        else:
            from huggingface_hub import hf_hub_download
            _prompt_cache[model_id] = Path(
                hf_hub_download(model_id, "prompt.txt")).read_text()
    return _prompt_cache[model_id]


def _server_base(settings: dict) -> str | None:
    """Resolved base URL of a vLLM server, or None for local inference."""
    raw = (settings.get("nara_server_url") or "").strip()
    if not raw:
        return None
    if raw.startswith("spark:"):
        from . import remote
        port = int(raw.split(":", 1)[1] or 8061)
        return f"http://127.0.0.1:{remote.tunnel_port(port)}"
    return raw.rstrip("/")


def _transcribe_server(jpeg_path: Path, settings: dict, base: str) -> str:
    """One page through vLLM's OpenAI-compatible chat completions API."""
    import base64

    import requests

    model_id = (settings.get("nara_model_id") or "").strip() or DEFAULT_NARA_MODEL
    b64 = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _prompt_for(model_id)},
            ],
        }],
        "max_tokens": 3072,
        "temperature": 0.0,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _load(model_id: str) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    return {"model_id": model_id, "model": model,
            "processor": processor, "prompt": _prompt_for(model_id)}


def transcribe_nara(jpeg_path: Path, settings: dict) -> str:
    """Transcribe one page image. Raises on load/inference errors."""
    base = _server_base(settings)
    if base:
        return _transcribe_server(jpeg_path, settings, base)

    from PIL import Image

    model_id = (settings.get("nara_model_id") or "").strip() or DEFAULT_NARA_MODEL

    global _engine
    with _lock:
        if _engine is None or _engine["model_id"] != model_id:
            _engine = _load(model_id)
        model, processor = _engine["model"], _engine["processor"]

        image = Image.open(jpeg_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": _engine["prompt"]},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = processor(text=[text], images=[image],
                           return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=3072,
                             do_sample=False, repetition_penalty=1.1)
        return processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
