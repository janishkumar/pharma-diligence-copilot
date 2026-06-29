"""Answer generation with bounded retry, graceful degradation, and usage accounting.

Returns a GenerationResult (not a bare string) so the pipeline can surface
truncation and token/cost accounting. Any provider error degrades to a
GenerationResult flagged as an error rather than crashing the request.
"""
import os
import time
from dataclasses import dataclass

from src.config import cfg, ANTHROPIC_API_KEY, OPENAI_API_KEY, OFFLINE_MODE
from src.generate.prompt import SYSTEM_PROMPT, ABSTENTION_STRING, build_user_message
from src.logging_setup import get_logger

log = get_logger("generator")

# scripts/03_ask.py --local sets this to switch to the local Ollama backend.
BACKEND = os.getenv("GENERATOR_BACKEND_OVERRIDE") or cfg["generation"]["backend"]
MODEL = cfg["generation"]["model"]
TEMPERATURE = cfg["generation"]["temperature"]
TOP_P = cfg["generation"]["top_p"]
MAX_TOKENS = cfg["generation"]["max_output_tokens"]

MAX_RETRIES = 3
BACKOFF_BASE = 1.0

# Approx USD per 1M tokens (input, output) for cost accounting.
_PRICES = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "gpt-4o": (2.5, 10.0),
}


@dataclass
class GenerationResult:
    text: str
    truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = _PRICES.get(model, (0.0, 0.0))
    return (in_tok / 1e6) * pin + (out_tok / 1e6) * pout


def _with_retry(fn):
    """Call fn() with bounded exponential backoff on transient errors. On
    exhaustion or a non-retryable error, return a GenerationResult error."""
    delay = BACKOFF_BASE
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            name = type(e).__name__
            retryable = any(k in name for k in ("RateLimit", "APIStatus", "APIConnection", "Overloaded", "Timeout", "InternalServer"))
            if not retryable or attempt == MAX_RETRIES:
                log.error("generation_failed", error=name, detail=str(e)[:200], attempt=attempt + 1)
                return GenerationResult(text=ABSTENTION_STRING, error=f"{name}: {e}")
            log.warning("generation_retry", error=name, attempt=attempt + 1, wait_sec=round(delay, 1))
            time.sleep(delay)
            delay *= 2
    return GenerationResult(text=ABSTENTION_STRING, error="retries_exhausted")


def generate(question: str, chunks: list[dict]) -> GenerationResult:
    if OFFLINE_MODE and BACKEND != "ollama":
        raise RuntimeError("OFFLINE_MODE=1 but backend is not ollama. Set backend=ollama or unset OFFLINE_MODE.")

    user_msg = build_user_message(question, chunks)

    if BACKEND == "anthropic":
        return _with_retry(lambda: _anthropic(user_msg))
    elif BACKEND == "openai":
        return _with_retry(lambda: _openai(user_msg))
    elif BACKEND == "ollama":
        return _with_retry(lambda: _ollama(user_msg))
    else:
        raise ValueError(f"Unknown generator backend: {BACKEND}")


def _anthropic(user_msg: str) -> GenerationResult:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # Anthropic rejects specifying BOTH temperature and top_p. temperature=0.1 is
    # our determinism control, so we send only temperature here (top_p applies to
    # the openai/ollama backends).
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        stop_sequences=["\n\nQUESTION:"],
    )
    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    text = "".join(text_parts).strip()
    if not text:
        return GenerationResult(text=ABSTENTION_STRING, error="empty_response")
    in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
    return GenerationResult(
        text=text,
        truncated=(resp.stop_reason == "max_tokens"),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=_cost(MODEL, in_tok, out_tok),
    )


def _openai(user_msg: str) -> GenerationResult:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stop=["\n\nQUESTION:"],
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        return GenerationResult(text=ABSTENTION_STRING, error="empty_response")
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    return GenerationResult(
        text=text,
        truncated=(choice.finish_reason == "length"),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=_cost(MODEL, in_tok, out_tok),
    )


def _ollama(user_msg: str) -> GenerationResult:
    import httpx
    host = cfg["serving"]["ollama_host"]
    model_name = cfg["serving"]["ollama_model_name"]
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "top_p": TOP_P, "num_predict": MAX_TOKENS},
    }
    r = httpx.post(f"{host}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    text = (data.get("message", {}).get("content") or "").strip()
    if not text:
        return GenerationResult(text=ABSTENTION_STRING, error="empty_response")
    return GenerationResult(
        text=text,
        truncated=(data.get("done_reason") == "length"),
    )
