from src.config import cfg, ANTHROPIC_API_KEY, OPENAI_API_KEY, OFFLINE_MODE
from src.generate.prompt import SYSTEM_PROMPT, ABSTENTION_STRING, build_user_message
from src.logging_setup import get_logger

log = get_logger("generator")

BACKEND = cfg["generation"]["backend"]
MODEL = cfg["generation"]["model"]
TEMPERATURE = cfg["generation"]["temperature"]
TOP_P = cfg["generation"]["top_p"]
MAX_TOKENS = cfg["generation"]["max_output_tokens"]


def generate(question: str, chunks: list[dict]) -> str:
    if OFFLINE_MODE and BACKEND != "ollama":
        raise RuntimeError("OFFLINE_MODE=1 but backend is not ollama. Set backend=ollama or unset OFFLINE_MODE.")

    user_msg = build_user_message(question, chunks)

    if BACKEND == "anthropic":
        return _anthropic(user_msg)
    elif BACKEND == "openai":
        return _openai(user_msg)
    elif BACKEND == "ollama":
        return _ollama(user_msg)
    else:
        raise ValueError(f"Unknown generator backend: {BACKEND}")


def _anthropic(user_msg: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        stop_sequences=["\n\nQUESTION:"],
    )
    return response.content[0].text.strip()


def _openai(user_msg: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
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
    return response.choices[0].message.content.strip()


def _ollama(user_msg: str) -> str:
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
    return r.json()["message"]["content"].strip()
