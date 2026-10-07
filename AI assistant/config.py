"""Shared configuration and LLM helper (works with any OpenAI-compatible provider)."""
import os

from openai import OpenAI

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "workspace_files")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# Free / low-cost providers that speak the OpenAI API format.
PROVIDERS = {
    "Google Gemini (free)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
        "needs_key": True,
    },
    "Groq (free)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "needs_key": True,
    },
    "Ollama (local, no key)": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "needs_key": False,
    },
    "OpenAI (paid)": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "needs_key": True,
    },
}
DEFAULT_PROVIDER = "Google Gemini (free)"

_state = {
    "provider": DEFAULT_PROVIDER,
    "api_key": os.getenv("LLM_API_KEY", ""),
    "model": os.getenv("LLM_MODEL", PROVIDERS[DEFAULT_PROVIDER]["model"]),
}
_client = None


def configure(provider: str, api_key: str, model: str):
    """Update provider settings at runtime; the client is rebuilt only if something changed."""
    global _client
    new = {"provider": provider, "api_key": (api_key or "").strip(),
           "model": (model or "").strip() or PROVIDERS[provider]["model"]}
    if new != _state:
        _state.update(new)
        _client = None


def get_model() -> str:
    return _state["model"]


def has_api_key() -> bool:
    return bool(_state["api_key"]) or not PROVIDERS[_state["provider"]]["needs_key"]


def get_client() -> OpenAI:
    global _client
    if not has_api_key():
        raise RuntimeError("No API key set. Enter it in the sidebar.")
    if _client is None:
        cfg = PROVIDERS[_state["provider"]]
        _client = OpenAI(api_key=_state["api_key"] or "ollama", base_url=cfg["base_url"],
                         timeout=40.0, max_retries=1)  # fail fast instead of hanging
    return _client


def ask(system: str, user: str, max_tokens: int = 1500) -> str:
    """Single-turn LLM call that returns plain text."""
    resp = get_client().chat.completions.create(
        model=get_model(),
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    text = (resp.choices[0].message.content or "").strip()
    return text or "(The model returned no text.)"
