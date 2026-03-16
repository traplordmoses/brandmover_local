"""
Model fallback chains — automatic retry with alternative models + providers.

When the primary model fails (rate limit, overload, outage), automatically
falls back to the next model in the chain. Supports cross-provider fallback:
Anthropic → OpenAI → Gemini.

Default chains:
- Agent: claude-sonnet-4-6 → claude-haiku-4-5-20251001 → gpt-4o → gemini-2.0-flash
- Haiku tasks: claude-haiku-4-5-20251001 (no fallback — already cheapest)

Usage:
    response = await call_with_fallback(
        messages=messages,
        system=system_prompt,
        max_tokens=4096,
        primary_model="claude-sonnet-4-6",
    )
"""

import logging
import os

import anthropic

logger = logging.getLogger(__name__)

# HTTP status codes that trigger fallback (transient errors)
_RETRIABLE_STATUS = {429, 500, 502, 503, 529}

# Provider detection from model name
_PROVIDER_PREFIXES = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
}


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    model_lower = model.lower()
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model_lower.startswith(prefix):
            return provider
    return "anthropic"  # default


# Default fallback chains (cross-provider)
DEFAULT_CHAINS: dict[str, list[str]] = {
    "agent": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4o", "gemini-2.0-flash"],
    "generation": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4o"],
    "haiku": ["claude-haiku-4-5-20251001"],
}


def _parse_env_chain() -> list[str] | None:
    """Parse AGENT_FALLBACK_MODELS env var into a chain."""
    raw = os.getenv("AGENT_FALLBACK_MODELS", "")
    if not raw:
        return None
    return [m.strip() for m in raw.split(",") if m.strip()]


def get_fallback_chain(primary_model: str) -> list[str]:
    """Get the fallback chain for a model.

    Priority: AGENT_FALLBACK_MODELS env → DEFAULT_CHAINS → [primary]
    """
    # Check env override first
    env_chain = _parse_env_chain()
    if env_chain:
        if primary_model in env_chain:
            idx = env_chain.index(primary_model)
            return env_chain[idx:]
        return [primary_model] + env_chain

    # Check default chains
    for chain in DEFAULT_CHAINS.values():
        if primary_model in chain:
            idx = chain.index(primary_model)
            return chain[idx:]
    return [primary_model]


# --- Provider-specific clients (lazy-initialized) ---

_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_client = None
_gemini_configured = False


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        from config import settings
        _anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic_client


async def _call_anthropic(model: str, **kwargs) -> dict:
    """Call Anthropic API and return normalized response."""
    client = _get_anthropic_client()
    response = await client.messages.create(model=model, **kwargs)
    return _normalize_anthropic(response)


def _normalize_anthropic(response: anthropic.types.Message) -> dict:
    """Normalize Anthropic response to common format."""
    blocks = []
    for block in response.content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    return {
        "provider": "anthropic",
        "model": response.model,
        "content": blocks,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "_raw": response,
    }


async def _call_openai(model: str, **kwargs) -> dict:
    """Call OpenAI API and return normalized response."""
    from config import settings
    import httpx

    messages = kwargs.get("messages", [])
    system = kwargs.get("system", "")
    max_tokens = kwargs.get("max_tokens", 4096)

    # Convert Anthropic message format to OpenAI format
    oai_messages = []
    if system:
        if isinstance(system, list):
            system_text = "\n".join(
                b["text"] for b in system if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            system_text = system
        oai_messages.append({"role": "system", "content": system_text})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Convert content blocks to text
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    text_parts.append(str(block.get("content", "")))
            content = "\n".join(text_parts)
        oai_messages.append({"role": role, "content": content})

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": oai_messages,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]
    return {
        "provider": "openai",
        "model": data.get("model", model),
        "content": [{"type": "text", "text": choice["message"]["content"]}],
        "stop_reason": choice.get("finish_reason", "end_turn"),
        "usage": data.get("usage", {}),
        "_raw": data,
    }


async def _call_gemini(model: str, **kwargs) -> dict:
    """Call Google Gemini API and return normalized response."""
    from config import settings
    import httpx

    messages = kwargs.get("messages", [])
    system = kwargs.get("system", "")
    max_tokens = kwargs.get("max_tokens", 4096)

    # Build Gemini request
    contents = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            content = "\n".join(text_parts)
        contents.append({"role": role, "parts": [{"text": content}]})

    system_text = ""
    if system:
        if isinstance(system, list):
            system_text = "\n".join(
                b["text"] for b in system if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            system_text = system

    body: dict = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": settings.GEMINI_API_KEY},
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

    candidate = data["candidates"][0]
    text = candidate["content"]["parts"][0]["text"]
    return {
        "provider": "google",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": candidate.get("finishReason", "STOP"),
        "usage": data.get("usageMetadata", {}),
        "_raw": data,
    }


# Provider dispatch
_PROVIDER_CALLERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_gemini,
}


def _provider_available(provider: str) -> bool:
    """Check if a provider has API keys configured."""
    from config import settings
    if provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY)
    if provider == "google":
        return bool(settings.GEMINI_API_KEY)
    return False


async def call_with_fallback(
    client: anthropic.AsyncAnthropic | None = None,
    models: list[str] | None = None,
    primary_model: str | None = None,
    **kwargs,
) -> anthropic.types.Message:
    """Call LLM with automatic model + provider fallback.

    Tries each model in the chain. If a model's provider is unavailable
    (no API key), skips it. Cross-provider fallback converts message
    formats automatically.

    For backwards compatibility, returns the raw Anthropic Message if the
    winning model is Anthropic. For other providers, wraps the response
    in a compatible object.

    Args:
        client: Optional Anthropic client (for backwards compatibility).
        models: Explicit fallback chain [primary, fallback1, ...].
        primary_model: Primary model name (used to look up default chain).
        **kwargs: Passed to the provider API (system, messages, max_tokens, tools, etc.)

    Returns:
        Anthropic Message response (or compatible wrapper for other providers).
    """
    if models is None:
        model = primary_model or kwargs.pop("model", "claude-sonnet-4-6")
        models = get_fallback_chain(model)

    kwargs.pop("model", None)

    # Filter to models whose providers have API keys
    available_models = [m for m in models if _provider_available(_detect_provider(m))]
    if not available_models:
        available_models = models[:1]  # try anyway — will error with clear message

    last_error = None
    for i, model in enumerate(available_models):
        provider = _detect_provider(model)
        try:
            if provider == "anthropic":
                # Use direct Anthropic client for full tool_use support
                actual_client = client or _get_anthropic_client()
                response = await actual_client.messages.create(model=model, **kwargs)
                if i > 0:
                    logger.info("Fallback succeeded: %s/%s (after %d failed)", provider, model, i)
                return response
            else:
                # Cross-provider fallback — text only (no tool_use)
                caller = _PROVIDER_CALLERS[provider]
                # Strip tools for non-Anthropic providers (they can't handle Anthropic tool format)
                fallback_kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
                result = caller(model, **fallback_kwargs)
                # Handle both sync and async callers
                if hasattr(result, '__await__'):
                    result = await result
                if i > 0:
                    logger.info("Fallback succeeded: %s/%s (after %d failed)", provider, model, i)
                # Wrap in Anthropic-compatible response
                return _wrap_as_anthropic(result)

        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code in _RETRIABLE_STATUS and i < len(available_models) - 1:
                logger.warning(
                    "Model %s/%s returned %d — falling back to %s",
                    provider, model, e.status_code, available_models[i + 1],
                )
                continue
            raise
        except (anthropic.APIConnectionError, Exception) as e:
            last_error = e
            if i < len(available_models) - 1:
                logger.warning(
                    "Model %s/%s failed — falling back to %s: %s",
                    provider, model, available_models[i + 1], e,
                )
                continue
            raise

    raise last_error  # type: ignore[misc]


def _wrap_as_anthropic(result: dict) -> anthropic.types.Message:
    """Wrap a normalized provider response as an Anthropic Message for compatibility."""
    # Extract text from the normalized result
    text_content = ""
    for block in result.get("content", []):
        if block.get("type") == "text":
            text_content += block["text"]

    # Build a minimal Anthropic-compatible Message
    return anthropic.types.Message(
        id=f"msg_{result.get('provider', 'fallback')}",
        type="message",
        role="assistant",
        content=[anthropic.types.TextBlock(type="text", text=text_content)],
        model=result.get("model", "unknown"),
        stop_reason="end_turn",
        usage=anthropic.types.Usage(
            input_tokens=result.get("usage", {}).get("input_tokens", 0),
            output_tokens=result.get("usage", {}).get("output_tokens", 0),
        ),
    )
