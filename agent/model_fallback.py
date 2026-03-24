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

from agent._client import get_anthropic, get_httpx

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
    "agent": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-5.4", "gemini-2.0-flash"],
    "generation": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-5.4"],
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

_openai_client = None
_gemini_configured = False


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    """Return the shared AsyncAnthropic client from agent._client."""
    return get_anthropic()


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


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    oai_tools = []
    for tool in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return oai_tools


def _anthropic_tool_choice_to_openai(tool_choice: dict) -> str | dict | None:
    """Convert Anthropic tool_choice to OpenAI format."""
    tc_type = tool_choice.get("type", "auto")
    if tc_type == "auto":
        return "auto"
    if tc_type == "none":
        return "none"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


async def _call_openai(model: str, **kwargs) -> dict:
    """Call OpenAI API and return normalized response."""
    import json as _json
    from config import settings

    messages = kwargs.get("messages", [])
    system = kwargs.get("system", "")
    max_tokens = kwargs.get("max_tokens", 4096)
    tools = kwargs.get("tools", [])
    tool_choice = kwargs.get("tool_choice", None)

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

    def _block_type(block) -> str:
        """Get type from a block — works with both dicts and Anthropic SDK objects."""
        if isinstance(block, dict):
            return block.get("type", "")
        return getattr(block, "type", "")

    def _block_attr(block, attr, default=""):
        """Get attribute from a block — works with both dicts and Anthropic SDK objects."""
        if isinstance(block, dict):
            return block.get(attr, default)
        return getattr(block, attr, default)

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            tool_call_blocks = []
            tool_result_blocks = []
            for block in content:
                btype = _block_type(block)
                if btype == "text":
                    text_parts.append(_block_attr(block, "text", ""))
                elif btype == "tool_use":
                    tool_call_blocks.append(block)
                elif btype == "tool_result":
                    tool_result_blocks.append(block)

            # If this is an assistant message with tool_use blocks, emit as tool_calls
            if role == "assistant" and tool_call_blocks:
                oai_msg: dict = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
                oai_msg["tool_calls"] = [
                    {
                        "id": _block_attr(b, "id"),
                        "type": "function",
                        "function": {
                            "name": _block_attr(b, "name"),
                            "arguments": _json.dumps(_block_attr(b, "input", {})) if isinstance(_block_attr(b, "input", {}), dict) else str(_block_attr(b, "input", {})),
                        },
                    }
                    for b in tool_call_blocks
                ]
                oai_messages.append(oai_msg)
                continue

            # If this is a user message with tool_result blocks, emit as tool messages
            if tool_result_blocks:
                if text_parts:
                    oai_messages.append({"role": "user", "content": "\n".join(text_parts)})
                for b in tool_result_blocks:
                    result_content = _block_attr(b, "content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            _block_attr(p, "text", "") for p in result_content if _block_type(p) == "text"
                        )
                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": _block_attr(b, "tool_use_id", ""),
                        "content": str(result_content),
                    })
                continue

            content = "\n".join(text_parts)
        oai_messages.append({"role": role, "content": content})

    body: dict = {
        "model": model,
        "messages": oai_messages,
        "max_completion_tokens": max_tokens,
    }

    # Add tools if provided
    if tools:
        body["tools"] = _anthropic_tools_to_openai(tools)
        if tool_choice:
            body["tool_choice"] = _anthropic_tool_choice_to_openai(tool_choice)

    client = get_httpx()
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )
    if resp.status_code != 200:
        try:
            error_body = resp.json()
        except Exception:
            error_body = resp.text
        logger.error("OpenAI API error %d: %s", resp.status_code, error_body)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    msg = choice["message"]

    # Build content blocks — handle both text and tool_calls
    blocks = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            try:
                args = _json.loads(tc["function"]["arguments"])
            except (_json.JSONDecodeError, TypeError):
                args = {}
            blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": args,
            })

    return {
        "provider": "openai",
        "model": data.get("model", model),
        "content": blocks,
        "stop_reason": "tool_use" if msg.get("tool_calls") else choice.get("finish_reason", "end_turn"),
        "usage": data.get("usage", {}),
        "_raw": data,
    }


def _anthropic_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to Gemini function-calling format."""
    function_declarations = []
    for tool in tools:
        decl: dict = {
            "name": tool["name"],
            "description": tool.get("description", ""),
        }
        schema = tool.get("input_schema")
        if schema:
            decl["parameters"] = schema
        function_declarations.append(decl)
    return [{"function_declarations": function_declarations}]


async def _call_gemini(model: str, **kwargs) -> dict:
    """Call Google Gemini API and return normalized response."""
    import json as _json

    from config import settings

    messages = kwargs.get("messages", [])
    system = kwargs.get("system", "")
    max_tokens = kwargs.get("max_tokens", 4096)
    tools = kwargs.get("tools", [])

    def _block_type(block) -> str:
        if isinstance(block, dict):
            return block.get("type", "")
        return getattr(block, "type", "")

    def _block_attr(block, attr, default=""):
        if isinstance(block, dict):
            return block.get(attr, default)
        return getattr(block, attr, default)

    # Build Gemini request — convert Anthropic messages to Gemini format
    contents = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        parts = []
        if isinstance(content, list):
            for block in content:
                btype = _block_type(block)
                if btype == "text":
                    text_val = _block_attr(block, "text", "")
                    if text_val:
                        parts.append({"text": text_val})
                elif btype == "tool_use":
                    # Assistant requesting a function call
                    parts.append({
                        "functionCall": {
                            "name": _block_attr(block, "name"),
                            "args": _block_attr(block, "input", {}),
                        }
                    })
                elif btype == "tool_result":
                    # User providing function response
                    result_content = _block_attr(block, "content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            _block_attr(p, "text", "") for p in result_content if _block_type(p) == "text"
                        )
                    parts.append({
                        "functionResponse": {
                            "name": _block_attr(block, "tool_use_id", "unknown"),
                            "response": {"result": str(result_content)},
                        }
                    })
            if not parts:
                parts.append({"text": ""})
        else:
            parts.append({"text": content})
        contents.append({"role": role, "parts": parts})

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

    # Add tools if provided
    if tools:
        body["tools"] = _anthropic_tools_to_gemini(tools)

    client = get_httpx()
    resp = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": settings.GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    candidate = data["candidates"][0]
    parts = candidate["content"].get("parts", [])

    # Parse response parts — may contain text and/or functionCall
    blocks = []
    for part in parts:
        if "text" in part:
            blocks.append({"type": "text", "text": part["text"]})
        elif "functionCall" in part:
            fc = part["functionCall"]
            blocks.append({
                "type": "tool_use",
                "id": f"toolu_gemini_{fc['name']}",
                "name": fc["name"],
                "input": fc.get("args", {}),
            })

    # Determine stop reason
    finish_reason = candidate.get("finishReason", "STOP")
    has_tool_use = any(b.get("type") == "tool_use" for b in blocks)
    stop_reason = "tool_use" if has_tool_use else finish_reason

    return {
        "provider": "google",
        "model": model,
        "content": blocks if blocks else [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
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
                # Cross-provider fallback with tool-use support for OpenAI and Gemini.
                # Function-calling is converted automatically for both providers.
                caller = _PROVIDER_CALLERS[provider]
                had_tools = "tools" in kwargs or "tool_choice" in kwargs
                if provider in ("openai", "google"):
                    # OpenAI and Gemini support function calling — pass tools through
                    fallback_kwargs = dict(kwargs)
                    # Gemini doesn't support tool_choice — strip it
                    if provider == "google":
                        fallback_kwargs.pop("tool_choice", None)
                    logger.info("Falling back to %s/%s with tool-use support", provider, model)
                else:
                    # Other providers — strip tools (degraded)
                    fallback_kwargs = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
                    if had_tools:
                        logger.warning(
                            "ARCH-03: Falling back to %s/%s — tools stripped. "
                            "Tool-use capability is DEGRADED.",
                            provider, model,
                        )
                result = caller(model, **fallback_kwargs)
                if hasattr(result, '__await__'):
                    result = await result
                if i > 0:
                    logger.info("Fallback succeeded: %s/%s (after %d failed)", provider, model, i)
                tools_degraded = had_tools and provider not in ("openai", "google")
                return _wrap_as_anthropic(result, tools_degraded=tools_degraded)

        except anthropic.APIStatusError as e:
            last_error = e
            # Treat credit-balance errors (400) as retriable so we fall back
            is_credit_error = e.status_code == 400 and "credit balance" in str(e).lower()
            if (e.status_code in _RETRIABLE_STATUS or is_credit_error) and i < len(available_models) - 1:
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


def _wrap_as_anthropic(result: dict, *, tools_degraded: bool = False) -> anthropic.types.Message:
    """Wrap a normalized provider response as an Anthropic Message for compatibility."""
    content_blocks: list[anthropic.types.TextBlock | anthropic.types.ToolUseBlock] = []

    for block in result.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            text = block["text"]
            if tools_degraded and not content_blocks:
                text = (
                    "[FALLBACK NOTICE: This response was generated by a non-Anthropic provider "
                    f"({result.get('provider', 'unknown')}/{result.get('model', 'unknown')}) "
                    "without tool-use capability. The agent cannot call tools in this turn.]\n\n"
                    + text
                )
            content_blocks.append(anthropic.types.TextBlock(type="text", text=text))
        elif block.get("type") == "tool_use":
            content_blocks.append(anthropic.types.ToolUseBlock(
                type="tool_use",
                id=block["id"],
                name=block["name"],
                input=block["input"],
            ))

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append(anthropic.types.TextBlock(type="text", text=""))

    # Map stop_reason
    stop_reason = result.get("stop_reason", "end_turn")
    if stop_reason == "tool_use":
        stop_reason = "tool_use"
    elif stop_reason in ("stop", "end_turn"):
        stop_reason = "end_turn"
    else:
        stop_reason = "end_turn"

    return anthropic.types.Message(
        id=f"msg_{result.get('provider', 'fallback')}",
        type="message",
        role="assistant",
        content=content_blocks,
        model=result.get("model", "unknown"),
        stop_reason=stop_reason,
        usage=anthropic.types.Usage(
            input_tokens=result.get("usage", {}).get("input_tokens", 0) or 0,
            output_tokens=result.get("usage", {}).get("output_tokens", 0) or 0,
        ),
    )
