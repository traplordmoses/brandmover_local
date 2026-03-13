"""
Model fallback chains — automatic retry with alternative models.

When the primary model fails (rate limit, overload, outage), automatically
falls back to the next model in the chain. This keeps the bot operational
even during API issues.

Default chains:
- Agent: claude-sonnet-4-6 → claude-haiku-4-5-20251001
- Haiku tasks: claude-haiku-4-5-20251001 (no fallback — already cheapest)

Usage:
    response = await call_with_fallback(
        client=client,
        models=["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        tools=tools,
    )
"""

import logging
import anthropic

logger = logging.getLogger(__name__)

# HTTP status codes that trigger fallback (transient errors)
_RETRIABLE_STATUS = {429, 500, 502, 503, 529}

# Default fallback chains
DEFAULT_CHAINS: dict[str, list[str]] = {
    "agent": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "generation": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "haiku": ["claude-haiku-4-5-20251001"],
}


def get_fallback_chain(primary_model: str) -> list[str]:
    """Get the fallback chain for a model. Returns [primary] if no chain defined."""
    for chain in DEFAULT_CHAINS.values():
        if primary_model in chain:
            # Return from the primary model onward
            idx = chain.index(primary_model)
            return chain[idx:]
    return [primary_model]


async def call_with_fallback(
    client: anthropic.AsyncAnthropic,
    models: list[str] | None = None,
    primary_model: str | None = None,
    **kwargs,
) -> anthropic.types.Message:
    """Call Claude with automatic model fallback.

    Args:
        client: Anthropic client instance.
        models: Explicit fallback chain [primary, fallback1, ...].
                If not provided, uses primary_model to look up default chain.
        primary_model: Primary model name (used to look up default chain).
        **kwargs: Passed directly to client.messages.create() (system, messages, tools, etc.)

    Returns:
        Anthropic Message response.

    Raises:
        The last error if all models fail.
    """
    if models is None:
        model = primary_model or kwargs.pop("model", "claude-sonnet-4-6")
        models = get_fallback_chain(model)

    # Remove 'model' from kwargs if present — we set it per attempt
    kwargs.pop("model", None)

    last_error = None
    for i, model in enumerate(models):
        try:
            response = await client.messages.create(model=model, **kwargs)
            if i > 0:
                logger.info("Fallback succeeded: %s (after %d failed attempt(s))", model, i)
            return response
        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code in _RETRIABLE_STATUS and i < len(models) - 1:
                logger.warning(
                    "Model %s returned %d — falling back to %s",
                    model, e.status_code, models[i + 1],
                )
                continue
            raise
        except anthropic.APIConnectionError as e:
            last_error = e
            if i < len(models) - 1:
                logger.warning(
                    "Model %s connection error — falling back to %s: %s",
                    model, models[i + 1], e,
                )
                continue
            raise

    # Should not reach here, but just in case
    raise last_error  # type: ignore[misc]
