"""
Token-budget context assembly engine.

Instead of stuffing everything into the system prompt, this module assembles
context blocks in priority order within a configurable token budget.

PRIORITY LEVELS (lower number = higher priority):
0 — System prompt core (always included, never truncated)
1 — Brand guidelines (essential, truncated if too long)
2 — Skill registry (small, almost always fits)
3 — Session context (recent posts, rejections)
4 — Learned preferences
5 — Personality injection
6 — Feedback history (raw entries, lowest priority)

Usage:
    engine = ContextEngine(budget_tokens=8000)
    engine.add("system_prompt", priority=0, content=system_prompt)
    engine.add("guidelines", priority=1, content=brand_context, truncatable=True)
    engine.add("skills", priority=2, content=skill_summary)
    engine.add("session", priority=3, content=session_context)
    assembled = engine.assemble()
"""

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 characters (conservative for English text)
CHARS_PER_TOKEN = 4

# Default budget — leaves room for the agent's response + tool definitions
DEFAULT_BUDGET_TOKENS = 12000


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return len(text) // CHARS_PER_TOKEN


@dataclass
class ContextBlock:
    """A named block of context with priority and optional truncation."""
    name: str
    content: str
    priority: int = 5
    truncatable: bool = False  # If True, can be shortened to fit budget
    min_chars: int = 500      # Minimum chars to keep if truncated

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def truncate(self, max_chars: int) -> str:
        """Return content truncated to max_chars with a marker."""
        if len(self.content) <= max_chars:
            return self.content
        cut = max(self.min_chars, max_chars - 50)  # Reserve space for marker
        return self.content[:cut] + f"\n\n[... truncated {len(self.content) - cut} chars to fit context budget]"


class ContextEngine:
    """Assembles context blocks within a token budget.

    Blocks are added with priorities. On assemble(), blocks are included
    in priority order. Lower priority numbers are included first.
    If the budget is exceeded, truncatable blocks are shortened.
    Non-truncatable blocks that don't fit are dropped with a warning.
    """

    def __init__(self, budget_tokens: int = DEFAULT_BUDGET_TOKENS):
        self.budget_tokens = budget_tokens
        self.blocks: list[ContextBlock] = []

    def add(
        self,
        name: str,
        content: str,
        priority: int = 5,
        truncatable: bool = False,
        min_chars: int = 500,
    ) -> None:
        """Add a context block. Empty content is silently skipped."""
        if not content or not content.strip():
            return
        self.blocks.append(ContextBlock(
            name=name,
            content=content.strip(),
            priority=priority,
            truncatable=truncatable,
            min_chars=min_chars,
        ))

    def assemble(self) -> str:
        """Assemble all blocks within the token budget.

        Returns the concatenated context string.
        Also logs what was included/excluded for debugging.
        """
        sorted_blocks = sorted(self.blocks, key=lambda b: b.priority)
        budget_chars = self.budget_tokens * CHARS_PER_TOKEN

        included: list[tuple[str, str]] = []  # (name, content)
        used_chars = 0

        for block in sorted_blocks:
            remaining = budget_chars - used_chars
            content_len = len(block.content)

            if content_len <= remaining:
                # Fits completely
                included.append((block.name, block.content))
                used_chars += content_len
            elif block.truncatable and remaining >= block.min_chars:
                # Truncate to fit
                truncated = block.truncate(remaining)
                included.append((block.name, truncated))
                used_chars += len(truncated)
                logger.info(
                    "Context block '%s' truncated: %d → %d chars",
                    block.name, content_len, len(truncated),
                )
            elif block.priority == 0:
                # Priority 0 is always included regardless of budget
                included.append((block.name, block.content))
                used_chars += content_len
                logger.warning(
                    "Context block '%s' exceeds budget but is priority 0 — included anyway (%d chars)",
                    block.name, content_len,
                )
            else:
                # Doesn't fit, skip
                logger.info(
                    "Context block '%s' dropped: needs %d chars, only %d remaining",
                    block.name, content_len, remaining,
                )

        total_tokens = estimate_tokens("".join(c for _, c in included))
        logger.info(
            "Context assembled: %d/%d blocks, ~%d/%d tokens",
            len(included), len(sorted_blocks), total_tokens, self.budget_tokens,
        )

        return "\n\n".join(content for _, content in included)

    def get_stats(self) -> dict:
        """Return stats about registered blocks (for debugging)."""
        return {
            "budget_tokens": self.budget_tokens,
            "blocks": [
                {
                    "name": b.name,
                    "priority": b.priority,
                    "tokens": b.tokens,
                    "truncatable": b.truncatable,
                }
                for b in sorted(self.blocks, key=lambda b: b.priority)
            ],
            "total_tokens": sum(b.tokens for b in self.blocks),
        }


async def build_brand_context_block(budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> str:
    """Build brand context using priority-based budget assembly.

    Assembles brand guidelines, examples, references, session context, and
    learned preferences within a token budget. Higher-priority blocks are
    included first; lower-priority blocks are truncated or dropped if they
    don't fit.

    This replaces ad-hoc string concatenation with intelligent budgeting —
    as the brand corpus grows, this ensures the most important context
    always fits within the model's effective attention window.

    Loads guidelines, examples, and references concurrently via asyncio.gather.

    Returns:
        Assembled brand context string.
    """
    engine = ContextEngine(budget_tokens=budget_tokens)

    from agent import guidelines

    # Load all brand data concurrently — these are independent file I/O calls
    guidelines_text, examples, references = await asyncio.gather(
        asyncio.to_thread(guidelines.load_guidelines),
        asyncio.to_thread(guidelines.load_examples),
        asyncio.to_thread(guidelines.load_references),
    )

    # Priority 0: Brand guidelines (core — always included)
    if guidelines_text:
        engine.add(
            "guidelines",
            f"--- BRAND GUIDELINES ---\n{guidelines_text}",
            priority=0,
        )

    # Priority 1: Example posts (important for voice matching, truncatable)
    if examples:
        examples_text = "--- EXAMPLE POSTS ---\n" + "\n\n".join(
            f"Example {i}:\n{ex}" for i, ex in enumerate(examples, 1)
        )
        engine.add("examples", examples_text, priority=1, truncatable=True)

    # NOTE: Session context (recent posts, rejections, preferences) is injected
    # directly into the user message by engine.run_agent() to avoid duplication.
    # Do NOT add it here — it was previously called twice per agent run.

    # Priority 3: Reference materials (PDFs, docs — truncatable, lower priority)
    if references:
        refs_text = "--- REFERENCE MATERIALS ---\n" + "\n\n".join(
            f"[{ref['name']}]\n{ref['text']}" for ref in references
        )
        engine.add(
            "references", refs_text, priority=3,
            truncatable=True, min_chars=1000,
        )

    assembled = engine.assemble()
    stats = engine.get_stats()
    logger.info(
        "Brand context assembled: ~%d/%d tokens, %d/%d blocks included",
        stats["total_tokens"], budget_tokens,
        len([b for b in stats["blocks"]]), len(stats["blocks"]),
    )
    return assembled
