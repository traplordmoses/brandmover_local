"""
Reads and parses brand guidelines and example content from local files.
"""

import glob
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def load_guidelines() -> str:
    """Read the brand guidelines markdown file and return its full text."""
    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
    if not guidelines_path.exists():
        logger.warning("guidelines.md not found at %s", guidelines_path)
        return ""
    text = guidelines_path.read_text(encoding="utf-8")
    logger.info("Loaded guidelines: %d characters from %s", len(text), guidelines_path)
    return text


def load_examples() -> list[str]:
    """Load all .txt example articles from brand/examples/articles/."""
    articles_dir = Path(settings.BRAND_FOLDER) / "examples" / "articles"
    if not articles_dir.exists():
        return []
    files = sorted(glob.glob(str(articles_dir / "*.txt")))
    examples = []
    for f in files:
        text = Path(f).read_text(encoding="utf-8").strip()
        if text:
            examples.append(text)
            logger.info("Loaded example: %s (%d chars)", Path(f).name, len(text))
    return examples


def get_brand_context() -> str:
    """Combine brand name, guidelines, and example articles into a single context string."""
    parts = [f"Brand Name: {settings.BRAND_NAME}"]

    guidelines = load_guidelines()
    if guidelines:
        parts.append(f"--- BRAND GUIDELINES ---\n{guidelines}")

    examples = load_examples()
    if examples:
        parts.append("--- EXAMPLE POSTS ---")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:\n{ex}")

    context = "\n\n".join(parts)
    logger.info("Built brand context: %d total characters", len(context))
    return context
