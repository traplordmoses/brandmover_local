"""
Reads and parses brand guidelines, example content, and reference materials from local files.
Supports PDF, Markdown, and plain text references.
"""

import glob
import logging
import os
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


def load_pdf(path: str | Path) -> str:
    """Extract text from a PDF file using PyMuPDF. Falls back to OCR for image-based PDFs."""
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
            else:
                # Image-based page — try OCR via Tesseract
                try:
                    ocr_text = page.get_textpage_ocr(flags=0, full=True).extractText()
                    if ocr_text.strip():
                        pages.append(ocr_text)
                        logger.debug("OCR'd page %d of %s: %d chars", page.number, Path(path).name, len(ocr_text))
                except Exception as ocr_err:
                    logger.debug("OCR failed on page %d of %s: %s", page.number, Path(path).name, ocr_err)
        doc.close()
        text = "\n".join(pages).strip()
        logger.info("Loaded PDF: %s (%d chars, %d pages)", Path(path).name, len(text), len(pages))
        return text
    except Exception as e:
        logger.warning("Failed to read PDF %s: %s", path, e)
        return ""


def load_references() -> list[dict]:
    """
    Scan brand/references/ recursively for .pdf, .md, .txt files.
    Returns list of dicts with keys: name, path, text, mtime.
    Sorted newest-first. Respects MAX_REFERENCE_CHARS budget.
    """
    refs_dir = Path(settings.REFERENCES_FOLDER)
    if not refs_dir.exists():
        logger.info("References folder not found: %s", refs_dir)
        return []

    entries = []
    for root, _dirs, files in os.walk(refs_dir):
        for fname in files:
            fpath = Path(root) / fname
            suffix = fpath.suffix.lower()
            if suffix not in (".pdf", ".md", ".txt"):
                continue
            if fname.startswith("."):
                continue
            entries.append({
                "name": fname,
                "path": str(fpath),
                "mtime": fpath.stat().st_mtime,
                "suffix": suffix,
            })

    # Sort newest first
    entries.sort(key=lambda e: e["mtime"], reverse=True)

    budget = settings.MAX_REFERENCE_CHARS
    used = 0
    results = []

    for entry in entries:
        if used >= budget:
            logger.info("Reference budget exhausted (%d/%d chars), skipping remaining files", used, budget)
            break

        if entry["suffix"] == ".pdf":
            text = load_pdf(entry["path"])
        else:
            try:
                text = Path(entry["path"]).read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning("Failed to read %s: %s", entry["path"], e)
                continue

        if not text:
            continue

        # Truncate if it would exceed budget
        remaining = budget - used
        if len(text) > remaining:
            text = text[:remaining]
            logger.info("Truncated %s to %d chars (budget limit)", entry["name"], remaining)

        results.append({
            "name": entry["name"],
            "path": entry["path"],
            "text": text,
            "chars": len(text),
        })
        used += len(text)

    logger.info("Loaded %d reference files (%d total chars)", len(results), used)
    return results


def get_reference_summary() -> str:
    """
    Lightweight inventory of reference files for the /refs command.
    Lists files with sizes, no content loaded.
    """
    refs_dir = Path(settings.REFERENCES_FOLDER)
    if not refs_dir.exists():
        return "No references folder found. Create brand/references/ and drop PDFs/docs there."

    entries = []
    for root, _dirs, files in os.walk(refs_dir):
        for fname in files:
            fpath = Path(root) / fname
            suffix = fpath.suffix.lower()
            if suffix not in (".pdf", ".md", ".txt"):
                continue
            if fname.startswith("."):
                continue
            rel = fpath.relative_to(refs_dir)
            size_kb = fpath.stat().st_size / 1024
            entries.append((str(rel), size_kb))

    if not entries:
        return "References folder is empty. Drop PDFs, .md, or .txt files into brand/references/"

    lines = [f"Found {len(entries)} reference file(s):\n"]
    for rel_path, size_kb in sorted(entries):
        lines.append(f"  {rel_path} ({size_kb:.1f} KB)")
    lines.append(f"\nMax context budget: {settings.MAX_REFERENCE_CHARS:,} chars")
    return "\n".join(lines)


async def extract_brand_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF and use Claude to generate brand guidelines markdown.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        Generated guidelines.md content as a string.
    """
    import anthropic
    from config import settings as _settings

    raw_text = load_pdf(pdf_path)
    if not raw_text:
        raise ValueError("Could not extract text from PDF. The file may be empty or image-only.")

    # Truncate very long PDFs to fit in context
    if len(raw_text) > 80000:
        raw_text = raw_text[:80000] + "\n\n[... truncated ...]"

    client = anthropic.AsyncAnthropic(api_key=_settings.ANTHROPIC_API_KEY)

    system = (
        "You are a brand strategist. Given raw text extracted from a brand guidelines PDF, "
        "produce a clean, structured Markdown document that a social media AI agent can use. "
        "Include these sections: Brand Voice, Visual Style (colors, typography, imagery), "
        "Approved Hashtags, Content Dos, Content Don'ts / Never Use, Target Audience, "
        "and any other relevant brand rules found in the document. "
        "Be thorough but concise. Output ONLY the Markdown — no commentary."
    )

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Here is the raw text from a brand guidelines PDF for {_settings.BRAND_NAME}:\n\n{raw_text}",
        }],
    )

    guidelines_md = response.content[0].text.strip()
    logger.info("Generated guidelines from PDF: %d chars", len(guidelines_md))
    return guidelines_md


def get_brand_context() -> str:
    """Combine brand name, guidelines, example articles, and reference materials into a single context string."""
    parts = [f"Brand Name: {settings.BRAND_NAME}"]

    guidelines = load_guidelines()
    if guidelines:
        parts.append(f"--- BRAND GUIDELINES ---\n{guidelines}")

    examples = load_examples()
    if examples:
        parts.append("--- EXAMPLE POSTS ---")
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:\n{ex}")

    references = load_references()
    if references:
        parts.append("--- REFERENCE MATERIALS ---")
        for ref in references:
            parts.append(f"[{ref['name']}]\n{ref['text']}")

    context = "\n\n".join(parts)
    logger.info("Built brand context: %d total characters", len(context))
    return context
