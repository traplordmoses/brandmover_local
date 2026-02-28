"""
Brand config parser — reads brand/guidelines.md into structured config.

Provides typed color/font/identity lookups with fallbacks so the compositor
works even if the guidelines file is missing or malformed.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_GUIDELINES_PATH = Path(__file__).resolve().parent.parent / "brand" / "guidelines.md"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColorEntry:
    role: str       # "primary", "accent_1", "background", etc.
    name: str       # "Aqua", "Candy Pink"
    hex: str        # "#72e1ff"
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class FontEntry:
    use: str        # "display", "body", "terminal"
    family: str     # "Orbitron", "Inter", "VT323"
    weight: str     # "Bold", "Regular/Medium"


@dataclass
class BrandConfig:
    brand_name: str = ""
    tagline: str = ""
    website: str = ""
    x_handle: str = ""
    colors: dict[str, ColorEntry] = field(default_factory=dict)
    fonts: dict[str, FontEntry] = field(default_factory=dict)
    style_keywords: list[str] = field(default_factory=list)
    raw_hash: str = ""
    parsed_at: float = 0.0
    source_path: str = ""


# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------

def _normalize_role(raw: str) -> str:
    """'Accent 1' -> 'accent_1', 'Background Alt' -> 'background_alt'."""
    return re.sub(r"\s+", "_", raw.strip().lower())


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_COLOR_ROW = re.compile(
    r"\|\s*(?P<role>[^|]+?)\s*\|\s*(?P<name>[^|]+?)\s*\|"
    r"\s*`?(?P<hex>#[0-9a-fA-F]{6})`?\s*\|"
    r"\s*\(?\s*(?P<r>\d+)\s*,\s*(?P<g>\d+)\s*,\s*(?P<b>\d+)\s*\)?\s*\|",
)

_FONT_ROW = re.compile(
    r"\|\s*(?P<use>[^|]+?)\s*\|\s*(?P<font>[^|]+?)\s*\|"
    r"\s*(?P<weight>[^|]+?)\s*\|\s*(?P<style>[^|]*?)\s*\|",
)


def _parse_colors(text: str) -> dict[str, ColorEntry]:
    colors: dict[str, ColorEntry] = {}
    for m in _COLOR_ROW.finditer(text):
        role = _normalize_role(m.group("role"))
        # Skip header row artifacts
        if role in ("color", "-------", "---"):
            continue
        colors[role] = ColorEntry(
            role=role,
            name=m.group("name").strip(),
            hex=m.group("hex").strip().lower(),
            rgb=(int(m.group("r")), int(m.group("g")), int(m.group("b"))),
        )
    return colors


def _parse_fonts(text: str) -> dict[str, FontEntry]:
    fonts: dict[str, FontEntry] = {}
    # Scope to TYPOGRAPHY section only
    section_match = re.search(r"##\s*TYPOGRAPHY(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if not section_match:
        return fonts
    section = section_match.group(1)
    for m in _FONT_ROW.finditer(section):
        raw_use = m.group("use").strip()
        # Skip header/separator rows
        if raw_use.lower() in ("use", "---", "-------") or raw_use.startswith("-"):
            continue
        use_key = _normalize_role(raw_use.split("/")[0])  # "Display / Headlines" -> "display"
        fonts[use_key] = FontEntry(
            use=use_key,
            family=m.group("font").strip(),
            weight=m.group("weight").strip(),
        )
    return fonts


def _parse_identity(text: str) -> dict[str, str]:
    identity: dict[str, str] = {}
    for key, field_name in [
        ("Brand Name", "brand_name"),
        ("Tagline", "tagline"),
        ("Website", "website"),
        ("X Handle", "x_handle"),
    ]:
        m = re.search(rf"\*\*{re.escape(key)}:\*\*\s*(.+)", text)
        if m:
            identity[field_name] = m.group(1).strip()
    return identity


def _parse_style_keywords(text: str) -> list[str]:
    """Extract style keywords from ILLUSTRATION STYLE section."""
    keywords: list[str] = []
    m = re.search(r"##\s*ILLUSTRATION STYLE(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if not m:
        return keywords
    section = m.group(1)
    # Pull out the bold keyword phrases
    for bm in re.finditer(r"\*\*([^*]+)\*\*", section):
        val = bm.group(1).strip().rstrip(":")
        if val and len(val) < 80:
            keywords.append(val)
    return keywords


# ---------------------------------------------------------------------------
# Config cache
# ---------------------------------------------------------------------------

_cached_config: BrandConfig | None = None


def get_config(path: Path | None = None) -> BrandConfig:
    """Return the current BrandConfig, re-parsing only when the file changes."""
    global _cached_config
    src = path or _GUIDELINES_PATH

    if not src.exists():
        logger.warning("Brand guidelines not found at %s — using empty config", src)
        if _cached_config is None:
            _cached_config = BrandConfig(source_path=str(src))
        return _cached_config

    raw = src.read_text(encoding="utf-8")
    md5 = hashlib.md5(raw.encode()).hexdigest()

    if _cached_config is not None and _cached_config.raw_hash == md5:
        return _cached_config

    colors = _parse_colors(raw)
    fonts = _parse_fonts(raw)
    identity = _parse_identity(raw)
    style_kw = _parse_style_keywords(raw)

    _cached_config = BrandConfig(
        brand_name=identity.get("brand_name", ""),
        tagline=identity.get("tagline", ""),
        website=identity.get("website", ""),
        x_handle=identity.get("x_handle", ""),
        colors=colors,
        fonts=fonts,
        style_keywords=style_kw,
        raw_hash=md5,
        parsed_at=time.time(),
        source_path=str(src),
    )
    logger.info(
        "Parsed brand config: %d colors, %d fonts, brand=%s",
        len(colors), len(fonts), _cached_config.brand_name,
    )
    return _cached_config


def invalidate_cache() -> None:
    """Force re-parse on next get_config() call."""
    global _cached_config
    _cached_config = None


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

def get_color_rgb(role: str, fallback: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    cfg = get_config()
    entry = cfg.colors.get(role)
    return entry.rgb if entry else fallback


def get_color_hex(role: str, fallback: str = "#ffffff") -> str:
    cfg = get_config()
    entry = cfg.colors.get(role)
    return entry.hex if entry else fallback


def get_font_family(use: str) -> str:
    cfg = get_config()
    entry = cfg.fonts.get(use)
    return entry.family if entry else ""


def get_brand_summary() -> dict:
    """Return a dict suitable for the /brand command display."""
    cfg = get_config()
    return {
        "brand_name": cfg.brand_name,
        "tagline": cfg.tagline,
        "website": cfg.website,
        "x_handle": cfg.x_handle,
        "colors": {
            role: {"name": c.name, "hex": c.hex, "rgb": c.rgb}
            for role, c in cfg.colors.items()
        },
        "fonts": {
            use: {"family": f.family, "weight": f.weight}
            for use, f in cfg.fonts.items()
        },
        "style_keywords": cfg.style_keywords,
        "parsed_at": cfg.parsed_at,
        "source_path": cfg.source_path,
    }
