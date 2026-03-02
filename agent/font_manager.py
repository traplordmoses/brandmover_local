"""
Font Manager — dynamic font resolution with Google Fonts download + caching.

Resolution chain: local fonts dir → Google Fonts download → system fonts → PIL default.

Public API:
    get_font(family, size, weight) -> ImageFont
    list_available_fonts() -> list[str]
    clear_cache() -> None
"""

import logging
from pathlib import Path

import httpx
from PIL import ImageFont

from config import settings

logger = logging.getLogger(__name__)

_FONTS_DIR = Path(settings.BRAND_FOLDER) / "assets" / "fonts"

# Runtime cache: (family, size, weight) → loaded font object
_font_cache: dict[tuple[str, int, str], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

# ---------------------------------------------------------------------------
# Google Fonts registry — common fonts with known GitHub URLs
# ---------------------------------------------------------------------------

# Variable fonts use a single file for all weights
_GOOGLE_VARIABLE_FONTS: dict[str, str] = {
    "Inter": "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    "Orbitron": "https://github.com/google/fonts/raw/main/ofl/orbitron/Orbitron%5Bwght%5D.ttf",
    "Poppins": "",  # static font, handled below
    "Roboto": "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    "Montserrat": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Open Sans": "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
    "Lato": "https://github.com/google/fonts/raw/main/ofl/lato/Lato%5Bwght%5D.ttf",
    "Raleway": "https://github.com/google/fonts/raw/main/ofl/raleway/Raleway%5Bwght%5D.ttf",
    "Oswald": "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "Nunito": "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf",
    "Playfair Display": "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Source Code Pro": "https://github.com/google/fonts/raw/main/ofl/sourcecodepro/SourceCodePro%5Bwght%5D.ttf",
    "Space Grotesk": "https://github.com/google/fonts/raw/main/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf",
    "Space Mono": "",  # static
    "JetBrains Mono": "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
    "Fira Code": "https://github.com/google/fonts/raw/main/ofl/firacode/FiraCode%5Bwght%5D.ttf",
    "DM Sans": "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf",
    "Work Sans": "https://github.com/google/fonts/raw/main/ofl/worksans/WorkSans%5Bwght%5D.ttf",
    "Archivo": "https://github.com/google/fonts/raw/main/ofl/archivo/Archivo%5Bwdth%2Cwght%5D.ttf",
    "Sora": "https://github.com/google/fonts/raw/main/ofl/sora/Sora%5Bwght%5D.ttf",
}

# Static fonts — weight-specific files
_GOOGLE_STATIC_FONTS: dict[str, dict[str, str]] = {
    "Poppins": {
        "Regular": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf",
        "Bold": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
        "SemiBold": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf",
        "ExtraBold": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-ExtraBold.ttf",
        "Black": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Black.ttf",
        "Light": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Light.ttf",
        "Medium": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Medium.ttf",
    },
    "VT323": {
        "Regular": "https://github.com/google/fonts/raw/main/ofl/vt323/VT323-Regular.ttf",
    },
    "Space Mono": {
        "Regular": "https://github.com/google/fonts/raw/main/ofl/spacemono/SpaceMono-Regular.ttf",
        "Bold": "https://github.com/google/fonts/raw/main/ofl/spacemono/SpaceMono-Bold.ttf",
    },
}

# System font fallback paths
_SYSTEM_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_family(family: str) -> str:
    """Normalize font family name for matching."""
    return family.strip()


def _normalize_weight(weight: str) -> str:
    """Normalize weight string: 'bold' → 'Bold', '' → 'Regular'."""
    if not weight:
        return "Regular"
    w = weight.strip().lower()
    mapping = {
        "bold": "Bold",
        "semibold": "SemiBold",
        "semi-bold": "SemiBold",
        "extrabold": "ExtraBold",
        "extra-bold": "ExtraBold",
        "black": "Black",
        "light": "Light",
        "medium": "Medium",
        "regular": "Regular",
        "normal": "Regular",
    }
    return mapping.get(w, weight.strip().title())


def _variable_filename(family: str) -> str:
    """Convert family name to variable font filename: 'Inter' → 'Inter-Variable.ttf'."""
    return f"{family.replace(' ', '')}-Variable.ttf"


def _static_filename(family: str, weight: str) -> str:
    """Convert to static font filename: ('Poppins', 'Bold') → 'Poppins-Bold.ttf'."""
    return f"{family.replace(' ', '')}-{weight}.ttf"


def _try_load_local(family: str, size: int, weight: str) -> ImageFont.FreeTypeFont | None:
    """Try loading from local fonts directory."""
    _FONTS_DIR.mkdir(parents=True, exist_ok=True)
    nw = _normalize_weight(weight)

    # 1. Try variable font
    var_name = _variable_filename(family)
    var_path = _FONTS_DIR / var_name
    if var_path.exists():
        try:
            return ImageFont.truetype(str(var_path), size)
        except (OSError, IOError):
            pass

    # 2. Try weight-specific static font
    static_name = _static_filename(family, nw)
    static_path = _FONTS_DIR / static_name
    if static_path.exists():
        try:
            return ImageFont.truetype(str(static_path), size)
        except (OSError, IOError):
            pass

    # 3. Try any file matching the family name
    for p in _FONTS_DIR.glob(f"{family.replace(' ', '')}*"):
        if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
            try:
                return ImageFont.truetype(str(p), size)
            except (OSError, IOError):
                continue

    return None


def _download_from_google(family: str, weight: str) -> bool:
    """Download font from Google Fonts. Returns True on success."""
    _FONTS_DIR.mkdir(parents=True, exist_ok=True)
    nf = _normalize_family(family)
    nw = _normalize_weight(weight)

    # Check variable fonts first
    if nf in _GOOGLE_VARIABLE_FONTS:
        url = _GOOGLE_VARIABLE_FONTS[nf]
        if url:
            dest = _FONTS_DIR / _variable_filename(nf)
            if dest.exists():
                return True
            return _download_file(url, dest)

    # Check static fonts
    if nf in _GOOGLE_STATIC_FONTS:
        weights = _GOOGLE_STATIC_FONTS[nf]
        url = weights.get(nw) or weights.get("Regular")
        if url:
            dest = _FONTS_DIR / _static_filename(nf, nw)
            if dest.exists():
                return True
            return _download_file(url, dest)

    # Try convention-based URL for unknown fonts
    slug = nf.lower().replace(" ", "")
    filename = f"{nf.replace(' ', '')}%5Bwght%5D.ttf"
    url = f"https://github.com/google/fonts/raw/main/ofl/{slug}/{filename}"
    dest = _FONTS_DIR / _variable_filename(nf)
    return _download_file(url, dest)


def _download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to dest path. Returns True on success."""
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("Downloaded font: %s (%d bytes)", dest.name, len(resp.content))
        return True
    except Exception as e:
        logger.debug("Font download failed for %s: %s", url, e)
        return False


def _try_system_font(size: int) -> ImageFont.FreeTypeFont | None:
    """Try loading a system font as fallback."""
    for sys_path in _SYSTEM_FONT_PATHS:
        try:
            return ImageFont.truetype(sys_path, size)
        except (OSError, IOError):
            continue
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_font(
    family: str,
    size: int,
    weight: str = "Regular",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolve and return a font, downloading from Google Fonts if needed.

    Resolution chain:
    1. Local fonts dir (brand/assets/fonts/)
    2. Google Fonts download (cached locally after first download)
    3. System fonts
    4. PIL default
    """
    cache_key = (family, size, weight)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    # 1. Try local
    font = _try_load_local(family, size, weight)
    if font:
        _font_cache[cache_key] = font
        return font

    # 2. Try Google Fonts download
    if _download_from_google(family, weight):
        font = _try_load_local(family, size, weight)
        if font:
            _font_cache[cache_key] = font
            return font

    # 3. System fallback
    font = _try_system_font(size)
    if font:
        logger.warning("Font %r not found, using system fallback", family)
        _font_cache[cache_key] = font
        return font

    # 4. PIL default
    logger.warning("Font %r not found anywhere, using PIL default", family)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    _font_cache[cache_key] = font
    return font


def list_available_fonts() -> list[str]:
    """List locally available font families (already downloaded or uploaded)."""
    _FONTS_DIR.mkdir(parents=True, exist_ok=True)
    families: set[str] = set()
    for p in _FONTS_DIR.iterdir():
        if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
            # Extract family from filename: "Orbitron-Variable.ttf" → "Orbitron"
            name = p.stem
            for sep in ("-Variable", "-Regular", "-Bold", "-SemiBold",
                        "-ExtraBold", "-Black", "-Light", "-Medium"):
                if name.endswith(sep):
                    name = name[:-len(sep)]
                    break
            families.add(name)
    return sorted(families)


def clear_cache() -> None:
    """Clear the in-memory font cache. Call after brand font changes."""
    _font_cache.clear()
    logger.info("Font cache cleared")
