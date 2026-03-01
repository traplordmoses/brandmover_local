"""
Template Memory — store, analyze, and apply user-uploaded visual templates.

Templates are image frames (Game Boy screen, polaroid, phone mockup, etc.)
that the bot composites generated images into for future posts.

Storage: brand/templates/manifest.json
"""

import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx
from PIL import Image, ImageDraw, ImageFont

from config import settings

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(settings.BRAND_FOLDER) / "templates"
_MANIFEST_PATH = _TEMPLATES_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TemplateRegion:
    type: str  # image|text|logo
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    description: str = ""


@dataclass
class BrandTemplate:
    id: str = ""
    name: str = ""
    path: str = ""
    width: int = 0
    height: int = 0
    regions: list[TemplateRegion] = field(default_factory=list)
    aspect_ratio: str = ""
    content_types: list[str] = field(default_factory=list)
    analysis_notes: str = ""


# ---------------------------------------------------------------------------
# TemplateMemory — manifest CRUD
# ---------------------------------------------------------------------------

class TemplateMemory:
    def __init__(self) -> None:
        self._manifest: list[dict] | None = None

    def _ensure_dir(self) -> None:
        _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> list[dict]:
        if self._manifest is not None:
            return self._manifest
        if not _MANIFEST_PATH.exists():
            self._manifest = []
            return self._manifest
        try:
            self._manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._manifest = []
        return self._manifest

    def save_manifest(self) -> None:
        self._ensure_dir()
        _MANIFEST_PATH.write_text(
            json.dumps(self._manifest or [], indent=2), encoding="utf-8",
        )

    def add_template(self, template: BrandTemplate) -> None:
        manifest = self.load_manifest()
        manifest.append({
            "id": template.id,
            "name": template.name,
            "path": template.path,
            "width": template.width,
            "height": template.height,
            "regions": [
                {
                    "type": r.type,
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                    "description": r.description,
                }
                for r in template.regions
            ],
            "aspect_ratio": template.aspect_ratio,
            "content_types": template.content_types,
            "analysis_notes": template.analysis_notes,
        })
        self.save_manifest()

    def remove_template(self, template_id: str) -> bool:
        manifest = self.load_manifest()
        before = len(manifest)
        self._manifest = [t for t in manifest if t.get("id") != template_id]
        if len(self._manifest) < before:
            self.save_manifest()
            return True
        return False

    def list_templates(self) -> list[BrandTemplate]:
        manifest = self.load_manifest()
        result = []
        for t in manifest:
            regions = [
                TemplateRegion(**r) for r in t.get("regions", [])
            ]
            result.append(BrandTemplate(
                id=t.get("id", ""),
                name=t.get("name", ""),
                path=t.get("path", ""),
                width=t.get("width", 0),
                height=t.get("height", 0),
                regions=regions,
                aspect_ratio=t.get("aspect_ratio", ""),
                content_types=t.get("content_types", []),
                analysis_notes=t.get("analysis_notes", ""),
            ))
        return result

    def get_template_for_content_type(self, content_type: str) -> BrandTemplate | None:
        templates = self.list_templates()
        # First look for a template that explicitly lists this content type
        for t in templates:
            if content_type in t.content_types:
                return t
        # Fall back to the first template if it has no content_type filter (universal)
        for t in templates:
            if not t.content_types:
                return t
        return None


# ---------------------------------------------------------------------------
# Template analysis — Claude Vision
# ---------------------------------------------------------------------------

_TEMPLATE_ANALYSIS_PROMPT = """\
Analyze this image as a design template or frame. Identify content regions where \
different types of content should be placed.

Look for:
1. Large empty/placeholder areas where a generated image should go (type: "image")
2. Areas with placeholder text or text-like spaces (type: "text")
3. Areas where a logo or brand mark would go (type: "logo")

For each region, provide pixel coordinates (x, y from top-left) and dimensions.

Return ONLY valid JSON:
{
  "regions": [
    {"type": "image", "x": 50, "y": 100, "width": 600, "height": 400, "description": "Main image area"},
    {"type": "text", "x": 50, "y": 520, "width": 600, "height": 60, "description": "Title text area"},
    {"type": "logo", "x": 550, "y": 20, "width": 80, "height": 80, "description": "Logo placement"}
  ],
  "analysis_notes": "Polaroid-style frame with space for a main photo and caption below",
  "suggested_content_types": ["announcement", "community"]
}

Return ONLY the JSON, no markdown formatting."""


async def analyze_template(image_path: str) -> dict:
    """Use Claude Vision to detect content regions in a template image."""
    import base64

    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": _TEMPLATE_ANALYSIS_PROMPT},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Template analysis returned non-JSON: %s", raw[:200])
        return {"regions": [], "analysis_notes": "", "suggested_content_types": []}


async def register_template(
    image_path: str,
    name: str,
    template_id: str | None = None,
) -> BrandTemplate:
    """Analyze an image and register it as a template.

    Opens the image with PIL for dimensions, calls Claude for region analysis,
    and saves to the manifest.
    """
    img = Image.open(image_path)
    width, height = img.size
    img.close()

    # Compute aspect ratio string
    if width and height:
        ratio = width / height
        if abs(ratio - 16 / 9) < 0.1:
            aspect_ratio = "16:9"
        elif abs(ratio - 9 / 16) < 0.1:
            aspect_ratio = "9:16"
        elif abs(ratio - 1.0) < 0.1:
            aspect_ratio = "1:1"
        elif abs(ratio - 4 / 3) < 0.1:
            aspect_ratio = "4:3"
        else:
            aspect_ratio = f"{width}:{height}"
    else:
        aspect_ratio = ""

    analysis = await analyze_template(image_path)

    regions = [
        TemplateRegion(**r) for r in analysis.get("regions", [])
    ]

    tid = template_id or str(uuid.uuid4())[:8]
    template = BrandTemplate(
        id=tid,
        name=name,
        path=image_path,
        width=width,
        height=height,
        regions=regions,
        aspect_ratio=aspect_ratio,
        content_types=[],  # Universal — matches all content types
        analysis_notes=analysis.get("analysis_notes", ""),
    )

    memory = TemplateMemory()
    memory.add_template(template)

    logger.info("Registered template '%s' (%dx%d, %d regions)", name, width, height, len(regions))
    return template


# ---------------------------------------------------------------------------
# Template application — PIL composition (Commit F)
# ---------------------------------------------------------------------------

async def apply_template(
    template: BrandTemplate,
    image_url: str,
    draft: dict,
) -> io.BytesIO | None:
    """Composite a generated image and text onto a template.

    1. Load template image from template.path
    2. Download generated image from image_url
    3. Find 'image' region → resize/crop to fit → paste
    4. Find 'text' regions → overlay draft title/subtitle
    5. Return as BytesIO PNG
    """
    # Find image region
    image_region = None
    text_regions = []
    for r in template.regions:
        if r.type == "image" and image_region is None:
            image_region = r
        elif r.type == "text":
            text_regions.append(r)

    if not image_region:
        return None

    try:
        tpl_img = Image.open(template.path).convert("RGBA")
    except Exception as e:
        logger.warning("Failed to open template image: %s", e)
        return None

    # Download generated image
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            gen_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as e:
        logger.warning("Failed to download generated image for template: %s", e)
        return None

    # Resize/crop generated image to fit image region
    target_w, target_h = image_region.width, image_region.height
    gen_img = _resize_crop(gen_img, target_w, target_h)
    tpl_img.paste(gen_img, (image_region.x, image_region.y))

    # Overlay text on text regions
    draw = ImageDraw.Draw(tpl_img)
    text_values = [draft.get("title", ""), draft.get("subtitle", "")]
    for i, region in enumerate(text_regions[:len(text_values)]):
        text = text_values[i]
        if not text:
            continue
        font = _get_text_font(region.height)
        draw.text(
            (region.x + 4, region.y + 2),
            text,
            fill=(255, 255, 255, 230),
            font=font,
        )

    buf = io.BytesIO()
    tpl_img.convert("RGB").save(buf, "PNG")
    buf.seek(0)
    return buf


def _resize_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize and center-crop an image to exactly target_w x target_h."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _get_text_font(region_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font sized for the region height, preferring brand fonts."""
    font_size = max(12, int(region_height * 0.6))
    # Try brand fonts first (same directory as compositor)
    fonts_dir = Path(settings.BRAND_FOLDER) / "assets" / "fonts"
    for candidate in ("Orbitron-Variable.ttf", "Inter-Variable.ttf"):
        font_path = fonts_dir / candidate
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except (OSError, IOError):
                continue
    # Fall back to system fonts
    for sys_font in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(sys_font, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Template-aware aspect ratio (Commit H)
# ---------------------------------------------------------------------------

def get_aspect_ratio_for_content_type(content_type: str) -> str | None:
    """Return the template's aspect_ratio if one matches the content type, else None."""
    memory = TemplateMemory()
    template = memory.get_template_for_content_type(content_type)
    if template and template.aspect_ratio:
        return template.aspect_ratio
    return None


# ---------------------------------------------------------------------------
# Template detection (Commit I)
# ---------------------------------------------------------------------------

_DETECT_PROMPT = """\
Does this image look like a design template, frame, or mockup? \
A template has clear empty/placeholder areas where content would be placed — \
like a phone screen mockup, polaroid frame, Game Boy screen, or bordered layout.

Answer with ONLY valid JSON:
{"is_template": true, "confidence": 0.9, "reason": "Has a clear phone screen cutout area"}

Return ONLY the JSON, no markdown formatting."""


async def detect_if_template(image_path: str) -> bool:
    """Lightweight Claude Vision check — is this image a template?

    Returns False on any error.
    """
    import base64

    try:
        path = Path(image_path)
        suffix = path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                    {"type": "text", "text": _DETECT_PROMPT},
                ],
            }],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        result = json.loads(raw)
        return bool(result.get("is_template", False))
    except Exception as e:
        logger.warning("Template detection failed: %s", e)
        return False
