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

    def update_template_regions(
        self, template_id: str, regions: list[TemplateRegion],
    ) -> BrandTemplate | None:
        """Overwrite a template's regions in the manifest. Returns updated template or None."""
        manifest = self.load_manifest()
        for entry in manifest:
            if entry.get("id") == template_id:
                entry["regions"] = [
                    {
                        "type": r.type,
                        "x": r.x,
                        "y": r.y,
                        "width": r.width,
                        "height": r.height,
                        "description": r.description,
                    }
                    for r in regions
                ]
                self.save_manifest()
                # Return the updated template
                return BrandTemplate(
                    id=entry.get("id", ""),
                    name=entry.get("name", ""),
                    path=entry.get("path", ""),
                    width=entry.get("width", 0),
                    height=entry.get("height", 0),
                    regions=regions,
                    aspect_ratio=entry.get("aspect_ratio", ""),
                    content_types=entry.get("content_types", []),
                    analysis_notes=entry.get("analysis_notes", ""),
                )
        return None

    def get_template_for_content_type(self, content_type: str) -> BrandTemplate | None:
        templates = self.list_templates()
        # 1. Exact content_type match
        for t in templates:
            if content_type in t.content_types:
                return t
        # 2. Universal templates — prefer matching aspect ratio
        universals = [t for t in templates if not t.content_types]
        if not universals:
            return None
        preferred_ratio = _CONTENT_TYPE_ASPECT.get(content_type, "16:9")
        for t in universals:
            if t.aspect_ratio == preferred_ratio:
                return t
        return universals[0]  # fallback to first universal


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
# Natural language region parsing
# ---------------------------------------------------------------------------

_REGION_PARSE_PROMPT = """\
You are converting a natural language description of template regions into pixel coordinates.

The template canvas is {width}x{height} pixels.

The user says:
"{description}"

Convert this into exact pixel coordinates for each region mentioned. Each region is one of:
- "image": where the generated image goes
- "text": where text overlay goes (title is the first text region, subtitle is the second)
- "logo": where the brand logo goes

Return ONLY valid JSON:
{{
  "regions": [
    {{"type": "image", "x": 0, "y": 0, "width": 1200, "height": 700, "description": "Full canvas background image"}},
    {{"type": "text", "x": 0, "y": 0, "width": 1200, "height": 105, "description": "Top text band"}},
    {{"type": "text", "x": 0, "y": 595, "width": 1200, "height": 105, "description": "Bottom text band"}}
  ]
}}

Rules:
- "full canvas" or "entire background" for image means x=0, y=0, width={width}, height={height}
- Percentages are relative to the canvas: "top 15%" means height = {height} * 0.15, y = 0
- "bottom 15%" means y = {height} - ({height} * 0.15), height = {height} * 0.15
- "full width" means x=0, width={width}
- "centered" means center the region within its described area
- Round all values to integers

Return ONLY the JSON, no markdown formatting."""


async def parse_region_description(
    description: str, width: int, height: int,
) -> list[TemplateRegion]:
    """Use Claude to convert a natural language region description into pixel coordinates."""
    prompt = _REGION_PARSE_PROMPT.format(
        width=width, height=height, description=description,
    )

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        return [TemplateRegion(**r) for r in data.get("regions", [])]
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Region parse failed: %s — raw: %s", e, raw[:200])
        return []


# ---------------------------------------------------------------------------
# Template application — PIL composition (Commit F)
# ---------------------------------------------------------------------------

async def apply_template(
    template: BrandTemplate,
    image_source: str,
    draft: dict,
) -> io.BytesIO | None:
    """Composite a generated image and text onto a template using alpha compositing.

    Uses alpha_composite layering so the template's transparent areas reveal the
    generated image beneath, while opaque frame regions sit on top.

    Args:
        template: The brand template to apply.
        image_source: URL (http/https) or local file path of the source image.
        draft: Dict with 'title' and/or 'subtitle' for text regions.

    Returns:
        BytesIO PNG of the composed image, or None on failure.
    """
    # Classify regions
    image_region = None
    text_regions = []
    logo_region = None
    for r in template.regions:
        if r.type == "image" and image_region is None:
            image_region = r
        elif r.type == "text":
            text_regions.append(r)
        elif r.type == "logo" and logo_region is None:
            logo_region = r

    if not image_region:
        return None

    # 1. Open template as RGBA
    try:
        tpl_img = Image.open(template.path).convert("RGBA")
    except Exception as e:
        logger.warning("Failed to open template image: %s", e)
        return None

    # 2. Load source image (URL or local path)
    try:
        if image_source.startswith("http"):
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(image_source)
                resp.raise_for_status()
                gen_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        else:
            gen_img = Image.open(image_source).convert("RGBA")
    except Exception as e:
        logger.warning("Failed to load source image for template: %s", e)
        return None

    # 3. Alpha-composite layering: image → canvas → template on top
    canvas = Image.new("RGBA", tpl_img.size, (0, 0, 0, 0))
    target_w, target_h = image_region.width, image_region.height
    gen_img = _resize_crop(gen_img, target_w, target_h)
    canvas.paste(gen_img, (image_region.x, image_region.y))
    canvas = Image.alpha_composite(canvas, tpl_img)

    # 4. Place logo if region exists
    if logo_region:
        _place_logo(canvas, logo_region)

    # 5. Draw text on text regions with smart fitting
    # Sort text regions by y-coordinate so top region gets title, bottom gets subtitle
    text_regions.sort(key=lambda r: r.y)
    text_values = [draft.get("title", ""), draft.get("subtitle", "")]
    font_names = ["title", "subtitle"]
    for i, region in enumerate(text_regions[:len(text_values)]):
        text = text_values[i]
        if not text:
            continue
        font_role = font_names[i] if i < len(font_names) else "subtitle"
        _draw_fitted_text(canvas, text, region, font_role)

    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, "PNG")
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


def _get_text_font(font_size: int, role: str = "title") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font at the given size, selecting by role.

    role='title' prefers Orbitron, role='subtitle' prefers Inter.
    """
    fonts_dir = Path(settings.BRAND_FOLDER) / "assets" / "fonts"
    if role == "title":
        candidates = ("Orbitron-Variable.ttf", "Inter-Variable.ttf")
    else:
        candidates = ("Inter-Variable.ttf", "Orbitron-Variable.ttf")
    for candidate in candidates:
        font_path = fonts_dir / candidate
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except (OSError, IOError):
                continue
    for sys_font in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(sys_font, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> list[str]:
    """Word-wrap text into lines that each fit within max_width pixels."""
    words = text.split()
    if not words:
        return []
    lines = []
    current_line = words[0]
    for word in words[1:]:
        test = f"{current_line} {word}"
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def _draw_fitted_text(
    canvas: Image.Image,
    text: str,
    region: TemplateRegion,
    role: str = "title",
) -> None:
    """Draw text fitted into a region with word wrap, size fitting, and shadow.

    Binary-searches from a large font size down to 16px to find the largest
    size where the word-wrapped text fits within the region.
    """
    draw = ImageDraw.Draw(canvas)
    padding = 8
    max_w = region.width - 2 * padding
    max_h = region.height - 2 * padding
    if max_w <= 0 or max_h <= 0:
        return

    # Binary search for best font size
    lo, hi = 16, max(16, int(region.height * 0.7))
    best_size = lo
    best_lines: list[str] = [text]

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_text_font(mid, role)
        lines = _fit_text(draw, text, font, max_w)
        # Measure total height
        line_height = draw.textbbox((0, 0), "Ay", font=font)[3]
        total_h = line_height * len(lines)
        if total_h <= max_h:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    font = _get_text_font(best_size, role)
    lines = _fit_text(draw, text, font, max_w)
    line_height = draw.textbbox((0, 0), "Ay", font=font)[3]
    total_h = line_height * len(lines)

    # Vertical centering
    y_offset = region.y + (region.height - total_h) // 2

    # Outline thickness scales with font size — thicker for large meme text
    outline_w = max(2, best_size // 15)

    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_w = line_bbox[2] - line_bbox[0]
        x_offset = region.x + (region.width - line_w) // 2  # center horizontally
        # Dark outline for readability over any background (no scrim needed)
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text(
                    (x_offset + dx, y_offset + dy), line,
                    fill=(0, 0, 0, 220), font=font,
                )
        # Main text on top
        draw.text((x_offset, y_offset), line, fill=(255, 255, 255, 255), font=font)
        y_offset += line_height


def _place_logo(canvas: Image.Image, region: TemplateRegion) -> None:
    """Place brand logo inside a logo region, proportionally scaled and centered."""
    logo_path = Path(settings.BRAND_FOLDER) / "assets" / "logo.png"
    if not logo_path.exists():
        logger.debug("No logo.png found at %s — skipping logo placement", logo_path)
        return
    try:
        logo = Image.open(logo_path).convert("RGBA")
        # Scale proportionally to fit inside the region
        scale = min(region.width / logo.width, region.height / logo.height)
        new_w = int(logo.width * scale)
        new_h = int(logo.height * scale)
        logo = logo.resize((new_w, new_h), Image.LANCZOS)
        # Center within region
        x = region.x + (region.width - new_w) // 2
        y = region.y + (region.height - new_h) // 2
        canvas.paste(logo, (x, y), logo)
    except Exception as e:
        logger.warning("Logo placement failed: %s", e)


# ---------------------------------------------------------------------------
# Content type → preferred aspect ratio mapping
# ---------------------------------------------------------------------------

_CONTENT_TYPE_ASPECT = {
    "announcement": "16:9",
    "campaign": "16:9",
    "market": "16:9",
    "meme": "1:1",
    "engagement": "1:1",
    "community": "1:1",
    "lifestyle": "16:9",
    "event": "16:9",
    "brand_asset": "1:1",
}


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


def get_image_region_aspect_ratio(content_type: str) -> str | None:
    """Return the aspect ratio of the image region in the matching template.

    Unlike get_aspect_ratio_for_content_type which returns the overall template
    aspect ratio, this returns the aspect ratio of the image region within the
    template — what the generated image should match.
    """
    memory = TemplateMemory()
    template = memory.get_template_for_content_type(content_type)
    if not template:
        return None
    for r in template.regions:
        if r.type == "image" and r.width and r.height:
            ratio = r.width / r.height
            if abs(ratio - 1.0) < 0.1:
                return "1:1"
            if abs(ratio - 16 / 9) < 0.1:
                return "16:9"
            if abs(ratio - 9 / 16) < 0.1:
                return "9:16"
            if abs(ratio - 4 / 3) < 0.1:
                return "4:3"
            return f"{r.width}:{r.height}"
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
