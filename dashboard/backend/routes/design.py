"""API routes for the BrandMover Design Studio Mini App."""

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dashboard.backend.services import brand_bridge, design_bridge

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/design", tags=["design"])


# ---------------------------------------------------------------------------
# Brand data endpoints
# ---------------------------------------------------------------------------

@router.get("/brand-board")
async def get_brand_board():
    """Get full brand data for the visual board."""
    return brand_bridge.get_brand_board()


@router.get("/templates")
async def get_templates():
    """Get all available templates."""
    return {"templates": brand_bridge.get_templates_list()}


@router.get("/content-types")
async def get_content_types():
    """Get available content types."""
    return {"content_types": brand_bridge.get_content_types_list()}


@router.get("/history")
async def get_history(limit: int = 20):
    """Get recent generation history."""
    return {"history": brand_bridge.get_generation_history(limit)}


# ---------------------------------------------------------------------------
# Design Agent chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    session_uploads: list[dict] | None = None


@router.post("/chat")
async def design_agent_chat(req: ChatRequest):
    """Chat with the Design Agent for spec refinement."""
    brand_data = brand_bridge.get_brand_board()
    brand_context = _format_brand_context(brand_data)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    response = await design_bridge.design_agent_chat(
        messages=messages,
        brand_context=brand_context,
        session_uploads=req.session_uploads,
    )
    return {"response": response}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class DesignSpec(BaseModel):
    content_type: str | None = None
    template_id: str | None = None
    title: str | None = None
    subtitle: str | None = None
    caption_guidance: str | None = None
    image_prompt: str | None = None
    color_overrides: dict | None = None
    layout_preset: str | None = None
    style_notes: str | None = None


@router.post("/generate")
async def generate_design(spec: DesignSpec):
    """Generate content from a design spec. Returns SSE stream."""
    async def event_stream():
        async for event in design_bridge.run_design_generation(
            spec.model_dump(exclude_none=True),
            user_id=0,  # Will be set from auth context
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Reference image analysis
# ---------------------------------------------------------------------------

@router.post("/analyze-reference")
async def analyze_reference(file: UploadFile = File(...)):
    """Upload and analyze a reference image."""
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        analysis = await brand_bridge.analyze_reference_image(tmp_path)
        return {"analysis": analysis, "filename": file.filename}
    except Exception as e:
        logger.error("Reference analysis failed: %s", e)
        return {"error": str(e)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_brand_context(brand_data: dict) -> str:
    """Format brand data as a compact context string for the Design Agent."""
    lines = [f"Brand: {brand_data.get('brand_name', 'Unknown')}"]

    if brand_data.get("tagline"):
        lines.append(f"Tagline: {brand_data['tagline']}")

    # Colors
    colors = brand_data.get("colors", {})
    if colors:
        color_strs = [
            f"{role}: {c.get('name', '')} ({c.get('hex', '')})"
            for role, c in colors.items()
        ]
        lines.append(f"Colors: {', '.join(color_strs)}")

    # Fonts
    fonts = brand_data.get("fonts", {})
    if fonts:
        font_strs = [
            f"{use}: {f.get('family', '')} {f.get('weight', '')}"
            for use, f in fonts.items()
        ]
        lines.append(f"Fonts: {', '.join(font_strs)}")

    # Style keywords
    keywords = brand_data.get("style_keywords", [])
    if keywords:
        lines.append(f"Style: {', '.join(keywords)}")

    # Avoid terms
    avoid = brand_data.get("avoid_terms", [])
    if avoid:
        lines.append(f"Avoid: {', '.join(avoid)}")

    # Content types
    content_types = brand_data.get("content_types", [])
    if content_types:
        ct_strs = [ct.get("id", "") for ct in content_types]
        lines.append(f"Content types: {', '.join(ct_strs)}")

    # Templates
    templates = brand_data.get("templates", [])
    if templates:
        tpl_strs = [
            f"{t.get('name', 'unnamed')} ({t.get('aspect_ratio', '?')})"
            for t in templates
        ]
        lines.append(f"Templates: {', '.join(tpl_strs)}")

    return "\n".join(lines)
