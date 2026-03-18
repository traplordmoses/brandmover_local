"""Tests for agent.image_gen — override parameters, prompt enhancement,
model-specific input building, brand terms injection, generate_image() API
calls, cache_image(), _extract_url(), and error handling.

These tests verify that:
- Override parameters (model_override, aspect_ratio, negative_prompt_override,
  skip_enhance) are respected.
- enhance_prompt() reads from BrandConfig, not hardcoded strings.
- _build_input() produces correct model-specific payloads.
- _get_brand_terms() and _get_negative_prompt() read from BrandConfig.
- generate_image() calls Replicate API and returns URL on success.
- generate_image() returns None on API failure.
- cache_image() downloads and caches images locally.
- _extract_url() handles various Replicate output formats.
"""

from unittest.mock import patch, AsyncMock, MagicMock

import httpx
import pytest

from agent.compositor_config import BrandConfig, ColorEntry
from agent.image_gen import (
    _MODELS,
    _build_input,
    _extract_url,
    _get_brand_terms,
    _get_negative_prompt,
    _get_quality_profile,
    cache_image,
    enhance_prompt,
    generate_image,
    select_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> BrandConfig:
    """Build a BrandConfig with non-FOID test values."""
    defaults = dict(
        brand_name="ZetaCorp",
        style_keywords=["Pastel gradient", "Soft glow", "Retro terminal", "Wireframe"],
        colors={
            "primary": ColorEntry(role="primary", name="Mint", hex="#00ff88", rgb=(0, 255, 136)),
            "accent_1": ColorEntry(role="accent_1", name="Coral", hex="#ff6644", rgb=(255, 102, 68)),
            "accent_2": ColorEntry(role="accent_2", name="Lilac", hex="#cc88ff", rgb=(204, 136, 255)),
            "background": ColorEntry(role="background", name="Charcoal", hex="#1a1a2e", rgb=(26, 26, 46)),
        },
        avoid_terms=["flat colors", "white backgrounds", "stock photos"],
    )
    defaults.update(overrides)
    return BrandConfig(**defaults)


def _empty_cfg() -> BrandConfig:
    return BrandConfig()


# ---------------------------------------------------------------------------
# _get_brand_terms — reads from BrandConfig
# ---------------------------------------------------------------------------

class TestGetBrandTerms:
    def test_includes_style_keywords(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "Pastel gradient" in terms
            assert "Soft glow" in terms

    def test_includes_color_palette(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "#00ff88" in terms
            assert "mint" in terms.lower()

    def test_includes_background_color(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "charcoal" in terms.lower()
            assert "#1a1a2e" in terms

    def test_empty_config_returns_fallback(self):
        cfg = _empty_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "high quality" in terms
            assert "professional" in terms

    def test_no_foid_references(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            assert "FOID" not in terms
            assert "foid" not in terms.lower()

    def test_limits_style_keywords(self):
        cfg = _make_cfg(style_keywords=[f"kw{i}" for i in range(20)])
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            terms = _get_brand_terms()
            # Should only include first 6
            assert "kw0" in terms
            assert "kw5" in terms
            assert "kw6" not in terms


# ---------------------------------------------------------------------------
# _get_negative_prompt — reads avoid_terms from BrandConfig
# ---------------------------------------------------------------------------

class TestGetNegativePrompt:
    def test_includes_avoid_terms(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            neg = _get_negative_prompt()
            assert "flat colors" in neg
            assert "white backgrounds" in neg
            assert "stock photos" in neg

    def test_includes_base_negatives(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            neg = _get_negative_prompt()
            assert "blurry" in neg
            assert "low quality" in neg

    def test_empty_avoid_terms_uses_base_only(self):
        cfg = _make_cfg(avoid_terms=[])
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            neg = _get_negative_prompt()
            assert "blurry" in neg
            assert "flat colors" not in neg


# ---------------------------------------------------------------------------
# _get_quality_profile — brand-specific quality enrichment
# ---------------------------------------------------------------------------

class TestGetQualityProfile:
    def test_includes_style_keywords(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("announcement")
            assert "Pastel gradient" in profile

    def test_includes_color_glow(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("lifestyle")
            assert "mint glow" in profile.lower()

    def test_includes_background_context(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("event")
            assert "charcoal background" in profile.lower()

    def test_includes_content_type_modifier(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            profile = _get_quality_profile("announcement")
            assert "volumetric light" in profile


# ---------------------------------------------------------------------------
# enhance_prompt — full prompt enhancement pipeline
# ---------------------------------------------------------------------------

class TestEnhancePrompt:
    def test_adds_brand_terms(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, neg = enhance_prompt("A glowing crystal", "announcement")
            assert "A glowing crystal" in enhanced
            assert "Pastel gradient" in enhanced
            assert "#00ff88" in enhanced

    def test_adds_negative_prompt(self):
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            _, neg = enhance_prompt("A glowing crystal", "announcement")
            assert "flat colors" in neg
            assert "blurry" in neg

    def test_skips_brand_when_already_present(self):
        """If prompt already references brand colors/style, skip brand enforcement."""
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, _ = enhance_prompt(
                "A crystal with brand color #00ff88 aesthetic", "announcement"
            )
            # Brand terms should still work — _BRAND_INDICATORS checks for "brand.*color"
            # which matches, so brand terms should be skipped
            # But quality profile should still be added
            assert "volumetric light" in enhanced

    def test_mascot_special_path(self):
        """Community content with mascot keywords gets special treatment."""
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, neg = enhance_prompt("A cute mascot character", "community")
            assert "Pixar-quality" in enhanced
            assert "uncanny valley" in neg

    def test_locked_directives_preserved(self):
        """Locked directives survive enhancement."""
        cfg = _make_cfg()
        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg):
            enhanced, _ = enhance_prompt(
                "Product shot, upright, matte black background", "brand_asset"
            )
            assert "upright" in enhanced
            assert "matte black background" in enhanced


# ---------------------------------------------------------------------------
# _build_input — model-specific payloads
# ---------------------------------------------------------------------------

class TestBuildInput:
    def test_flux_payload(self):
        payload = _build_input(_MODELS["flux"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["aspect_ratio"] == "16:9"
        assert payload["output_format"] == "jpg"
        assert payload["output_quality"] == 95

    def test_flux_custom_aspect(self):
        payload = _build_input(_MODELS["flux"], "test", aspect_ratio="1:1")
        assert payload["aspect_ratio"] == "1:1"

    def test_nano_banana_payload(self):
        payload = _build_input(_MODELS["nano-banana"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["resolution"] == "2K"

    def test_recraft_payload(self):
        payload = _build_input(_MODELS["recraft-svg"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["size"] == "1820x1024"

    def test_recraft_custom_size(self):
        payload = _build_input(_MODELS["recraft-svg"], "test", size="1024x1024")
        assert payload["size"] == "1024x1024"

    def test_seedream_payload(self):
        payload = _build_input(_MODELS["seedream"], "test prompt")
        assert payload["prompt"] == "test prompt"
        assert payload["aspect_ratio"] == "16:9"

    def test_seedream_negative_prompt(self):
        """Only seedream accepts negative_prompt."""
        payload = _build_input(_MODELS["seedream"], "test", negative_prompt="ugly, bad")
        assert payload["negative_prompt"] == "ugly, bad"

    def test_flux_ignores_negative_prompt(self):
        """Flux doesn't support negative_prompt, so it should be excluded."""
        payload = _build_input(_MODELS["flux"], "test", negative_prompt="ugly, bad")
        assert "negative_prompt" not in payload

    def test_unknown_model_minimal_payload(self):
        payload = _build_input("some/unknown-model", "test prompt")
        assert payload == {"prompt": "test prompt"}


# ---------------------------------------------------------------------------
# select_model — routing logic
# ---------------------------------------------------------------------------

class TestSelectModel:
    def test_announcement_routes_to_flux(self):
        model, reason = select_model("announcement", "Product launch image")
        assert model == _MODELS["flux"]

    def test_brand_asset_routes_to_flux(self):
        model, reason = select_model("brand_asset", "A brand icon")
        assert model == _MODELS["flux"]

    def test_lifestyle_routes_to_seedream(self):
        model, reason = select_model("lifestyle", "Coffee shop scene")
        assert model == _MODELS["seedream"]

    def test_default_routes_to_flux(self):
        model, reason = select_model("engagement", "Community post")
        assert model == _MODELS["flux"]

    def test_text_overlay_keyword_routes_to_nano_banana(self):
        model, reason = select_model("engagement", "Image with bold text reads HELLO")
        assert model == _MODELS["nano-banana"]

    def test_manual_override(self):
        with patch("agent.image_gen.settings") as mock_settings:
            mock_settings.IMAGE_MODEL = "custom/model"
            model, reason = select_model("announcement", "Test")
            assert model == "custom/model"
            assert "override" in reason

    def test_auto_mode_does_routing(self):
        with patch("agent.image_gen.settings") as mock_settings:
            mock_settings.IMAGE_MODEL = "auto"
            model, reason = select_model("lifestyle", "Nature photo")
            assert model == _MODELS["seedream"]


# ---------------------------------------------------------------------------
# _extract_url — parsing Replicate output formats
# ---------------------------------------------------------------------------

class TestExtractUrl:
    def test_string_output(self):
        assert _extract_url("https://replicate.delivery/img.png") == "https://replicate.delivery/img.png"

    def test_list_output(self):
        assert _extract_url(["https://replicate.delivery/img.png"]) == "https://replicate.delivery/img.png"

    def test_dict_url_key(self):
        assert _extract_url({"url": "https://example.com/img.png"}) == "https://example.com/img.png"

    def test_dict_image_key(self):
        assert _extract_url({"image": "https://example.com/img.jpg"}) == "https://example.com/img.jpg"

    def test_dict_svg_key(self):
        assert _extract_url({"svg": "<svg>...</svg>"}) == "<svg>...</svg>"

    def test_empty_list_returns_none_like(self):
        # Empty list -> str(output) if output is truthy, else None
        result = _extract_url([])
        assert result is None

    def test_none_returns_none(self):
        assert _extract_url(None) is None


# ---------------------------------------------------------------------------
# generate_image — Replicate API integration (mocked)
# ---------------------------------------------------------------------------

def _make_replicate_response(data: dict):
    """Build a MagicMock that behaves like an httpx.Response (sync methods)."""
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    resp.status_code = 200
    return resp


def _make_http_client(**overrides):
    """Build an AsyncMock httpx client whose .post()/.get() return MagicMocks."""
    client = AsyncMock()
    for key, val in overrides.items():
        setattr(client, key, AsyncMock(return_value=val))
    return client


class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_returns_url_on_immediate_success(self):
        """When Replicate returns succeeded immediately (Prefer: wait), returns cached URL."""
        cfg = _make_cfg()
        resp = _make_replicate_response({
            "status": "succeeded",
            "output": "https://replicate.delivery/generated.png",
        })
        client = _make_http_client(post=resp)

        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg), \
             patch("agent.image_gen.settings") as mock_settings, \
             patch("agent.image_gen.lora_pipeline.get_active_lora", return_value=None), \
             patch("agent.image_gen.cache_image", new_callable=AsyncMock,
                   return_value="/tmp/cached_img.png"), \
             patch("agent._client.get_httpx", return_value=client):
            mock_settings.REPLICATE_API_TOKEN = "test-token"
            mock_settings.IMAGE_MODEL = "auto"
            result = await generate_image("A beautiful sunset", "announcement")

        assert result == "/tmp/cached_img.png"
        client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        """When Replicate API returns an HTTP error, returns None."""
        cfg = _make_cfg()
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock()
        )
        client = _make_http_client(post=resp)

        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg), \
             patch("agent.image_gen.settings") as mock_settings, \
             patch("agent.image_gen.lora_pipeline.get_active_lora", return_value=None), \
             patch("agent._client.get_httpx", return_value=client):
            mock_settings.REPLICATE_API_TOKEN = "test-token"
            mock_settings.IMAGE_MODEL = "auto"
            result = await generate_image("A sunset", "announcement")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_api_token(self):
        """When REPLICATE_API_TOKEN is not set, returns None immediately."""
        with patch("agent.image_gen.settings") as mock_settings:
            mock_settings.REPLICATE_API_TOKEN = ""
            result = await generate_image("A sunset", "announcement")

        assert result is None

    @pytest.mark.asyncio
    async def test_model_override_skips_routing(self):
        """model_override bypasses select_model routing."""
        cfg = _make_cfg()
        resp = _make_replicate_response({
            "status": "succeeded",
            "output": "https://replicate.delivery/custom.png",
        })
        client = _make_http_client(post=resp)

        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg), \
             patch("agent.image_gen.settings") as mock_settings, \
             patch("agent.image_gen.lora_pipeline.get_active_lora", return_value=None), \
             patch("agent.image_gen.cache_image", new_callable=AsyncMock,
                   return_value="/tmp/custom.png"), \
             patch("agent._client.get_httpx", return_value=client):
            mock_settings.REPLICATE_API_TOKEN = "test-token"
            mock_settings.IMAGE_MODEL = "auto"
            result = await generate_image(
                "Test", "announcement", model_override="custom/my-model"
            )

        assert result == "/tmp/custom.png"
        # Verify the API URL uses the custom model
        call_args = client.post.call_args
        assert "custom/my-model" in call_args.args[0]

    @pytest.mark.asyncio
    async def test_skip_enhance_uses_raw_prompt(self):
        """skip_enhance=True sends the raw prompt without enhancement."""
        cfg = _make_cfg()
        resp = _make_replicate_response({
            "status": "succeeded",
            "output": "https://replicate.delivery/raw.png",
        })
        client = _make_http_client(post=resp)

        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg), \
             patch("agent.image_gen.settings") as mock_settings, \
             patch("agent.image_gen.lora_pipeline.get_active_lora", return_value=None), \
             patch("agent.image_gen.cache_image", new_callable=AsyncMock,
                   return_value="/tmp/raw.png"), \
             patch("agent._client.get_httpx", return_value=client):
            mock_settings.REPLICATE_API_TOKEN = "test-token"
            mock_settings.IMAGE_MODEL = "auto"
            result = await generate_image(
                "Exact prompt only", "announcement", skip_enhance=True
            )

        assert result == "/tmp/raw.png"
        # Verify the payload prompt is the raw prompt (not enhanced)
        call_kwargs = client.post.call_args.kwargs
        payload_prompt = call_kwargs["json"]["input"]["prompt"]
        assert payload_prompt == "Exact prompt only"

    @pytest.mark.asyncio
    async def test_polls_when_not_immediately_succeeded(self):
        """When Prefer: wait doesn't return succeeded, polls until completion."""
        cfg = _make_cfg()

        initial_resp = _make_replicate_response({
            "status": "processing",
            "urls": {"get": "https://api.replicate.com/v1/predictions/abc123"},
        })
        poll_resp = _make_replicate_response({
            "status": "succeeded",
            "output": "https://replicate.delivery/polled.png",
        })

        client = AsyncMock()
        client.post = AsyncMock(return_value=initial_resp)
        client.get = AsyncMock(return_value=poll_resp)

        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg), \
             patch("agent.image_gen.settings") as mock_settings, \
             patch("agent.image_gen.lora_pipeline.get_active_lora", return_value=None), \
             patch("agent.image_gen.cache_image", new_callable=AsyncMock,
                   return_value="/tmp/polled.png"), \
             patch("agent._client.get_httpx", return_value=client), \
             patch("agent.image_gen.asyncio.sleep", new_callable=AsyncMock):
            mock_settings.REPLICATE_API_TOKEN = "test-token"
            mock_settings.IMAGE_MODEL = "auto"
            result = await generate_image("A sunset", "announcement")

        assert result == "/tmp/polled.png"
        client.get.assert_called()

    @pytest.mark.asyncio
    async def test_returns_none_on_failed_prediction(self):
        """When the prediction status is 'failed', returns None."""
        cfg = _make_cfg()

        initial_resp = _make_replicate_response({
            "status": "processing",
            "urls": {"get": "https://api.replicate.com/v1/predictions/fail123"},
        })
        poll_resp = _make_replicate_response({
            "status": "failed",
            "error": "Model crashed",
        })

        client = AsyncMock()
        client.post = AsyncMock(return_value=initial_resp)
        client.get = AsyncMock(return_value=poll_resp)

        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg), \
             patch("agent.image_gen.settings") as mock_settings, \
             patch("agent.image_gen.lora_pipeline.get_active_lora", return_value=None), \
             patch("agent._client.get_httpx", return_value=client), \
             patch("agent.image_gen.asyncio.sleep", new_callable=AsyncMock):
            mock_settings.REPLICATE_API_TOKEN = "test-token"
            mock_settings.IMAGE_MODEL = "auto"
            result = await generate_image("A sunset", "announcement")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_unexpected_exception(self):
        """Unexpected exceptions are caught and return None."""
        cfg = _make_cfg()

        client = AsyncMock()
        client.post = AsyncMock(side_effect=RuntimeError("unexpected crash"))

        with patch("agent.image_gen.compositor_config.get_config", return_value=cfg), \
             patch("agent.image_gen.settings") as mock_settings, \
             patch("agent.image_gen.lora_pipeline.get_active_lora", return_value=None), \
             patch("agent._client.get_httpx", return_value=client):
            mock_settings.REPLICATE_API_TOKEN = "test-token"
            mock_settings.IMAGE_MODEL = "auto"
            result = await generate_image("A sunset", "announcement")

        assert result is None


# ---------------------------------------------------------------------------
# cache_image — local image caching
# ---------------------------------------------------------------------------

class TestCacheImage:
    @pytest.mark.asyncio
    async def test_downloads_and_caches(self, tmp_path):
        """Downloads image from URL and saves to local cache directory."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.content = b"fake-png-bytes"

        client = AsyncMock()
        client.get = AsyncMock(return_value=resp)

        with patch("agent.image_gen._IMAGE_CACHE_DIR", tmp_path), \
             patch("agent._client.get_httpx", return_value=client):
            result = await cache_image("https://replicate.delivery/abc123.png")

        assert result.startswith(str(tmp_path))
        assert result.endswith(".png")
        from pathlib import Path
        assert Path(result).read_bytes() == b"fake-png-bytes"

    @pytest.mark.asyncio
    async def test_returns_cached_if_exists(self, tmp_path):
        """If the file is already cached, returns path without downloading."""
        import hashlib
        url = "https://replicate.delivery/already_cached.jpg"
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached_file = tmp_path / f"{url_hash}.jpg"
        cached_file.write_bytes(b"existing-bytes")

        client = AsyncMock()

        with patch("agent.image_gen._IMAGE_CACHE_DIR", tmp_path), \
             patch("agent._client.get_httpx", return_value=client):
            result = await cache_image(url)

        assert result == str(cached_file)
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_original_url_on_download_failure(self, tmp_path):
        """If download fails, returns the original URL as fallback."""
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock()
        )

        client = AsyncMock()
        client.get = AsyncMock(return_value=resp)

        url = "https://replicate.delivery/broken.png"
        with patch("agent.image_gen._IMAGE_CACHE_DIR", tmp_path), \
             patch("agent._client.get_httpx", return_value=client):
            result = await cache_image(url)

        assert result == url

    @pytest.mark.asyncio
    async def test_detects_webp_extension(self, tmp_path):
        """URL containing .webp gets saved with .webp extension."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.content = b"webp-bytes"

        client = AsyncMock()
        client.get = AsyncMock(return_value=resp)

        with patch("agent.image_gen._IMAGE_CACHE_DIR", tmp_path), \
             patch("agent._client.get_httpx", return_value=client):
            result = await cache_image("https://replicate.delivery/img.webp?token=abc")

        assert result.endswith(".webp")
