"""Tests for audit-driven guidelines generation, brand_check inventory context, and /regen_guidelines."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(
    brand_name="TestBrand",
    description="A test product",
    platforms=None,
    audit=None,
    visual_prefs=None,
):
    from agent.onboarding import OnboardingSession

    return OnboardingSession(
        user_id=1,
        brand_name=brand_name,
        description=description,
        platforms=platforms or ["x"],
        asset_audit=audit or {},
        visual_preferences=visual_prefs or {},
    )


def _mock_strategy(compositor_enabled=False, badge_text=None, default_mode="image_optional"):
    from agent.strategy import StrategyRecommendation

    return StrategyRecommendation(
        archetype="has_identity",
        compositor_enabled=compositor_enabled,
        badge_text=badge_text,
        default_mode=default_mode,
        recommended_content_types=["announcement", "community"],
        platforms=["x"],
    )


def _sample_audit_data():
    return {
        "archetype": "has_identity",
        "consolidated_colors": [
            {"hex": "#ff5500", "name": "Flame Orange", "role": "primary"},
            {"hex": "#1a1a2e", "name": "Deep Navy", "role": "neutral"},
            {"hex": "#00d4ff", "name": "Cyan", "role": "accent"},
        ],
        "consolidated_style": ["bold", "geometric", "modern", "tech-forward"],
        "missing_items": ["font_specimen"],
        "entry_count": 5,
        "collection_analysis": {
            "coherence": "high",
            "visual_language": "geometric shapes with neon accents on dark backgrounds",
        },
        "brand_insights": {
            "personality": ["confident", "innovative", "approachable"],
            "audience": "tech-savvy creatives",
            "tone": "casual-professional",
        },
    }


def _mock_update(user_id=123, text=""):
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    update.message.text = text
    return update


def _mock_context():
    ctx = MagicMock()
    ctx.user_data = {}
    ctx.bot = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# Change 1: Audit data passthrough
# ---------------------------------------------------------------------------

class TestAuditDataPassthrough:
    def test_run_onboarding_audit_includes_collection_analysis(self):
        """_run_onboarding_audit should pass collection_analysis and brand_insights to session."""
        import tempfile, os
        from bot.handlers import _run_onboarding_audit
        from agent.asset_audit import AssetInventory, AssetAuditEntry

        async def _run():
            # Create a real temp file so Path.exists() works
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
                f.write(b"fake image data")

            try:
                inventory = AssetInventory(
                    entries=[AssetAuditEntry(path=tmp_path, category="logo")],
                    consolidated_colors=[{"hex": "#ff0000", "name": "Red", "role": "primary"}],
                    consolidated_style=["bold"],
                    archetype="has_identity",
                    collection_analysis={"coherence": "high"},
                    brand_insights={"personality": ["bold"]},
                )

                # Capture the audit_data passed to finalize_audit
                captured = {}
                def capture_finalize(s, data):
                    captured.update(data)
                    return s, "done"

                session = MagicMock()
                session.uploaded_assets = [{"path": tmp_path, "type": "reference"}]

                with patch("agent.asset_audit.audit_batch", AsyncMock(return_value=inventory)), \
                     patch("agent.asset_audit.save_inventory"), \
                     patch("agent.onboarding.finalize_audit", side_effect=capture_finalize), \
                     patch("agent.onboarding.save_session"):
                    update = _mock_update()
                    await _run_onboarding_audit(update, session)

                return captured
            finally:
                os.unlink(tmp_path)

        result = asyncio.run(_run())
        assert "collection_analysis" in result
        assert result["collection_analysis"] == {"coherence": "high"}
        assert "brand_insights" in result
        assert result["brand_insights"] == {"personality": ["bold"]}


# ---------------------------------------------------------------------------
# Change 2: Guidelines generation from audit
# ---------------------------------------------------------------------------

class TestGenerateGuidelinesFromAudit:
    def test_uses_audit_colors(self):
        """Generated guidelines should contain hex values from audit data."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            session = _mock_session(audit=_sample_audit_data())
            rec = _mock_strategy()

            # Mock Claude to return guidelines containing the audit colors
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=(
                "# TestBrand Brand Guidelines\n\n"
                "## COLOR PALETTE\n\n"
                "| Role | Name | Hex | RGB |\n"
                "|------|------|-----|-----|\n"
                "| Primary | Flame Orange | #ff5500 | (255, 85, 0) |\n"
                "| Neutral | Deep Navy | #1a1a2e | (26, 26, 46) |\n"
                "| Accent | Cyan | #00d4ff | (0, 212, 255) |\n"
            ))]

            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await generate_guidelines_from_audit(session, rec)

        result = asyncio.run(_run())
        assert "#ff5500" in result
        assert "#1a1a2e" in result
        assert "#00d4ff" in result

    def test_uses_style_keywords(self):
        """Generated guidelines should reference audit style keywords."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            session = _mock_session(audit=_sample_audit_data())
            rec = _mock_strategy()

            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=(
                "# TestBrand Brand Guidelines\n\n"
                "## ILLUSTRATION STYLE\n\n"
                "- **Bold, geometric** aesthetic with modern tech-forward elements\n"
            ))]

            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await generate_guidelines_from_audit(session, rec)

        result = asyncio.run(_run())
        # Style keywords from audit should appear
        assert "bold" in result.lower() or "geometric" in result.lower()

    def test_includes_brand_insights_in_prompt(self):
        """The prompt sent to Claude should include brand_insights data."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            audit = _sample_audit_data()
            session = _mock_session(audit=audit)
            rec = _mock_strategy()

            captured_prompt = {}
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="# Guidelines\n")]

            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                async def capture_create(**kwargs):
                    captured_prompt["messages"] = kwargs.get("messages", [])
                    return mock_response
                mock_client.messages.create = capture_create
                mock_cls.return_value = mock_client
                await generate_guidelines_from_audit(session, rec)

            return captured_prompt

        result = asyncio.run(_run())
        user_msg = result["messages"][0]["content"]
        assert "confident" in user_msg  # from brand_insights.personality
        assert "tech-savvy" in user_msg  # from brand_insights.audience

    def test_fallback_on_error(self):
        """When Claude call fails, finalize_onboarding should use template fallback."""
        async def _run():
            from agent.onboarding import finalize_onboarding, OnboardingSession

            session = OnboardingSession(
                user_id=1,
                brand_name="FallbackBrand",
                description="Test product",
                platforms=["x"],
                asset_audit={
                    "consolidated_colors": [
                        {"hex": "#abc123", "name": "Lime", "role": "primary"},
                    ],
                    "consolidated_style": ["playful"],
                },
                visual_preferences={"style": "playful"},
                strategy={
                    "archetype": "starting_fresh",
                    "compositor_enabled": False,
                    "default_mode": "image_optional",
                    "recommended_content_types": ["community"],
                },
            )

            import tempfile, os
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("agent.onboarding.settings") as mock_settings, \
                     patch("agent.onboarding.generate_guidelines_from_audit", side_effect=Exception("API error")), \
                     patch("agent.onboarding.anthropic"):
                    mock_settings.BRAND_FOLDER = tmpdir
                    mock_settings.ANTHROPIC_API_KEY = "test"

                    # Create state dir for session persistence
                    state_dir = os.path.join(tmpdir, "..", "state")
                    os.makedirs(state_dir, exist_ok=True)

                    with patch("agent.onboarding._STATE_PATH", Path(state_dir) / "onboarding.json"), \
                         patch("agent.strategy.save_strategy"), \
                         patch("agent.strategy.generate_content_calendar", AsyncMock()), \
                         patch("agent.compositor_config.invalidate_cache"):
                        await finalize_onboarding(session)

                    # Guidelines should exist (template fallback)
                    guidelines_path = os.path.join(tmpdir, "guidelines.md")
                    assert os.path.exists(guidelines_path)
                    content = open(guidelines_path).read()
                    assert "#abc123" in content  # Template used audit colors
                    assert "FallbackBrand" in content

        asyncio.run(_run())

    def test_strips_code_fences(self):
        """Code fences around Claude's response should be stripped."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            session = _mock_session(audit=_sample_audit_data())
            rec = _mock_strategy()

            fenced = '```markdown\n# TestBrand Brand Guidelines\n\nContent here\n```'
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=fenced)]

            with patch("agent.onboarding.anthropic.AsyncAnthropic") as mock_cls:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_cls.return_value = mock_client
                return await generate_guidelines_from_audit(session, rec)

        result = asyncio.run(_run())
        assert not result.startswith("```")
        assert not result.endswith("```")
        assert result.startswith("# TestBrand")


# ---------------------------------------------------------------------------
# Change 3: Brand check inventory context
# ---------------------------------------------------------------------------

class TestBrandCheckInventoryContext:
    def test_build_inventory_context_with_data(self):
        """_build_inventory_context should format inventory data."""
        import tempfile, os
        from agent.brand_check import _build_inventory_context

        inv_data = {
            "consolidated_colors": [
                {"hex": "#ff5500", "name": "Orange", "role": "primary"},
                {"hex": "#001122", "name": "Dark", "role": "neutral"},
            ],
            "consolidated_style": ["bold", "geometric"],
            "entries": [
                {"category": "logo"},
                {"category": "logo"},
                {"category": "illustration"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            inv_path = os.path.join(tmpdir, "asset_inventory.json")
            with open(inv_path, "w") as f:
                json.dump(inv_data, f)

            with patch("agent.brand_check.settings") as mock_settings:
                mock_settings.BRAND_FOLDER = tmpdir
                result = _build_inventory_context()

        assert "#ff5500" in result
        assert "bold" in result
        assert "logo" in result

    def test_build_inventory_context_no_file(self):
        """_build_inventory_context returns empty string if no inventory file."""
        from agent.brand_check import _build_inventory_context

        with patch("agent.brand_check.settings") as mock_settings:
            mock_settings.BRAND_FOLDER = "/nonexistent/path"
            result = _build_inventory_context()

        assert result == ""

    def test_check_prompt_includes_inventory(self):
        """_build_check_prompt should include inventory context when provided."""
        from agent.brand_check import _build_check_prompt

        prompt = _build_check_prompt("Brand: Test", "Asset library colors:\n  Orange #ff5500")
        assert "Asset library colors" in prompt
        assert "#ff5500" in prompt
        assert "asset library" in prompt.lower()

    def test_check_prompt_without_inventory(self):
        """_build_check_prompt without inventory should not have asset library section."""
        from agent.brand_check import _build_check_prompt

        prompt = _build_check_prompt("Brand: Test")
        assert "ASSET LIBRARY CONTEXT" not in prompt


# ---------------------------------------------------------------------------
# Change 4: /regen_guidelines command
# ---------------------------------------------------------------------------

class TestRegenGuidelines:
    def test_regen_no_inventory(self):
        """Command should fail gracefully if no asset_inventory.json."""
        from bot.handlers import regen_guidelines_command

        async def _run():
            with patch("bot.handlers._authorized", return_value=True), \
                 patch("bot.handlers.settings") as mock_settings:
                mock_settings.BRAND_FOLDER = "/nonexistent/path"

                update = _mock_update()
                ctx = _mock_context()
                await regen_guidelines_command(update, ctx)

                msg = update.message.reply_text.call_args[0][0]
                assert "no asset inventory" in msg.lower()

        asyncio.run(_run())

    def test_regen_success(self):
        """Command should regenerate guidelines from inventory."""
        import tempfile, os
        from bot.handlers import regen_guidelines_command
        from agent.asset_audit import AssetInventory, AssetAuditEntry

        async def _run():
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create asset_inventory.json
                inv_data = {
                    "entries": [{"path": "/tmp/img.jpg", "category": "logo", "dominant_colors": [],
                                 "style_keywords": [], "description": "", "quality_score": 7,
                                 "content_potential": [], "brand_signals": [], "recommended_formats": []}],
                    "consolidated_colors": [{"hex": "#ff0000", "name": "Red", "role": "primary"}],
                    "consolidated_style": ["bold"],
                    "missing_items": [],
                    "archetype": "has_identity",
                    "collection_analysis": {},
                    "brand_insights": {},
                }
                inv_path = os.path.join(tmpdir, "asset_inventory.json")
                with open(inv_path, "w") as f:
                    json.dump(inv_data, f)

                inventory = AssetInventory(
                    entries=[AssetAuditEntry(path="/tmp/img.jpg", category="logo")],
                    consolidated_colors=[{"hex": "#ff0000", "name": "Red", "role": "primary"}],
                    consolidated_style=["bold"],
                    archetype="has_identity",
                    collection_analysis={},
                    brand_insights={},
                )

                mock_cfg = MagicMock(brand_name="TestBrand", product_description="Test")

                with patch("bot.handlers._authorized", return_value=True), \
                     patch("bot.handlers.settings") as mock_settings, \
                     patch("agent.asset_audit.load_inventory", return_value=inventory), \
                     patch("agent.compositor_config.get_config", return_value=mock_cfg), \
                     patch("agent.compositor_config.invalidate_cache") as mock_invalidate, \
                     patch("agent.onboarding.generate_guidelines_from_audit", AsyncMock(
                         return_value="# TestBrand Brand Guidelines\n\n## COLOR PALETTE\n"
                     )) as mock_gen:
                    mock_settings.BRAND_FOLDER = tmpdir

                    update = _mock_update()
                    ctx = _mock_context()
                    await regen_guidelines_command(update, ctx)

                    # Should have called generate_guidelines_from_audit
                    mock_gen.assert_called_once()

                    # Should have written guidelines.md
                    guidelines_path = os.path.join(tmpdir, "guidelines.md")
                    assert os.path.exists(guidelines_path)

                    # Should have invalidated cache
                    mock_invalidate.assert_called_once()

        asyncio.run(_run())
