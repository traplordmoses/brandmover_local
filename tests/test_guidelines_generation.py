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
    def test_run_onboarding_audit_includes_entries_creative(self):
        """_run_onboarding_audit should pass entries_creative to session."""
        import tempfile, os
        from bot.handlers import _run_onboarding_audit
        from agent.asset_audit import AssetInventory, AssetAuditEntry

        async def _run():
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
                f.write(b"fake image data")

            try:
                entry = AssetAuditEntry(
                    path=tmp_path, category="logo",
                    first_impression="Striking and bold",
                    creative_dna=["geometric precision"],
                    overall_energy="quiet confidence",
                    what_makes_it_special="The negative space",
                    never_do=["Never add drop shadows"],
                    character_system="A minimalist architect",
                )
                inventory = AssetInventory(
                    entries=[entry],
                    consolidated_colors=[],
                    consolidated_style=["bold"],
                    archetype="has_identity",
                )

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
        assert "entries_creative" in result
        assert len(result["entries_creative"]) == 1
        assert result["entries_creative"][0]["first_impression"] == "Striking and bold"
        assert result["entries_creative"][0]["creative_dna"] == ["geometric precision"]
        assert result["entries_creative"][0]["never_do"] == ["Never add drop shadows"]

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

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
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

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
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

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
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
                     patch("agent._client.anthropic"):
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

    def test_prompt_contains_creative_brief_instructions(self):
        """The merge prompt template should contain instructions for CREATIVE BRIEF and NEVER DO."""
        from agent.onboarding import _GUIDELINES_MERGE_PROMPT
        assert "CREATIVE BRIEF" in _GUIDELINES_MERGE_PROMPT
        assert "NEVER DO" in _GUIDELINES_MERGE_PROMPT

    def test_creative_section_built_from_entries(self):
        """Creative section should be built from entries_creative in audit data."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            audit = _sample_audit_data()
            audit["entries_creative"] = [
                {
                    "first_impression": "Bold and electric",
                    "creative_dna": ["neon energy", "digital native"],
                    "overall_energy": "cyberpunk optimism",
                    "what_makes_it_special": "The color contrast is unmistakable",
                    "never_do": ["Never use pastels", "Avoid serif fonts"],
                    "character_system": "A rebellious hacker with a heart of gold",
                }
            ]
            session = _mock_session(audit=audit)
            rec = _mock_strategy()

            captured_prompt = {}
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="# Guidelines\n")]

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
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
        assert "cyberpunk optimism" in user_msg
        assert "neon energy" in user_msg
        assert "Never use pastels" in user_msg

    def test_empty_creative_entries_gracefully_omitted(self):
        """When no entries_creative, prompt should not contain CREATIVE AUDIT DATA."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            audit = _sample_audit_data()
            # No entries_creative key
            session = _mock_session(audit=audit)
            rec = _mock_strategy()

            captured_prompt = {}
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="# Guidelines\n")]

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
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
        assert "CREATIVE AUDIT DATA" not in user_msg

    def test_strips_code_fences(self):
        """Code fences around Claude's response should be stripped."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            session = _mock_session(audit=_sample_audit_data())
            rec = _mock_strategy()

            fenced = '```markdown\n# TestBrand Brand Guidelines\n\nContent here\n```'
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=fenced)]

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls:
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
    def test_regen_no_inventory_no_images(self):
        """Command should fail gracefully if no asset_inventory.json and no images."""
        from bot.handlers import regen_guidelines_command

        async def _run():
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("bot.handlers._authorized", return_value=True), \
                     patch("bot.handlers.settings") as mock_settings:
                    mock_settings.BRAND_FOLDER = tmpdir

                    update = _mock_update()
                    ctx = _mock_context()
                    await regen_guidelines_command(update, ctx)

                    msg = update.message.reply_text.call_args[0][0]
                    assert "no asset inventory" in msg.lower() or "no images" in msg.lower()

        asyncio.run(_run())

    def test_regen_success_with_merge(self):
        """Command should merge existing guidelines with inventory data."""
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

                # Create existing guidelines.md to be merged
                existing_guidelines = "# TestBrand\n## VOICE & TONE\nLowercase energy.\n"
                guidelines_path = os.path.join(tmpdir, "guidelines.md")
                with open(guidelines_path, "w") as f:
                    f.write(existing_guidelines)

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

                    # Should have called generate_guidelines_from_audit with existing_guidelines
                    mock_gen.assert_called_once()
                    call_kwargs = mock_gen.call_args
                    assert call_kwargs.kwargs.get("existing_guidelines") == existing_guidelines

                    # Should have written guidelines.md
                    assert os.path.exists(guidelines_path)

                    # Should have invalidated cache
                    mock_invalidate.assert_called_once()

        asyncio.run(_run())

    def test_regen_auto_audits_when_images_exist(self):
        """Command should auto-audit images when no inventory but images exist."""
        import tempfile, os
        from bot.handlers import regen_guidelines_command
        from agent.asset_audit import AssetInventory, AssetAuditEntry

        async def _run():
            with tempfile.TemporaryDirectory() as tmpdir:
                # No asset_inventory.json, but create reference images
                refs_dir = os.path.join(tmpdir, "references")
                os.makedirs(refs_dir)
                img_path = os.path.join(refs_dir, "logo.png")
                with open(img_path, "wb") as f:
                    f.write(b"fake png data")

                inventory = AssetInventory(
                    entries=[AssetAuditEntry(
                        path=img_path, category="logo",
                        first_impression="Bold anime style",
                        creative_dna=["anime-influenced"],
                        overall_energy="energetic",
                    )],
                    consolidated_colors=[{"hex": "#ff0000", "name": "Red", "role": "primary"}],
                    consolidated_style=["anime", "bold"],
                    archetype="has_identity",
                )

                mock_cfg = MagicMock(brand_name="TestBrand", product_description="Test")

                with patch("bot.handlers._authorized", return_value=True), \
                     patch("bot.handlers.settings") as mock_settings, \
                     patch("agent.asset_audit.audit_batch", AsyncMock(return_value=inventory)) as mock_audit, \
                     patch("agent.asset_audit.save_inventory") as mock_save_inv, \
                     patch("agent.compositor_config.get_config", return_value=mock_cfg), \
                     patch("agent.compositor_config.invalidate_cache"), \
                     patch("agent.onboarding.generate_guidelines_from_audit", AsyncMock(
                         return_value="# TestBrand\n## COLOR PALETTE\n"
                     )) as mock_gen:
                    mock_settings.BRAND_FOLDER = tmpdir

                    update = _mock_update()
                    ctx = _mock_context()
                    await regen_guidelines_command(update, ctx)

                    # Should have called audit_batch
                    mock_audit.assert_called_once()
                    audit_paths = mock_audit.call_args[0][0]
                    assert any("logo.png" in p for p in audit_paths)

                    # Should have saved inventory
                    mock_save_inv.assert_called_once()

                    # Should have called generate_guidelines_from_audit
                    mock_gen.assert_called_once()

                    # Session should include entries_creative
                    session_arg = mock_gen.call_args[0][0]
                    assert session_arg.asset_audit.get("entries_creative")
                    assert session_arg.asset_audit["entries_creative"][0]["first_impression"] == "Bold anime style"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Fix: Asset-first prompt prohibits invention
# ---------------------------------------------------------------------------

class TestMergePromptRules:
    def test_merge_prompt_prohibits_color_invention(self):
        """Merge prompt must prohibit inventing colors."""
        from agent.onboarding import _GUIDELINES_MERGE_PROMPT
        assert "ZERO old colors" in _GUIDELINES_MERGE_PROMPT

    def test_merge_prompt_preserves_voice_tone(self):
        """Merge prompt must instruct to preserve VOICE & TONE verbatim."""
        from agent.onboarding import _GUIDELINES_MERGE_PROMPT
        assert "VOICE & TONE" in _GUIDELINES_MERGE_PROMPT
        assert "VERBATIM" in _GUIDELINES_MERGE_PROMPT or "verbatim" in _GUIDELINES_MERGE_PROMPT

    def test_merge_prompt_replaces_colors(self):
        """Merge prompt must instruct to replace COLORS section."""
        from agent.onboarding import _GUIDELINES_MERGE_PROMPT
        assert "REPLACE" in _GUIDELINES_MERGE_PROMPT.upper()
        assert "COLOR" in _GUIDELINES_MERGE_PROMPT.upper()

    def test_merge_prompt_includes_available_fonts(self):
        """Merge prompt must reference available_fonts."""
        from agent.onboarding import _GUIDELINES_MERGE_PROMPT
        assert "available_fonts" in _GUIDELINES_MERGE_PROMPT

    def test_merge_mode_used_when_existing_guidelines_provided(self):
        """When existing_guidelines is provided, merge prompt should be used."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            session = _mock_session(audit=_sample_audit_data())
            rec = _mock_strategy()
            existing = "## VOICE & TONE\nLowercase energy.\n## COLORS\n| old | #000 |"

            captured_prompt = {}
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="# Guidelines\n")]

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.onboarding._detect_available_fonts", return_value=["Orbitron", "Inter"]):
                mock_client = AsyncMock()
                async def capture_create(**kwargs):
                    captured_prompt["messages"] = kwargs.get("messages", [])
                    return mock_response
                mock_client.messages.create = capture_create
                mock_cls.return_value = mock_client
                await generate_guidelines_from_audit(session, rec, existing_guidelines=existing)

            return captured_prompt

        result = asyncio.run(_run())
        user_msg = result["messages"][0]["content"]
        # Merge mode should contain the existing guidelines
        assert "Lowercase energy" in user_msg
        # Should contain merge instructions
        assert "PRESERVE" in user_msg.upper()
        # Should contain audit colors
        assert "#ff5500" in user_msg
        # Should list available fonts
        assert "Orbitron" in user_msg

    def test_fresh_mode_used_when_no_existing_guidelines(self):
        """When no existing_guidelines, fresh prompt should be used."""
        async def _run():
            from agent.onboarding import generate_guidelines_from_audit

            session = _mock_session(audit=_sample_audit_data())
            rec = _mock_strategy()

            captured_prompt = {}
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="# Guidelines\n")]

            with patch("agent._client.anthropic.AsyncAnthropic") as mock_cls, \
                 patch("agent.onboarding._detect_available_fonts", return_value=[]):
                mock_client = AsyncMock()
                async def capture_create(**kwargs):
                    captured_prompt["messages"] = kwargs.get("messages", [])
                    return mock_response
                mock_client.messages.create = capture_create
                mock_cls.return_value = mock_client
                await generate_guidelines_from_audit(session, rec, existing_guidelines="")

            return captured_prompt

        result = asyncio.run(_run())
        user_msg = result["messages"][0]["content"]
        # Fresh mode should NOT contain merge instructions
        assert "PRESERVE VERBATIM" not in user_msg.upper()
        # Should contain fresh generation instructions
        assert "BRAND INFO" in user_msg


# ---------------------------------------------------------------------------
# Fix: finalize_onboarding overwrites guidelines when audit data present
# ---------------------------------------------------------------------------

class TestFinalizeOnboardingPreservesGuidelines:
    def test_never_overwrites_existing_guidelines(self):
        """finalize_onboarding should NEVER overwrite existing guidelines.md, even with audit data."""
        async def _run():
            from agent.onboarding import finalize_onboarding, OnboardingSession
            import tempfile, os

            with tempfile.TemporaryDirectory() as tmpdir:
                # Pre-create a guidelines.md
                guidelines_path = os.path.join(tmpdir, "guidelines.md")
                original_content = "# Existing Curated Guidelines\nDo not destroy this."
                with open(guidelines_path, "w") as f:
                    f.write(original_content)

                session = OnboardingSession(
                    user_id=1,
                    brand_name="TestBrand",
                    description="Test product",
                    platforms=["x"],
                    asset_audit={
                        "consolidated_colors": [
                            {"hex": "#ff0000", "name": "Red", "role": "primary"},
                        ],
                        "consolidated_style": ["anime"],
                    },
                    strategy={
                        "archetype": "has_identity",
                        "compositor_enabled": False,
                        "default_mode": "image_optional",
                        "recommended_content_types": ["community"],
                    },
                )

                with patch("agent.onboarding.settings") as mock_settings, \
                     patch("agent.onboarding.generate_guidelines_from_audit", AsyncMock(
                         return_value="# New Guidelines"
                     )) as mock_gen, \
                     patch("agent._client.anthropic"), \
                     patch("agent.onboarding._STATE_PATH", Path(tmpdir) / "onboarding.json"), \
                     patch("agent.strategy.save_strategy"), \
                     patch("agent.strategy.generate_content_calendar", AsyncMock()), \
                     patch("agent.compositor_config.invalidate_cache"):
                    mock_settings.BRAND_FOLDER = tmpdir
                    mock_settings.ANTHROPIC_API_KEY = "test"

                    await finalize_onboarding(session)

                content = open(guidelines_path).read()
                # Should still have the original content — finalize never overwrites
                assert content == original_content
                # generate_guidelines_from_audit should NOT have been called
                mock_gen.assert_not_called()

        asyncio.run(_run())

    def test_creates_guidelines_when_none_exist(self):
        """finalize_onboarding should create guidelines.md when none exists."""
        async def _run():
            from agent.onboarding import finalize_onboarding, OnboardingSession
            import tempfile, os

            with tempfile.TemporaryDirectory() as tmpdir:
                session = OnboardingSession(
                    user_id=1,
                    brand_name="NewBrand",
                    description="Test product",
                    platforms=["x"],
                    asset_audit={},
                    strategy={
                        "archetype": "starting_fresh",
                        "compositor_enabled": False,
                        "default_mode": "image_optional",
                        "recommended_content_types": ["community"],
                    },
                )

                with patch("agent.onboarding.settings") as mock_settings, \
                     patch("agent.onboarding.generate_guidelines_from_audit", AsyncMock(
                         return_value="# NewBrand Guidelines\n## COLOR PALETTE\n"
                     )), \
                     patch("agent._client.anthropic"), \
                     patch("agent.onboarding._STATE_PATH", Path(tmpdir) / "onboarding.json"), \
                     patch("agent.strategy.save_strategy"), \
                     patch("agent.strategy.generate_content_calendar", AsyncMock()), \
                     patch("agent.compositor_config.invalidate_cache"):
                    mock_settings.BRAND_FOLDER = tmpdir
                    mock_settings.ANTHROPIC_API_KEY = "test"

                    await finalize_onboarding(session)

                guidelines_path = os.path.join(tmpdir, "guidelines.md")
                assert os.path.exists(guidelines_path)
                content = open(guidelines_path).read()
                assert "NewBrand" in content

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Fix: Brand check prompt includes visual DNA matching
# ---------------------------------------------------------------------------

class TestBrandCheckVisualDNA:
    def test_check_prompt_includes_visual_dna_language(self):
        """Brand check prompt should reference visual DNA matching."""
        from agent.brand_check import _build_check_prompt

        prompt = _build_check_prompt("Brand: Test", "Asset library colors:\n  Red #ff0000")
        assert "visual DNA" in prompt or "visual identity" in prompt

    def test_check_prompt_includes_raw_guidelines(self):
        """Brand check prompt should include raw guidelines when provided."""
        from agent.brand_check import _build_check_prompt

        raw = "## CREATIVE BRIEF\nBold anime energy\n## NEVER DO\n- No pastels"
        prompt = _build_check_prompt("Brand: Test", raw_guidelines=raw)
        assert "CREATIVE BRIEF" in prompt
        assert "NEVER DO" in prompt

    def test_check_prompt_color_tolerance(self):
        """Brand check prompt should allow color variations."""
        from agent.brand_check import _build_check_prompt

        prompt = _build_check_prompt("Brand: Test")
        assert "natural variation" in prompt or "perceptual range" in prompt

    def test_check_prompt_typography_defined_by_assets(self):
        """Brand check prompt should handle 'defined by brand assets' typography."""
        from agent.brand_check import _build_check_prompt

        prompt = _build_check_prompt("Brand: Test")
        assert "defined by brand assets" in prompt
