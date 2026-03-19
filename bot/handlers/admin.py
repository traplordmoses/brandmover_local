"""
Admin handlers — onboarding, setup, brand management, /reset_brand,
/template commands, /logo, /ingest, /apply, /brand_check, /regen_guidelines,
/train_lora, /lora_status, /lora_versions, /lora_switch, /lora_rollback,
/upload, /done, /template_import, /font_upload, /discord_setup,
help_command, refs_command, brand_command.
"""

__all__ = [
    "help_command",
    "platforms_command",
    "refs_command",
    "brand_command",
    "brand_edit_command",
    "confirm_edit_command",
    "cancel_edit_command",
    "setup_command",
    "discord_setup_command",
    "onboard_command",
    "onboard_cancel_command",
    "onboard_skip_command",
    "template_command",
    "template_upload_command",
    "template_test_command",
    "template_from_reference_command",
    "tplref_callback",
    "template_import_command",
    "font_upload_command",
    "logo_command",
    "ingest_command",
    "apply_command",
    "brand_check_command",
    "regen_guidelines_command",
    "train_lora_command",
    "lora_status_command",
    "lora_versions_command",
    "lora_switch_command",
    "lora_rollback_command",
    "reset_brand_command",
    "upload_command",
    "done_command",
    "health_command",
    "digest_command",
]

import asyncio as _aio
import io
import logging
import re
import tempfile
import time
from pathlib import Path

from PIL import Image as _PILImage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from agent import compositor, compositor_config, guidelines, onboarding, state
from agent import compositor_config as _cc
from config import settings

from bot.handlers.core import (
    _authorized,
    _can_operate,
    _esc,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# help, refs, brand commands
# ---------------------------------------------------------------------------


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show available commands (filtered by role)."""
    uid = update.effective_user.id
    if not _can_operate(uid):
        return

    mode = settings.AGENT_MODE

    if _authorized(uid):
        # Full admin help
        msg = (
            f"<b>BrandMover Local</b> (mode: {_esc(mode)})\n\n"
            "Send any message to generate a branded post draft.\n"
            "I'll think through it in multiple steps and show you my reasoning.\n\n"
            "<b>Commands:</b>\n"
            "/approve [N] \u2014 Approve the pending draft (option N if multiple)\n"
            "/reject <i>reason</i> \u2014 Revise the draft with feedback\n"
            "/refine <i>instruction</i> \u2014 Focused edit on pending draft (no reject)\n"
            "/edit <i>feedback</i> \u2014 Surgical edit on the last generated image\n"
            "/status \u2014 Show pending draft details\n"
            "/refs \u2014 Show loaded reference materials\n"
            "/feedback \u2014 Show approval/rejection stats\n"
            "/learn \u2014 Trigger preference learning from feedback history\n"
            "/review \u2014 Run a self-review of agent performance\n"
            "/style \u2014 Manage visual style profiles\n"
            "/brand \u2014 Show active brand config\n"
            "/brand_edit <i>instruction</i> \u2014 Edit guidelines via natural language\n"
            "/setup \u2014 Bootstrap guidelines from a PDF upload\n"
            "/cancel \u2014 Clear pending draft\n"
            "/schedule <i>time prompt</i> \u2014 Schedule a post for a specific time\n"
            "/scheduled \u2014 List upcoming scheduled posts\n"
            "/unschedule <i>id</i> \u2014 Cancel a scheduled post\n"
            "/campaign \u2014 List campaigns or show status\n"
            "/campaign_schedule <i>name</i> \u2014 Schedule all posts for a campaign\n"
            "/campaign_preview [<i>name</i>] \u2014 Get HTML preview of campaign posts\n"
            "/autostatus \u2014 Auto-posting scheduler status\n"
            "/autopause \u2014 Pause/resume auto-posting\n"
            "/autoforce <i>slot</i> \u2014 Force a specific auto-post slot\n"
            "/generate <i>type description</i> \u2014 Generate a standalone asset\n"
            "/logo \u2014 View/set brand logo\n"
            "/ingest \u2014 Extract brand info from an image\n"
            "/brand_check \u2014 Check an image against brand guidelines\n"
            "/train_lora \u2014 Trigger LoRA training from approved images\n"
            "/lora_status \u2014 Show LoRA training status and versions\n"
            "/lora_versions \u2014 List all trained LoRA versions\n"
            "/lora_switch <i>N</i> \u2014 Switch active LoRA to version N\n"
            "/lora_rollback \u2014 Roll back to previous LoRA version\n"
            "/history \u2014 Show generation history and stats\n"
            "/analytics \u2014 Show approval rates by content type and model\n"
            "/apply \u2014 Apply extracted brand info to guidelines\n"
            "/template \u2014 Toggle image composition on/off\n"
            "/template_upload \u2014 Upload a custom visual template\n"
            "/template_from_reference \u2014 Generate a template from a reference image\n"
            "/template_import \u2014 Import a template from Figma\n"
            "/font_upload \u2014 Upload a custom TTF/OTF font\n"
            "/onboard \u2014 Start conversational brand onboarding\n"
            "/onboard_cancel \u2014 Cancel onboarding\n"
            "/library \u2014 List or search the asset library\n"
            "/skills \u2014 List agent skills with usage stats\n"
            "/strategy \u2014 View current brand strategy and config\n"
            "/preview [topic] \u2014 Generate a sample post (no rate limit)\n"
            "/regen_guidelines \u2014 Regenerate guidelines from asset inventory\n"
            "/reset_brand \u2014 Wipe brand config and start fresh\n"
            "/upload \u2014 Add images to your brand asset library\n"
            "/done \u2014 Finish asset upload session\n"
            "/discord_setup \u2014 Create Discord server channels and roles\n"
            "/platforms \u2014 Show enabled publishing platforms\n"
            "/health \u2014 Show system health status\n"
            "/digest \u2014 Generate daily performance digest\n"
            "/help \u2014 Show this message"
        )
    else:
        # Operator help — limited commands
        msg = (
            f"<b>BrandMover Local</b> (operator mode)\n\n"
            "Send any message to generate a branded post draft.\n"
            "You can also send a photo with a caption to generate content.\n\n"
            "<b>Commands:</b>\n"
            "/approve [N] \u2014 Approve the pending draft (option N if multiple)\n"
            "/reject <i>reason</i> \u2014 Revise the draft with feedback\n"
            "/refine <i>instruction</i> \u2014 Focused edit on pending draft (no reject)\n"
            "/edit <i>feedback</i> \u2014 Surgical edit on the last generated image\n"
            "/status \u2014 Show pending draft details\n"
            "/cancel \u2014 Clear pending draft\n"
            "/help \u2014 Show this message"
        )

    await update.message.reply_text(msg, parse_mode="HTML")


async def platforms_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /platforms -- show enabled publishing platforms."""
    if not _authorized(update.effective_user.id):
        return

    from agent.platform_adapter import PLATFORM_CONFIGS

    enabled = settings.PUBLISH_PLATFORMS
    lines = ["<b>Publishing Platforms</b>\n"]
    for platform, config in PLATFORM_CONFIGS.items():
        status = "enabled" if platform in enabled else "disabled"
        icon = "+" if platform in enabled else "-"
        max_chars = config["max_chars"]
        lines.append(f"  {icon} <b>{_esc(platform.capitalize())}</b>: {status} (max {max_chars} chars)")

    lines.append(f"\nActive: {', '.join(enabled) if enabled else '(none)'}")
    lines.append(f"Discord cross-post: {'on' if settings.DISCORD_CROSSPOST_ENABLED else 'off'}")
    lines.append("\nSet <code>PUBLISH_PLATFORMS</code> env var to change (comma-separated).")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def refs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /refs — show loaded reference materials."""
    if not _authorized(update.effective_user.id):
        return

    summary = guidelines.get_reference_summary()
    await update.message.reply_text(
        f"<b>Reference Vault</b>\n\n<pre>{_esc(summary)}</pre>",
        parse_mode="HTML",
    )


async def brand_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /brand — show active brand config from guidelines.md."""
    if not _authorized(update.effective_user.id):
        return

    from agent import compositor_config
    summary = compositor_config.get_brand_summary()

    lines = [f"<b>{_esc(summary['brand_name'] or 'Brand Config')}</b>"]
    if summary["tagline"]:
        lines.append(f"<i>{_esc(summary['tagline'])}</i>")
    if summary["website"]:
        lines.append(f"Web: {_esc(summary['website'])}")
    if summary["x_handle"]:
        lines.append(f"X: {_esc(summary['x_handle'])}")

    if summary["colors"]:
        lines.append("\n<b>Colors</b>")
        for role, c in summary["colors"].items():
            lines.append(f"  {_esc(role)} \u2014 {_esc(c['name'])} <code>{_esc(c['hex'])}</code>")

    if summary["fonts"]:
        lines.append("\n<b>Fonts</b>")
        for use, f in summary["fonts"].items():
            lines.append(f"  {_esc(use)} \u2014 {_esc(f['family'])} {_esc(f['weight'])}")

    if summary["style_keywords"]:
        lines.append(f"\n<b>Style:</b> {_esc(', '.join(summary['style_keywords']))}")

    if summary["parsed_at"]:
        import datetime
        ts = datetime.datetime.fromtimestamp(summary["parsed_at"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"\n<i>Parsed {ts} from {_esc(summary['source_path'])}</i>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /brand_edit, /confirm_edit, /cancel_edit
# ---------------------------------------------------------------------------


async def brand_edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Edit brand guidelines through natural language.
    Usage: /brand_edit make the tone more casual
    """
    user_id = update.effective_user.id
    if not _authorized(user_id):
        return

    args = (update.message.text or "").split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text(
            "usage: /brand_edit <instruction>\n\n"
            "examples:\n"
            "  /brand_edit add color: Sunset Orange #FF6633\n"
            "  /brand_edit make the tone more casual\n"
            "  /brand_edit change tagline to 'culture lives here'\n"
            "  /brand_edit add 'the loreboard is alive' to brand phrases\n"
            "  /brand_edit remove 'revolutionizing' from the never-use list\n"
            "  /brand_edit update the posting cadence to 3-5 posts per day"
        )
        return

    instruction = args[1].strip()
    await update.message.reply_text(f"editing guidelines: {_esc(instruction)}...", parse_mode="HTML")

    from agent.guidelines_editor import apply_edit

    result = await apply_edit(instruction)

    if not result.get("success"):
        await update.message.reply_text(
            f"edit failed: {_esc(result.get('error', 'unknown error'))}",
            parse_mode="HTML",
        )
        return

    # Show preview
    diff_text = result["diff_preview"][:500]
    msg = (
        f"<b>Guidelines Edit Preview</b>\n\n"
        f"<b>Section:</b> {_esc(result['section_modified'])}\n"
        f"<b>Change:</b> {_esc(result['change_summary'])}\n\n"
        f"<pre>{_esc(diff_text)}</pre>\n\n"
        f"/confirm_edit to apply\n"
        f"/cancel_edit to discard"
    )

    context.user_data["_pending_guidelines_edit"] = result["new_content"]
    await update.message.reply_text(msg, parse_mode="HTML")


async def confirm_edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm and apply a pending guidelines edit."""
    user_id = update.effective_user.id
    if not _authorized(user_id):
        return

    new_content = context.user_data.get("_pending_guidelines_edit")
    if not new_content:
        await update.message.reply_text("no pending edit to confirm.")
        return

    from agent.guidelines_editor import confirm_edit

    success = await confirm_edit(new_content)
    context.user_data.pop("_pending_guidelines_edit", None)

    if success:
        await update.message.reply_text("guidelines updated. changes are live.")
    else:
        await update.message.reply_text("failed to write guidelines. check file permissions.")


async def cancel_edit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancel a pending guidelines edit."""
    context.user_data.pop("_pending_guidelines_edit", None)
    await update.message.reply_text("edit cancelled.")


# ---------------------------------------------------------------------------
# /setup
# ---------------------------------------------------------------------------


async def setup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /setup — start PDF brand bootstrap."""
    if not _authorized(update.effective_user.id):
        return

    await update.message.reply_text(
        "<b>Brand Setup from PDF</b>\n\n"
        "Send me a PDF of your brand guidelines and I'll extract them into "
        "a structured <code>guidelines.md</code> that the bot can use.\n\n"
        "Just upload the PDF as a document in this chat.",
        parse_mode="HTML",
    )
    context.user_data["awaiting_setup_pdf"] = True


# ---------------------------------------------------------------------------
# /discord_setup
# ---------------------------------------------------------------------------


async def discord_setup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /discord_setup — create Discord server structure."""
    if not _authorized(update.effective_user.id):
        return

    from agent import discord_bot

    if not settings.DISCORD_BOT_TOKEN:
        await update.message.reply_text(
            "DISCORD_BOT_TOKEN not set. Add it to .env and restart."
        )
        return

    if not discord_bot.is_ready():
        await update.message.reply_text(
            "Discord client is not connected. Check your bot token and restart."
        )
        return

    status_msg = await update.message.reply_text("Setting up Discord server...")

    async def _progress(msg: str):
        try:
            await status_msg.edit_text(f"Setting up Discord server...\n{msg}")
        except Exception:
            pass

    result = await discord_bot.setup_server(progress_callback=_progress)

    if result.get("error"):
        await status_msg.edit_text(f"Discord setup failed: {_esc(result['error'])}")
        return

    await status_msg.edit_text(
        f"Discord server setup complete!\n"
        f"Channels created: {result.get('created_channels', 0)}\n"
        f"Roles created: {result.get('created_roles', 0)}\n"
        f"Total channels mapped: {result.get('total_channels', 0)}"
    )


# ---------------------------------------------------------------------------
# Onboarding commands
# ---------------------------------------------------------------------------


async def onboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /onboard — start conversational onboarding."""
    if not _authorized(update.effective_user.id):
        return

    user_id = update.effective_user.id
    session = onboarding.get_session(user_id)

    if session and session.state not in (
        onboarding.OnboardingState.IDLE.value,
        onboarding.OnboardingState.COMPLETE.value,
    ):
        await update.message.reply_text(
            f"You have an onboarding session in progress "
            f"(state: {_esc(session.state)}, brand: {_esc(session.brand_name)}).\n\n"
            f"Continue where you left off, or /onboard_cancel to start fresh.",
            parse_mode="HTML",
        )
        return

    session = onboarding.OnboardingSession(user_id=user_id)
    session, response = onboarding.advance(session, None)
    await onboarding.async_save_session(session)
    await update.message.reply_text(response, parse_mode="HTML")


async def onboard_cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /onboard_cancel — cancel onboarding session."""
    if not _authorized(update.effective_user.id):
        return

    onboarding.delete_session(update.effective_user.id)
    await update.message.reply_text("Onboarding cancelled. Use /onboard to start again.")


async def onboard_skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /onboard_skip — skip upload phase during onboarding."""
    if not _authorized(update.effective_user.id):
        return

    session = onboarding.get_session(update.effective_user.id)
    if not session or session.state != onboarding.OnboardingState.UPLOADS.value:
        await update.message.reply_text("Not in upload phase. Use /onboard to start onboarding.")
        return

    session, response = onboarding.advance(session, "/onboard_skip")
    await onboarding.async_save_session(session)
    await update.message.reply_text(response, parse_mode="HTML")

    if session.state == onboarding.OnboardingState.AUDITING.value:
        await _run_onboarding_audit(update, session)


async def _run_onboarding_audit(update: Update, session: onboarding.OnboardingSession) -> None:
    """Run Claude Vision audit on uploaded assets and advance the session."""
    from agent import asset_audit

    await update.message.chat.send_action("typing")

    paths = [a["path"] for a in session.uploaded_assets if Path(a["path"]).exists()]
    if not paths:
        session.state = onboarding.OnboardingState.VISUAL_PREF.value
        await onboarding.async_save_session(session)
        await update.message.reply_text(
            "No valid assets found. Let's pick a visual style instead.\n\n"
            "Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
            "<b>minimal</b> / <b>bold</b> / <b>elegant</b>",
            parse_mode="HTML",
        )
        return

    try:
        inventory = await asset_audit.audit_batch(paths)
        asset_audit.save_inventory(inventory)

        entries_creative = [
            {
                "first_impression": e.first_impression,
                "creative_dna": e.creative_dna,
                "overall_energy": e.overall_energy,
                "what_makes_it_special": e.what_makes_it_special,
                "never_do": e.never_do,
                "character_system": e.character_system,
            }
            for e in inventory.entries
            if e.first_impression
        ]
        audit_data = {
            "archetype": inventory.archetype,
            "consolidated_colors": inventory.consolidated_colors,
            "consolidated_style": inventory.consolidated_style,
            "missing_items": inventory.missing_items,
            "entry_count": len(inventory.entries),
            "collection_analysis": inventory.collection_analysis,
            "brand_insights": inventory.brand_insights,
            "entries_creative": entries_creative,
        }
        session, response = onboarding.finalize_audit(session, audit_data)
        await onboarding.async_save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")
    except Exception as e:
        logger.error("Onboarding audit failed: %s", e)
        session.state = onboarding.OnboardingState.VISUAL_PREF.value
        await onboarding.async_save_session(session)
        await update.message.reply_text(
            f"Asset analysis failed: {_esc(str(e))}\n\nLet's pick a visual style instead.\n"
            "Options: <b>modern</b> / <b>playful</b> / <b>corporate</b> / "
            "<b>minimal</b> / <b>bold</b> / <b>elegant</b>",
            parse_mode="HTML",
        )


async def _run_onboarding_strategy(update: Update, session: onboarding.OnboardingSession) -> None:
    """Run strategy recommendation and advance session to CONFIRM."""
    from agent import strategy as strategy_mod
    from agent.asset_audit import AssetInventory, load_inventory

    await update.message.chat.send_action("typing")

    try:
        inventory = load_inventory()
        rec = await strategy_mod.recommend_strategy(
            brand_name=session.brand_name,
            description=session.description,
            platforms=session.platforms,
            inventory=inventory,
            visual_preferences=session.visual_preferences,
        )

        strategy_data = {
            "archetype": rec.archetype,
            "compositor_enabled": rec.compositor_enabled,
            "badge_text": rec.badge_text,
            "default_mode": rec.default_mode,
            "recommended_content_types": rec.recommended_content_types,
            "visual_style_notes": rec.visual_style_notes,
            "reasoning": rec.reasoning,
        }
        session, response = onboarding.finalize_strategy(session, strategy_data)
        await onboarding.async_save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")
    except Exception as e:
        logger.error("Onboarding strategy failed: %s", e)
        from agent.strategy import _ARCHETYPE_DEFAULTS
        archetype = session.asset_audit.get("archetype", "starting_fresh")
        defaults = _ARCHETYPE_DEFAULTS.get(archetype, _ARCHETYPE_DEFAULTS["starting_fresh"])
        strategy_data = {"archetype": archetype, **defaults, "reasoning": f"Auto-configured (strategy generation failed: {e})"}
        session, response = onboarding.finalize_strategy(session, strategy_data)
        await onboarding.async_save_session(session)
        await update.message.reply_text(response, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Template commands
# ---------------------------------------------------------------------------


async def template_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template — toggle image composition on/off."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    arg = text.partition("/template")[2].strip().lower()

    guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"

    if not arg:
        cfg = _cc.get_config()
        status = "ON" if cfg.compositor_enabled else "OFF"
        badge = cfg.badge_text or "(none)"
        mode = cfg.default_mode

        from agent import template_memory as _tm
        memory = _tm.TemplateMemory()
        templates = memory.list_templates()
        tpl_lines = ""
        if templates:
            tpl_lines = "\n\n<b>Active Templates:</b>\n"
            for t in templates:
                types_str = ", ".join(t.content_types) if t.content_types else "all"
                source_tag = f" [{t.source}]" if t.source else ""
                has_spec = " (spec)" if t.spec_json else ""
                tpl_lines += (
                    f"- <code>{_esc(t.id)}</code> {_esc(t.name)} "
                    f"({t.aspect_ratio}, {types_str}){source_tag}{has_spec}\n"
                )

        await update.message.reply_text(
            f"<b>Compositor Status</b>\n\n"
            f"Enabled: <b>{status}</b>\n"
            f"Badge: <code>{_esc(badge)}</code>\n"
            f"Mode: <code>{_esc(mode)}</code>\n\n"
            f"<code>/template on</code> \u2014 enable\n"
            f"<code>/template off</code> \u2014 disable\n"
            f"<code>/template_upload</code> \u2014 upload a custom template\n"
            f"<code>/template_from_reference</code> \u2014 generate from reference image\n"
            f"<code>/template_import</code> \u2014 import from Figma\n"
            f"<code>/font_upload</code> \u2014 upload custom font"
            f"{tpl_lines}",
            parse_mode="HTML",
        )
        return

    if arg not in ("on", "off"):
        await update.message.reply_text(
            "Usage: /template on | /template off | /template",
        )
        return

    enabled_value = "true" if arg == "on" else "false"

    if not guidelines_path.exists():
        await update.message.reply_text("No guidelines.md found. Run /setup first.")
        return

    content = await _aio.to_thread(guidelines_path.read_text, encoding="utf-8")

    import re as _re
    section_match = _re.search(r"##\s*COMPOSITOR(.*?)(?=\n##|\Z)", content, _re.DOTALL)
    if section_match:
        section = section_match.group(0)
        updated = _re.sub(
            r"(\|\s*Enabled\s*\|\s*)(true|false|yes|no|on|off)(\s*\|)",
            rf"\g<1>{enabled_value}\g<3>",
            section,
            flags=_re.IGNORECASE,
        )
        content = content.replace(section, updated)
    else:
        content += f"\n\n## COMPOSITOR\n\n| Setting        | Value          |\n|----------------|----------------|\n| Enabled        | {enabled_value}           |\n"

    await _aio.to_thread(guidelines_path.write_text, content, encoding="utf-8")
    _cc.invalidate_cache()
    compositor.clear_font_cache()
    guidelines.invalidate_brand_context()

    status_str = "ON" if arg == "on" else "OFF"
    await update.message.reply_text(
        f"Compositor <b>{status_str}</b>",
        parse_mode="HTML",
    )


async def template_upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_upload [name] — start template upload flow."""
    if not _authorized(update.effective_user.id):
        return

    template_name = " ".join(context.args) if context.args else ""
    context.user_data["awaiting_template"] = True
    context.user_data["template_name"] = template_name
    name_note = f" (name: <b>{_esc(template_name)}</b>)" if template_name else ""
    await update.message.reply_text(
        f"Send me a template image (frame, mockup, bordered layout).{name_note}\n"
        "I'll analyze it and register it for future posts.",
        parse_mode="HTML",
    )


async def template_test_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_test [content_type] — render a test composition with placeholder content."""
    if not _authorized(update.effective_user.id):
        return

    from agent import template_memory as _tm

    content_type = " ".join(context.args).strip() if context.args else "announcement"
    memory = _tm.TemplateMemory()
    template = memory.get_template_for_content_type(content_type)
    if not template:
        await update.message.reply_text(
            "No templates registered. Use /template_upload to add one.",
        )
        return

    await update.message.chat.send_action("upload_photo")

    cfg = _cc.get_config()
    primary = cfg.colors.get("primary")
    if primary:
        primary_rgb = primary.rgb
    else:
        primary_rgb = (107, 159, 212)
    placeholder = _PILImage.new("RGBA", (template.width, template.height), primary_rgb + (255,))
    tmp_fd = tempfile.NamedTemporaryFile(suffix=".png", prefix="test_placeholder_", delete=False)
    placeholder_path = tmp_fd.name
    tmp_fd.close()
    await _aio.to_thread(lambda: placeholder.save(placeholder_path, "PNG"))

    test_draft = {"title": "HEADLINE HERE", "subtitle": "Subtitle text goes here"}

    try:
        result = await _tm.apply_template(template, placeholder_path, test_draft)
        if result:
            regions_str = ", ".join(f"{r.type}({r.width}x{r.height})" for r in template.regions)
            await update.message.reply_photo(
                photo=result,
                caption=(
                    f"Template test: <b>{_esc(template.name)}</b> (<code>{_esc(template.id)}</code>)\n"
                    f"Aspect: {_esc(template.aspect_ratio)} | Regions: {_esc(regions_str)}\n"
                    f"Content type: {_esc(content_type)}"
                ),
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("Template composition failed \u2014 check that the template has an image region.")
    except Exception as e:
        logger.error("Template test failed: %s", e)
        await update.message.reply_text(f"Template test failed: {_esc(str(e))}", parse_mode="HTML")
    finally:
        try:
            Path(placeholder_path).unlink(missing_ok=True)
        except Exception:
            pass


async def template_from_reference_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_from_reference [name] — generate a branded template from a reference image."""
    if not _authorized(update.effective_user.id):
        return

    template_name = " ".join(context.args) if context.args else ""
    context.user_data["awaiting_template_from_ref"] = True
    context.user_data["template_from_ref_name"] = template_name
    name_note = f" (name: <b>{_esc(template_name)}</b>)" if template_name else ""
    await update.message.reply_text(
        f"Send me a reference image (screenshot, post, layout you like).{name_note}\n"
        "I'll analyze the layout and recreate it as a branded template using your colors, fonts, and logo.",
        parse_mode="HTML",
    )


async def _handle_template_from_reference(update: Update, context: ContextTypes.DEFAULT_TYPE, tmp_path: str) -> None:
    """Generate a branded template preview from a reference image (no registration yet)."""
    from agent import template_generator as _tg

    context.user_data["awaiting_template_from_ref"] = False
    await update.message.chat.send_action("typing")
    await update.message.reply_text("Analyzing layout and building template spec...")

    try:
        design, preview_img = await _tg.analyze_and_generate(tmp_path)

        context.user_data["tplref_pending"] = {
            "design": _tg.design_to_dict(design),
            "name": (context.user_data or {}).get("template_from_ref_name", ""),
        }

        await _send_template_preview(update, context, design, preview_img)
    except Exception as e:
        logger.error("Template generation from reference failed: %s", e)
        context.user_data.pop("tplref_pending", None)
        await update.message.reply_text(
            f"Template generation failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


async def _send_template_preview(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    design,
    generated_img,
) -> None:
    """Send a template preview with inline save/adjust/discard buttons."""
    from agent import template_generator as _tg

    buf = io.BytesIO()
    await _aio.to_thread(lambda: generated_img.convert("RGB").save(buf, "PNG"))
    buf.seek(0)

    regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in design.regions)
    aspect = _tg._compute_aspect_ratio(design.canvas_width, design.canvas_height)
    name = (context.user_data.get("tplref_pending") or {}).get("name", "")

    has_spec = hasattr(design, "spec") and design.spec is not None
    method_note = "spec-rendered" if has_spec else "analyzed"
    caption = (
        f"<b>Template Preview</b> ({method_note})\n\n"
        f"Size: {design.canvas_width}x{design.canvas_height} ({aspect})\n"
        f"Style: {_esc(design.visual_style or 'detected')}\n"
        f"Regions: {_esc(regions_str) or 'none detected'}\n"
    )
    if name:
        caption += f"Name: <b>{_esc(name)}</b>\n"
    caption += "\nTap <b>Save</b> to register, <b>Adjust</b> to modify, or reply with feedback."

    buttons = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Save Template", callback_data="tplref_save"),
            InlineKeyboardButton("Adjust", callback_data="tplref_adjust"),
            InlineKeyboardButton("Discard", callback_data="tplref_discard"),
        ]
    ])

    await update.message.reply_photo(
        photo=buf,
        caption=caption,
        parse_mode="HTML",
        reply_markup=buttons,
    )


async def _handle_tplref_adjustment(update: Update, context: ContextTypes.DEFAULT_TYPE, feedback: str) -> None:
    """Apply user feedback to the pending template design — adjusts prompt and regenerates."""
    from agent import template_generator as _tg

    pending = context.user_data.get("tplref_pending")
    if not pending:
        return

    lower = feedback.lower().strip()
    if lower in ("looks good", "save", "save it", "done", "perfect", "yes", "keep it", "ok", "okay"):
        design = _tg.design_from_dict(pending["design"])
        name = pending.get("name") or None
        saved_path = await _tg.save_generated_image(design)
        template = _tg.register_design(design, saved_path, name)
        context.user_data.pop("tplref_pending", None)

        regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in template.regions)
        await update.message.reply_text(
            f"<b>Template Saved</b>\n\n"
            f"Name: <b>{_esc(template.name)}</b>\n"
            f"ID: <code>{_esc(template.id)}</code>\n"
            f"Size: {template.width}x{template.height} ({template.aspect_ratio})\n"
            f"Regions: {_esc(regions_str) or 'none detected'}\n\n"
            f"This template will be used for future posts.",
            parse_mode="HTML",
        )
        return

    if lower in ("discard", "cancel", "nevermind", "never mind", "nah", "no"):
        context.user_data.pop("tplref_pending", None)
        await update.message.reply_text("Template discarded.")
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Adjusting template spec and re-rendering...")

    try:
        design = _tg.design_from_dict(pending["design"])
        adjusted = await _tg.adjust_spec(design, feedback)

        if adjusted.spec:
            from agent.template_renderer import render_preview
            preview_img = render_preview(adjusted.spec)
        elif adjusted.generated_image_url:
            preview_img = await _tg.download_image(adjusted.generated_image_url)
            if not preview_img:
                raise ValueError("Failed to download regenerated template image.")
        else:
            from PIL import Image as _PILImg
            preview_img = _PILImg.new("RGB", (adjusted.canvas_width, adjusted.canvas_height), (14, 15, 43))

        pending["design"] = _tg.design_to_dict(adjusted)
        context.user_data["tplref_pending"] = pending

        await _send_template_preview(update, context, adjusted, preview_img)
    except Exception as e:
        logger.error("Template adjustment failed: %s", e)
        await update.message.reply_text(
            f"Adjustment failed: {_esc(str(e))}\nYou can try again or tap Save/Discard.",
            parse_mode="HTML",
        )


async def tplref_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button presses for template-from-reference previews."""
    from agent import template_generator as _tg

    query = update.callback_query
    await query.answer()

    if not _authorized(query.from_user.id):
        return

    action = query.data.replace("tplref_", "")
    pending = (context.user_data or {}).get("tplref_pending")

    if not pending:
        await query.message.reply_text("No template preview pending.")
        return

    if action == "save":
        design = _tg.design_from_dict(pending["design"])
        name = pending.get("name") or None
        saved_path = await _tg.save_generated_image(design)
        template = _tg.register_design(design, saved_path, name)

        context.user_data.pop("tplref_pending", None)

        regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in template.regions)
        await query.edit_message_caption(
            caption=(
                f"<b>Template Saved</b>\n\n"
                f"Name: <b>{_esc(template.name)}</b>\n"
                f"ID: <code>{_esc(template.id)}</code>\n"
                f"Size: {template.width}x{template.height} ({template.aspect_ratio})\n"
                f"Regions: {_esc(regions_str) or 'none detected'}\n\n"
                f"This template will be used for future posts."
            ),
            parse_mode="HTML",
            reply_markup=None,
        )

    elif action == "adjust":
        await query.message.reply_text(
            "What should I adjust? Reply with your feedback, e.g.:\n"
            "<i>make the colors more vibrant</i>\n"
            "<i>use a darker background</i>\n"
            "<i>add more spacing between elements</i>",
            parse_mode="HTML",
        )

    elif action == "discard":
        context.user_data.pop("tplref_pending", None)
        await query.edit_message_caption(
            caption="<b>Template discarded.</b>",
            parse_mode="HTML",
            reply_markup=None,
        )


async def _handle_template_region_update(
    update: Update, context: ContextTypes.DEFAULT_TYPE, description: str,
) -> None:
    """Convert a natural language region description into pixel coordinates and update the template."""
    from agent import template_memory as _tm

    template_id = context.user_data.get("last_uploaded_template_id", "")
    if not template_id:
        await update.message.reply_text("No recently uploaded template to update.")
        return

    memory = _tm.TemplateMemory()
    templates = memory.list_templates()
    template = next((t for t in templates if t.id == template_id), None)
    if not template:
        await update.message.reply_text(
            f"Template <code>{_esc(template_id)}</code> not found in manifest.",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Parsing region positions...")

    try:
        new_regions = await _tm.parse_region_description(
            description, template.width, template.height,
        )
    except Exception as e:
        logger.error("Region parsing failed: %s", e)
        await update.message.reply_text(f"Region parsing failed: {_esc(str(e))}", parse_mode="HTML")
        return

    if not new_regions:
        await update.message.reply_text("Couldn't parse any regions from that description. Try being more specific.")
        return

    updated = memory.update_template_regions(template_id, new_regions)
    if not updated:
        await update.message.reply_text("Failed to update template regions.")
        return

    context.user_data.pop("last_uploaded_template_id", None)

    region_lines = []
    for r in new_regions:
        region_lines.append(
            f"  {r.type}: ({r.x}, {r.y}, {r.width}, {r.height}) \u2014 {r.description}"
        )
    regions_display = "\n".join(region_lines)
    await update.message.reply_text(
        f"<b>Template updated</b> \u2014 <code>{_esc(template_id)}</code>\n\n"
        f"<pre>{_esc(regions_display)}</pre>",
        parse_mode="HTML",
    )

    # Auto-run template_test to show what it looks like
    await update.message.chat.send_action("upload_photo")

    content_type = updated.content_types[0] if updated.content_types else "announcement"

    cfg = _cc.get_config()
    primary = cfg.colors.get("primary")
    primary_rgb = primary.rgb if primary else (107, 159, 212)
    placeholder = _PILImage.new("RGBA", (updated.width, updated.height), primary_rgb + (255,))
    tmp_fd = tempfile.NamedTemporaryFile(suffix=".png", prefix="test_placeholder_", delete=False)
    placeholder_path = tmp_fd.name
    tmp_fd.close()
    await _aio.to_thread(lambda: placeholder.save(placeholder_path, "PNG"))

    test_draft = {"title": "HEADLINE HERE", "subtitle": "Subtitle text goes here"}

    try:
        result = await _tm.apply_template(updated, placeholder_path, test_draft)
        if result:
            regions_str = ", ".join(f"{r.type}({r.width}x{r.height})" for r in updated.regions)
            await update.message.reply_photo(
                photo=result,
                caption=(
                    f"Template test: <b>{_esc(updated.name)}</b>\n"
                    f"Regions: {_esc(regions_str)}"
                ),
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("Template test composition failed \u2014 check that the template has an image region.")
    except Exception as e:
        logger.warning("Auto template test failed: %s", e)
    finally:
        try:
            Path(placeholder_path).unlink(missing_ok=True)
        except Exception:
            pass


async def _handle_template_upload(update: Update, context: ContextTypes.DEFAULT_TYPE, tmp_path: str) -> None:
    """Process a template image upload — analyze and register."""
    from agent import template_memory as _tm

    context.user_data["awaiting_template"] = False
    await update.message.chat.send_action("typing")
    await update.message.reply_text("Analyzing template...")

    templates_dir = Path(settings.BRAND_FOLDER) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    dest = templates_dir / f"template_{int(time.time())}.png"
    try:
        await _aio.to_thread(lambda: _PILImage.open(tmp_path).convert("RGBA").save(str(dest), "PNG"))
    except Exception as e:
        await update.message.reply_text(f"Failed to process image: {_esc(str(e))}", parse_mode="HTML")
        return

    try:
        user_name = (context.user_data or {}).get("template_name", "")
        template = await _tm.register_template(
            str(dest),
            name=user_name or f"Template {int(time.time()) % 10000}",
        )

        context.user_data["last_uploaded_template_id"] = template.id

        regions_str = ", ".join(f"{r.type} ({r.width}x{r.height})" for r in template.regions)
        await update.message.reply_text(
            f"<b>Template Registered</b>\n\n"
            f"ID: <code>{_esc(template.id)}</code>\n"
            f"Size: {template.width}x{template.height} ({template.aspect_ratio})\n"
            f"Regions: {_esc(regions_str) or 'none detected'}\n"
            f"Notes: {_esc(template.analysis_notes or 'none')}\n\n"
            f"This template will be used for future posts.\n"
            f"You can describe region positions to adjust (e.g. \"top text across the top 15%, image fills the full canvas\").",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Template registration failed: %s", e)
        await update.message.reply_text(f"Template analysis failed: {_esc(str(e))}", parse_mode="HTML")


async def template_import_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /template_import <figma_url> [name] — import a template from Figma."""
    if not _authorized(update.effective_user.id):
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/template_import &lt;figma_url&gt; [name]</code>\n\n"
            "Example:\n<code>/template_import https://figma.com/design/abc123/MyFile?node-id=1-2 Hero Card</code>",
            parse_mode="HTML",
        )
        return

    figma_url = args[0]
    name = " ".join(args[1:]) if len(args) > 1 else ""

    from config import settings as _settings
    if not _settings.FIGMA_ACCESS_TOKEN:
        await update.message.reply_text(
            "Figma integration requires <code>FIGMA_ACCESS_TOKEN</code> in .env",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Importing template from Figma...")

    try:
        from agent import template_generator as _tg
        design, screenshot_img = await _tg.import_from_figma(figma_url, name or None)

        context.user_data["tplref_pending"] = {
            "design": _tg.design_to_dict(design),
            "name": name,
        }

        await _send_template_preview(update, context, design, screenshot_img)
    except Exception as e:
        logger.error("Figma template import failed: %s", e)
        await update.message.reply_text(
            f"Figma import failed: {_esc(str(e))}",
            parse_mode="HTML",
        )


async def font_upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /font_upload — mark next file upload as a font file."""
    if not _authorized(update.effective_user.id):
        return

    context.user_data["awaiting_font_upload"] = True
    await update.message.reply_text(
        "Send me a TTF or OTF font file. It will be saved to the brand fonts directory "
        "and available for use in templates and compositions.",
    )


# ---------------------------------------------------------------------------
# /logo, /ingest, /apply, /brand_check, /regen_guidelines
# ---------------------------------------------------------------------------


async def logo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /logo — show current logo or prepare for logo upload."""
    if not _authorized(update.effective_user.id):
        return

    logo_path = Path(settings.BRAND_FOLDER) / "assets" / "logo.png"

    if logo_path.exists():
        try:
            logo_bytes = await _aio.to_thread(logo_path.read_bytes)
            await update.message.reply_photo(
                photo=logo_bytes,
                caption="Current brand logo. Send a new image to replace it.",
            )
        except Exception as e:
            logger.warning("Failed to send logo: %s", e)
            await update.message.reply_text("Logo file exists but couldn't be sent.")
    else:
        await update.message.reply_text("No logo set yet.")

    context.user_data["awaiting_logo_upload"] = True
    await update.message.reply_text(
        "Send me an image to set as the brand logo.",
    )


async def ingest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ingest — prepare to extract brand info from an uploaded image."""
    if not _authorized(update.effective_user.id):
        return

    context.user_data["awaiting_ingest_image"] = True
    await update.message.reply_text(
        "Send me a brand asset (logo, screenshot, marketing material) and I'll "
        "extract colors, fonts, and style keywords from it using AI vision.\n\n"
        "The extracted info will be compared against your current brand guidelines.",
    )


async def apply_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /apply — merge last ingest extraction into guidelines.md."""
    if not _authorized(update.effective_user.id):
        return

    extracted = context.user_data.get("last_ingest_extracted")
    if not extracted:
        await update.message.reply_text(
            "No extracted data to apply. Use /ingest first and send a brand image.",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Merging extracted data into guidelines...")

    try:
        from agent import ingest
        import shutil

        guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"

        if guidelines_path.exists():
            backup_path = guidelines_path.with_suffix(".md.bak")
            await _aio.to_thread(shutil.copy2, guidelines_path, backup_path)
            logger.info("Guidelines backed up to %s", backup_path)

        new_content = await ingest.apply_extracted_to_guidelines(extracted)
        await _aio.to_thread(guidelines_path.write_text, new_content, encoding="utf-8")

        compositor_config.invalidate_cache()
        compositor.clear_font_cache()
        guidelines.invalidate_brand_context()

        context.user_data.pop("last_ingest_extracted", None)

        await update.message.reply_text(
            f"Guidelines updated ({len(new_content)} chars).\n"
            f"Backup saved to <code>guidelines.md.bak</code>\n"
            f"Config cache invalidated.",
            parse_mode="HTML",
        )
        logger.info("Guidelines updated from /apply (%d chars)", len(new_content))

    except Exception as e:
        logger.error("Apply command failed: %s", e)
        await update.message.reply_text(
            f"Failed to apply: {_esc(str(e))}",
            parse_mode="HTML",
        )


async def brand_check_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /brand_check — check if an image matches brand guidelines."""
    if not _authorized(update.effective_user.id):
        return

    reply = update.message.reply_to_message
    if reply and (reply.photo or (reply.document and (reply.document.mime_type or "").startswith("image/"))):
        tg_file = None
        if reply.photo:
            tg_file = await reply.photo[-1].get_file()
        elif reply.document:
            tg_file = await reply.document.get_file()

        if tg_file:
            await _run_brand_check(update, tg_file)
            return

    context.user_data["awaiting_brand_check"] = True
    await update.message.reply_text(
        "Send me an image and I'll check how well it matches your brand guidelines.\n\n"
        "I'll analyze colors, typography, visual style, brand elements, and layout.",
    )


async def _run_brand_check(update: Update, tg_file) -> None:
    """Download image, run brand compliance check, and send formatted report."""
    tmp_fd = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="brandmover_check_", delete=False)
    tmp_path = tmp_fd.name
    tmp_fd.close()

    try:
        await tg_file.download_to_drive(tmp_path)
    except Exception as e:
        logger.error("Failed to download image for brand check: %s", e)
        await update.message.reply_text(
            "couldn't download that image, try sending it as a photo instead of a file"
        )
        return

    try:
        _file_size = Path(tmp_path).stat().st_size
        if _file_size > 10 * 1024 * 1024:
            logger.warning("Uploaded image too large (%d bytes), skipping PIL processing", _file_size)
            await update.message.reply_text("That image is too large (>10 MB). Please send a smaller one.")
            return
    except OSError:
        pass

    try:
        def _convert_jpeg():
            img = _PILImage.open(tmp_path).convert("RGB")
            img.save(tmp_path, "JPEG", quality=95)
        await _aio.to_thread(_convert_jpeg)
    except Exception as e:
        logger.warning("Image conversion failed (using as-is): %s", e)

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Checking image against brand guidelines...")

    try:
        from agent import brand_check
        report = await brand_check.check_brand_compliance(tmp_path)
        formatted = brand_check.format_compliance_report(report)
        await update.message.reply_text(formatted, parse_mode="HTML")
    except Exception as e:
        logger.error("Brand check failed: %s", e)
        await update.message.reply_text(
            f"Brand check failed: {_esc(str(e))}", parse_mode="HTML"
        )


async def regen_guidelines_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /regen_guidelines — regenerate guidelines.md from asset_inventory.json."""
    if not _authorized(update.effective_user.id):
        return

    from agent import asset_audit, compositor_config
    from agent.strategy import StrategyRecommendation

    inv_path = Path(settings.BRAND_FOLDER) / "asset_inventory.json"

    if not inv_path.exists():
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
        scan_dirs = [
            Path(settings.BRAND_FOLDER) / "references",
            Path(settings.BRAND_FOLDER) / "assets",
        ]
        image_paths: list[str] = []
        for scan_dir in scan_dirs:
            if scan_dir.exists():
                for f in scan_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in image_exts:
                        image_paths.append(str(f))

        if not image_paths:
            await update.message.reply_text(
                "No asset inventory found and no images in brand/references/ or brand/assets/.\n"
                "Upload brand assets first via /upload or /onboard."
            )
            return

        await update.message.chat.send_action("typing")
        await update.message.reply_text(
            f"No asset inventory found. Auto-auditing {len(image_paths)} image(s) "
            f"from brand references..."
        )

        try:
            inventory = await asset_audit.audit_batch(image_paths)
            asset_audit.save_inventory(inventory)
        except Exception as e:
            logger.error("Auto-audit failed during regen_guidelines: %s", e)
            await update.message.reply_text(
                f"Auto-audit failed: {_esc(str(e))}", parse_mode="HTML"
            )
            return
    else:
        await update.message.chat.send_action("typing")
        inventory = asset_audit.load_inventory()
        if inventory is None:
            await update.message.reply_text(
                "Asset inventory file exists but couldn't be loaded. Try /upload to re-audit."
            )
            return

    await update.message.reply_text("Regenerating guidelines from your asset inventory...")

    try:
        entries_creative = []
        for entry in inventory.entries:
            ec: dict = {}
            if entry.first_impression:
                ec["first_impression"] = entry.first_impression
            if entry.creative_dna:
                ec["creative_dna"] = entry.creative_dna
            if entry.overall_energy:
                ec["overall_energy"] = entry.overall_energy
            if entry.what_makes_it_special:
                ec["what_makes_it_special"] = entry.what_makes_it_special
            if entry.never_do:
                ec["never_do"] = entry.never_do
            if entry.character_system:
                ec["character_system"] = entry.character_system
            if ec:
                entries_creative.append(ec)

        session = onboarding.OnboardingSession(
            user_id=update.effective_user.id,
            brand_name=compositor_config.get_config().brand_name or "Brand",
            description=compositor_config.get_config().product_description or "",
            platforms=["x"],
            asset_audit={
                "archetype": inventory.archetype,
                "consolidated_colors": inventory.consolidated_colors,
                "consolidated_style": inventory.consolidated_style,
                "missing_items": inventory.missing_items,
                "entry_count": len(inventory.entries),
                "collection_analysis": inventory.collection_analysis,
                "brand_insights": inventory.brand_insights,
                "entries_creative": entries_creative,
            },
        )

        config_path = Path(settings.BRAND_FOLDER) / "config.json"
        rec_data = {}
        if config_path.exists():
            try:
                import json as _json
                rec_data = _json.loads(await _aio.to_thread(config_path.read_text, encoding="utf-8"))
            except Exception:
                pass
        rec = StrategyRecommendation(
            archetype=rec_data.get("archetype", inventory.archetype),
            compositor_enabled=rec_data.get("compositor_enabled", False),
            badge_text=rec_data.get("badge_text"),
            default_mode=rec_data.get("default_mode", "image_optional"),
            recommended_content_types=rec_data.get("recommended_content_types", []),
            platforms=["x"],
        )

        guidelines_path = Path(settings.BRAND_FOLDER) / "guidelines.md"
        existing_guidelines = ""
        if guidelines_path.exists():
            try:
                existing_guidelines = await _aio.to_thread(guidelines_path.read_text, encoding="utf-8")
            except OSError:
                pass

        guidelines_md = await onboarding.generate_guidelines_from_audit(
            session, rec, existing_guidelines=existing_guidelines,
        )
        await _aio.to_thread(guidelines_path.write_text, guidelines_md, encoding="utf-8")
        compositor_config.invalidate_cache()

        mode = "merged with" if existing_guidelines else "generated from"
        await update.message.reply_text(
            f"guidelines.md has been {mode} your asset inventory.\n"
            f"Voice/tone/positioning preserved, visuals updated from assets.\n\n"
            "Use /brand to review the updated config.",
        )
    except Exception as e:
        logger.error("Regen guidelines failed: %s", e)
        await update.message.reply_text(
            f"Failed to regenerate guidelines: {_esc(str(e))}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# LoRA commands
# ---------------------------------------------------------------------------


async def train_lora_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /train_lora — trigger LoRA training on Replicate."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    stats = lora_pipeline.get_training_stats()
    total = stats["total_images"]
    threshold = stats["threshold"]

    if total < threshold:
        await update.message.reply_text(
            f"Not enough training images yet.\n\n"
            f"Images: <b>{total}</b> / {threshold} required\n"
            f"Keep approving drafts \u2014 each /approve adds to the training set.",
            parse_mode="HTML",
        )
        return

    await update.message.chat.send_action("typing")
    await update.message.reply_text("Starting LoRA training on Replicate...")

    try:
        result = await lora_pipeline.trigger_training(
            bot=context.bot,
            chat_id=update.effective_user.id,
        )
    except Exception as e:
        logger.error("train_lora failed: %s", e)
        await update.message.reply_text(
            f"Training failed: {_esc(str(e))}", parse_mode="HTML"
        )
        return

    if result.get("error"):
        await update.message.reply_text(
            f"Training error: {_esc(result['error'])}", parse_mode="HTML"
        )
        return

    await update.message.reply_text(
        f"LoRA training started!\n\n"
        f"Version: <b>{_esc(result.get('version', '?'))}</b>\n"
        f"Prediction ID: <code>{_esc(result.get('prediction_id', '?'))}</code>\n"
        f"Images: {result.get('image_count', '?')}\n"
        f"Trigger word: <code>{_esc(result.get('trigger_word', 'BRAND3D'))}</code>\n\n"
        f"Polling in background \u2014 I'll notify you when training completes "
        f"and auto-download the weights.\n"
        f"Use /lora_status to check progress.",
        parse_mode="HTML",
    )


async def lora_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_status — show LoRA training status and version history."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    stats = lora_pipeline.get_training_stats()
    total = stats["total_images"]
    threshold = stats["threshold"]
    versions = stats["versions"]
    lora_manifest = stats.get("lora_manifest", {})

    lora = lora_pipeline.get_active_lora()

    lines = ["<b>LoRA Training Status</b>\n"]

    if lora:
        lines.append(f"Active LoRA: <b>{_esc(lora.get('version', '?'))}</b>")
        lines.append(f"Trigger word: <code>{_esc(lora.get('trigger_word', 'BRAND3D'))}</code>")
        lines.append(f"Weights: <code>{_esc(lora.get('weights_path', 'N/A'))}</code>")
    else:
        lines.append("Active LoRA: <i>none</i>")

    lora_versions = lora_manifest.get("versions", [])
    lines.append(f"\nTotal versions: <b>{len(lora_versions)}</b>")
    lines.append(f"Training images: <b>{total}</b> / {threshold}")

    if lora_versions:
        lines.append(f"\n<b>Trained versions:</b>")
        lines.append(lora_pipeline.format_versions_list(lora_manifest))
    elif versions:
        lines.append("\n<b>Training history:</b>")
        for v in versions[-5:]:
            status_icon = {"completed": "\u2705", "training": "\u23F3", "failed": "\u274C"}.get(v.get("status", ""), "\u2753")
            lines.append(
                f"  {status_icon} {_esc(v.get('version', '?'))} \u2014 "
                f"{_esc(v.get('status', '?'))} "
                f"({v.get('image_count', '?')} images)"
            )

    lines.append("\nUse /lora_versions for details, /lora_switch N to change.")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def lora_versions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_versions — list all trained LoRA versions."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    manifest = lora_pipeline.get_lora_manifest()
    versions = manifest.get("versions", [])

    if not versions:
        await update.message.reply_text(
            "No LoRA versions trained yet.\n"
            "Use /train_lora to start training.",
        )
        return

    formatted = lora_pipeline.format_versions_list(manifest)
    await update.message.reply_text(
        f"<b>LoRA Versions</b>\n\n{formatted}\n\n"
        f"Use /lora_switch <i>N</i> to switch, /lora_rollback to revert.",
        parse_mode="HTML",
    )


async def lora_switch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_switch <N> — switch the active LoRA to version N."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    args = (update.message.text or "").split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        await update.message.reply_text(
            "Usage: /lora_switch <i>N</i>\n\n"
            "Example: <code>/lora_switch 2</code>",
            parse_mode="HTML",
        )
        return

    version_str = args[1].strip().lstrip("v")
    try:
        version_num = int(version_str)
    except ValueError:
        await update.message.reply_text(
            f"Invalid version number: {_esc(args[1].strip())}\n"
            f"Use /lora_versions to see available versions.",
            parse_mode="HTML",
        )
        return

    result = lora_pipeline.switch_active_version(version_num)

    if isinstance(result, str):
        await update.message.reply_text(
            f"Switch failed: {_esc(result)}",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            f"Switched active LoRA to <b>v{version_num}</b>\n\n"
            f"Trigger word: <code>{_esc(result.get('trigger_word', 'BRAND3D'))}</code>\n"
            f"Training images: {result.get('image_count', '?')}\n"
            f"Weights copied to brand3d.safetensors",
            parse_mode="HTML",
        )


async def lora_rollback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lora_rollback — roll back to previous LoRA version."""
    if not _authorized(update.effective_user.id):
        return

    from agent import lora_pipeline

    result = lora_pipeline.rollback_version()

    if isinstance(result, str):
        await update.message.reply_text(
            f"Rollback failed: {_esc(result)}",
            parse_mode="HTML",
        )
    else:
        vn = result.get("version_number", "?")
        await update.message.reply_text(
            f"Rolled back to <b>v{vn}</b>\n\n"
            f"Trigger word: <code>{_esc(result.get('trigger_word', 'BRAND3D'))}</code>\n"
            f"Training images: {result.get('image_count', '?')}\n"
            f"Weights copied to brand3d.safetensors",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /reset_brand, /upload, /done
# ---------------------------------------------------------------------------


async def reset_brand_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset_brand — wipe brand config and start fresh."""
    if not _authorized(update.effective_user.id):
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    confirm_word = parts[1].strip() if len(parts) > 1 else ""

    user_id = update.effective_user.id

    if confirm_word.upper() == "RESET":
        brand_path = Path(settings.BRAND_FOLDER)

        gl = brand_path / "guidelines.md"
        if gl.exists():
            import shutil
            await _aio.to_thread(shutil.copy2, str(gl), str(gl) + ".bak")

        deleted = []
        for fname in ("config.json", "strategy.md", "content_calendar.md"):
            p = brand_path / fname
            if p.exists():
                p.unlink()
                deleted.append(fname)

        onboarding.delete_session(user_id)
        compositor_config.invalidate_cache()

        summary = ", ".join(deleted) if deleted else "no config files found"
        await update.message.reply_text(
            f"Brand reset complete.\n"
            f"Deleted: {summary}\n"
            f"guidelines.md backed up to guidelines.md.bak\n\n"
            f"Run /onboard to set up again.",
        )
    else:
        await update.message.reply_text(
            "This will wipe your brand config and start fresh.\n\n"
            "Type <code>/reset_brand RESET</code> to confirm.",
            parse_mode="HTML",
        )


async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /upload — set flag to receive photos as brand assets."""
    if not _authorized(update.effective_user.id):
        return

    context.user_data["awaiting_asset_upload"] = True
    context.user_data["_asset_upload_count"] = 0
    await update.message.reply_text(
        "Send me images to add to your brand library.\n"
        "I'll index them automatically. Send /done when finished.",
    )


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /done — clear the asset upload flag."""
    if not _authorized(update.effective_user.id):
        return

    was_uploading = context.user_data.pop("awaiting_asset_upload", False)
    count = context.user_data.pop("_asset_upload_count", 0)

    if was_uploading:
        await update.message.reply_text(
            f"Asset upload complete. {count} image(s) added to your library.\n"
            f"Use /library to browse your assets.",
        )
    else:
        await update.message.reply_text("Nothing to finish.")


# ---------------------------------------------------------------------------
# /health, /digest — monitoring commands
# ---------------------------------------------------------------------------


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /health -- show current system health status."""
    if not _authorized(update.effective_user.id):
        return

    await update.message.chat.send_action("typing")
    try:
        from agent.health_monitor import run_health_checks
        health = await run_health_checks()
        await update.message.reply_text(
            f"<b>System Health</b>\n\n<pre>{_esc(health.summary())}</pre>",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Health check command failed: %s", e)
        await update.message.reply_text(f"Health check failed: {_esc(str(e))}", parse_mode="HTML")


async def digest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /digest -- force generate daily digest."""
    if not _authorized(update.effective_user.id):
        return

    await update.message.chat.send_action("typing")
    try:
        from agent.digest import generate_daily_digest
        report_path = await generate_daily_digest()
        if report_path:
            with open(report_path, "rb") as f:
                await update.message.reply_document(
                    document=f,
                    filename=Path(report_path).name,
                    caption="<b>Daily Digest</b> -- generated on demand.",
                    parse_mode="HTML",
                )
        else:
            await update.message.reply_text("Failed to generate daily digest.")
    except Exception as e:
        logger.error("Digest command failed: %s", e)
        await update.message.reply_text(f"Digest generation failed: {_esc(str(e))}", parse_mode="HTML")
