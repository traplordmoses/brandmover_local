# BrandMover Local — Architecture Guide

## What This Is

A Telegram bot that generates on-brand social media content (text + images) using Claude for text and Replicate for image generation. It posts to X/Twitter and supports auto-posting on a schedule.

## Project Structure

```
agent/               Core logic — no Telegram dependency
  # Core generation engine
  engine.py            Tool-use agent loop + post-processing pipeline
  tools.py             Tool definitions and handlers for agent mode
  skill_prompt.py      System prompt builder
  context_engine.py    Budget-aware context assembly for system prompt
  content_types.py     Canonical content type definitions (single source of truth)
  _client.py           Anthropic client singleton
  model_fallback.py    Multi-model fallback for API calls

  # Legacy (deprecated — kept for backward compat)
  brain.py, unified_brain.py, unified_tools.py, unified_prompt.py

  # Image & composition
  image_gen.py         Replicate image generation (Flux, Seedream, Recraft, Nano-Banana)
  compositor.py        PIL image composition (glass-morphism backgrounds, text overlay)
  compositor_config.py Brand config parser (guidelines.md → BrandConfig dataclass)
  asset_gen.py         Standalone asset generation (/generate, /logo)
  asset_library.py     Asset storage and retrieval
  asset_pipeline.py    Asset processing pipeline
  asset_ingest.py      Brand asset ingestion via Claude Vision
  font_manager.py      Font loading and management
  template_*.py        Template system (spec, renderer, generator, memory)

  # Video & audio
  video_gen.py         Video generation
  video_reverse.py     Video style reverse-engineering
  video_styler.py      Video style application
  audio_pipeline.py    Audio generation pipeline
  scene_analysis.py    Scene analysis for video
  demo_recorder.py, demo_narrator.py, smart_recorder.py  Demo recording

  # State & persistence
  state.py             Pending draft management + draft versioning
  state_manager.py     FileStore abstraction for JSON state files
  auto_state.py        Auto-post scheduler state (rate limiting, dedup)
  feedback.py          Feedback log + learned preferences (Claude-summarized)
  generation_history.py  Append-only generation log with cost tracking (auto-rotated)
  session.py           Session memory (recent posts, rejections, preferences)
  session_plan.py      Session-level content planning
  paths.py             Canonical path definitions + state file migration

  # Publishing & channels
  publisher.py         X/Twitter posting via tweepy
  publish_queue.py     Async publish queue
  platform_adapter.py  Platform-specific formatting
  discord_publisher.py Discord publishing
  discord_bot.py       Discord bot integration
  channels/            Channel abstraction (base, twitter, discord, registry)

  # Scheduling & planning
  scheduler.py         Cron-like schedule parser
  schedule_queue.py    Scheduled post queue
  heartbeat.py         Health heartbeat
  content_planner.py   Content calendar planning
  topic_bank.py        Topic ideas bank
  topic_refresh.py     Topic refresh logic

  # Quality gates & scoring
  scoring.py           Weighted quality scoring
  self_review.py       Draft quality gate (default-FAIL checks)
  self_review_scheduler.py  Scheduled self-review
  dedup.py             Caption deduplication
  risk_score.py        Risk scoring (compliance flags)
  brand_check.py       Brand guideline compliance
  brand_alignment.py   Brand alignment scoring
  asset_audit.py       Asset audit checks

  # Learning & preferences
  pref_extractor.py    Auto-extract preferences from feedback patterns
  preference_engine.py Preference-based scoring
  memory.py            Searchable generation memory
  diversity_tracker.py Structural diversity tracking
  skeleton_library.py  Content structure templates

  # Brand context
  guidelines.py        Brand context loader with mtime-based caching
  guidelines_editor.py Conversational guidelines editing
  ingest.py            Legacy brand ingestion
  figma.py             Figma design reference integration
  context_feed.py      External context feed

  # Utilities
  resource_log.py      Resource usage tracking
  cost_gate.py         Cost limit enforcement
  net_guard.py         Network request guardrails
  hooks.py             Lifecycle hooks
  intent_router.py     Intent classification
  conversation_context.py  Conversation context management
  chat.py              Conversational mode
  subagent.py          Sub-agent delegation
  skills.py            Skill definitions
  refinement.py        Draft refinement logic
  onchain.py           On-chain event fetching
  lora_pipeline.py     LoRA training data collection
  campaigns.py         Campaign management
  campaign_preview.py  Campaign HTML preview
  calendar_generator.py  Content calendar output
  report_generator.py  Report HTML generation
  digest.py            Daily/weekly digest
  weekly_digest.py     Weekly digest scheduling
  health_monitor.py    System health monitoring
  onboarding.py        Brand onboarding flow
  performance.py       Performance metrics
  strategy.py          Content strategy
  web_fetch.py         Web content fetching

bot/                 Telegram interface
  telegram_bot.py      Bot setup, handler registration, scheduler launch
  handlers/            Command and message handlers (split by concern)
    core.py            Auth, rate limiting, help, shared utilities
    draft.py           Draft approval/rejection/revision flow
    generation.py      Content generation + intent routing
    media.py           Photo/document upload handling
    admin.py           Admin commands (health, digest, onboarding)
    scheduling.py      Schedule management commands
    debug.py           Debug/diagnostic commands

config/              Configuration
  settings.py          .env loader with startup validation
  schedule.json        Auto-post time slots

scripts/             Standalone scripts
  auto_post.py         Background scheduler loop
  bootstrap_brand.py   PDF → guidelines.md bootstrap
  extract_voice.py     Brand voice extraction
  record_demo.py       Demo recording script

dashboard/           Web dashboard (React frontend)
tests/               pytest test suite
brand/               Brand assets (guidelines.md, prompts/, references/, loras/)
state/               Runtime state files (gitignored, auto-rotated)
eval/                Evaluation framework
docs/                Documentation
demos/               Demo recordings
video/               Video assets and templates
```

## Key Patterns

- **Agent mode**: The agent loop in `engine.py` uses Claude's tool-use API with tools defined in `tools.py`. Post-processing (quality gates, scoring, dedup, risk) is handled by `_post_process_draft()`.
- **State files are JSON in `state/`**: `state.json`, `feedback.json`, `generation_history.json`, `auto_post_state.json`. All have migration logic from old root-level locations. Append-only logs auto-rotate on startup and at write time.
- **Async wrappers**: Blocking file I/O is wrapped in `asyncio.to_thread()` (e.g., `async_save_pending`, `async_log_feedback`) for non-blocking Telegram handlers.
- **Brand config caching**: `compositor_config.get_config()` and `guidelines.get_brand_context()` use mtime-based caching. Call `invalidate_cache()` / `invalidate_brand_context()` after modifying brand files.
- **Content type routing**: `content_types.py` is the single source of truth. Image model selection lives in `image_gen._select_model()`. Compositor profile mapping lives in `COMPOSITOR_PROFILE_MAP`.
- **Handlers package**: `bot/handlers/` is split by concern (core, draft, generation, media, admin, scheduling, debug). The `__init__.py` re-exports all public names and provides stable test-facing aliases for internal functions.

## Deployment Constraints

- **Single-instance only**: State files in `state/` use in-process `threading.RLock` for thread safety and `os.replace()` for atomic writes. There is no inter-process or distributed locking. Running multiple bot instances against the same `state/` directory will cause race conditions and data loss.
- **Multi-brand isolation**: For multiple brands, use separate `BRAND_FOLDER` and `STATE_FOLDER` env vars per instance. Each instance must have its own state directory.
- **OpenClaw script allowlist**: Default allowlist is hardcoded. Override by creating `brand/openclaw_allowlist.txt` (one script name per line).
- **Tool result truncation**: Configurable via `AGENT_TOOL_RESULT_MAX_CHARS` (default: 15000). Increase for workflows that need large tool outputs.

## Running

```bash
cp .env.example .env   # fill in API keys
pip install -r requirements.txt
python main.py
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Common Tasks

- **Add a new content type**: Add to `ALL_CONTENT_TYPES` in `content_types.py`, update `COMPOSITOR_PROFILE_MAP`, and optionally add to `LORA_ELIGIBLE_TYPES` / `AGENT_SELECTABLE_TYPES`.
- **Add a Telegram command**: Add handler in the appropriate `bot/handlers/*.py` submodule, register in `bot/telegram_bot.py`, update help text in `help_command`.
- **Customize image prompts**: Create `brand/prompts/{asset_type}.txt` with `{description}`, `{style_keywords}`, `{colors}`, `{background}` placeholders.
- **Tune compositor visuals**: Add a `## VISUAL EFFECTS` table to `brand/guidelines.md` with Glass opacity/blur/radius/inset and Orb alpha/count values.
