# BrandMover Local — Architecture Guide

## What This Is

A Telegram bot that generates on-brand social media content (text + images) using Claude for text and Replicate for image generation. It posts to X/Twitter and supports auto-posting on a schedule.

## Project Structure

```
agent/           Core logic — no Telegram dependency
  brain.py         Claude LLM calls (pipeline + agent modes)
  engine.py        Tool-use agent loop (8 registered tools)
  tools.py         Tool definitions and handlers for agent mode
  image_gen.py     Replicate image generation (Flux, Seedream, Recraft, Nano-Banana)
  compositor.py    PIL image composition (glass-morphism backgrounds, text overlay)
  compositor_config.py  Brand config parser (guidelines.md → BrandConfig dataclass)
  content_types.py      Canonical content type definitions (single source of truth)
  asset_gen.py     Standalone asset generation (/generate, /logo)
  guidelines.py    Brand context loader with mtime-based caching
  state.py         Pending draft management + draft versioning + style profiles
  auto_state.py    Auto-post scheduler state (rate limiting, dedup)
  feedback.py      Feedback log + learned preferences (Claude-summarized)
  generation_history.py  Append-only generation log with cost tracking
  lora_pipeline.py LoRA training data collection, training trigger, polling
  ingest.py        Brand ingestion via Claude Vision
  publisher.py     X/Twitter posting via tweepy
  scheduler.py     Cron-like schedule parser
  onchain.py       On-chain event fetching
  skill_prompt.py  System prompt builder
  figma.py         Figma design reference integration
  resource_log.py  Resource usage tracking

bot/             Telegram interface
  telegram_bot.py  Bot setup, handler registration, scheduler launch
  handlers.py      All command and message handlers

config/          Configuration
  settings.py      .env loader with startup validation
  schedule.json    Auto-post time slots

scripts/         Standalone scripts
  auto_post.py     Background scheduler loop
  bootstrap_brand.py  PDF → guidelines.md bootstrap
  extract_voice.py    Brand voice extraction

tests/           pytest test suite
brand/           Brand assets (guidelines.md, prompts/, references/, loras/)
state/           Runtime state files (gitignored)
eval/            Evaluation framework
```

## Key Patterns

- **Pipeline vs Agent mode**: `AGENT_MODE=pipeline` uses a multi-step pipeline (analyze → plan → verify → generate). `AGENT_MODE=agent` uses a Claude tool-use loop with 8 tools.
- **State files are JSON in `state/`**: `state.json`, `feedback.json`, `generation_history.json`, `auto_post_state.json`. All have migration logic from old root-level locations.
- **Async wrappers**: Blocking file I/O is wrapped in `asyncio.to_thread()` (e.g., `async_save_pending`, `async_log_feedback`) for non-blocking Telegram handlers.
- **Brand config caching**: `compositor_config.get_config()` and `guidelines.get_brand_context()` use mtime-based caching. Call `invalidate_cache()` / `invalidate_brand_context()` after modifying brand files.
- **Content type routing**: `content_types.py` is the single source of truth. Image model selection lives in `image_gen._select_model()`. Compositor profile mapping lives in `COMPOSITOR_PROFILE_MAP`.

## Running

```bash
cp .env.example .env   # fill in API keys
pip install -r requirements.txt
python main.py
```

## Testing

```bash
python -m pytest tests/ -v
```

## Common Tasks

- **Add a new content type**: Add to `ALL_CONTENT_TYPES` in `content_types.py`, update `COMPOSITOR_PROFILE_MAP`, and optionally add to `LORA_ELIGIBLE_TYPES` / `AGENT_SELECTABLE_TYPES`.
- **Add a Telegram command**: Add handler in `bot/handlers.py`, register in `bot/telegram_bot.py`, update help text in `help_command`.
- **Customize image prompts**: Create `brand/prompts/{asset_type}.txt` with `{description}`, `{style_keywords}`, `{colors}`, `{background}` placeholders.
- **Tune compositor visuals**: Add a `## VISUAL EFFECTS` table to `brand/guidelines.md` with Glass opacity/blur/radius/inset and Orb alpha/count values.
