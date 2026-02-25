# BrandMover Local

An autonomous AI marketing agent that runs via Telegram. Send a natural language request, get a branded post draft with a generated image, review it, and publish to X with one command.

**Pipeline:** Telegram message -> Read brand guidelines -> LLM generates caption + hashtags + image prompt -> Smart model routing generates image -> Branded compositor overlays template -> Draft sent to Telegram for review -> /approve posts to X.

## Features

- **Agent mode** — Claude tool-use loop with brand guidelines, Figma, feedback history, and onchain scripts
- **Smart image routing** — auto-selects the best Replicate model per content type (Nano Banana for text overlays, Recraft SVG for brand assets, Seedream for lifestyle, Flux 1.1 Pro general)
- **brand_3d pipeline** — dedicated 3D asset generation with master prompt splicing, category-based reference image routing, optional LoRA trigger, and parallel N=3 option generation via `asyncio.gather`
- **Parallel image options** — brand_3d generates 3 options simultaneously; the CMO picks the best with `/approve 1`, `/approve 2`, or `/approve 3`
- **Surgical /edit** — apply targeted img2img edits to the last generated image without re-running the full pipeline (`/edit make the background darker`)
- **Adaptive compositor** — branded image templates with text overlays, logo placement, and platform badges
- **Style profiles** — named collections of reference images that apply a consistent visual style (e.g. Revolut-style 3D cards) via img2img
- **Finny mascot** — character-consistent generation using multi-reference stitched grids
- **Feedback learning** — learns from approve/reject history and auto-summarizes preferences
- **PDF brand bootstrap** — upload a brand guidelines PDF and auto-extract structured guidelines
- **OpenClaw integration** — execute onchain scripts for campaign logging, task management, vault reads

## Setup

### 1. Clone and install

```bash
cd /Users/bengalagan/brandmover_local
pip install -r requirements.txt
cp .env.example .env
```

### 2. Fill in `.env`

You need API keys for the following services:

**Telegram Bot Token:**
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`, follow prompts
3. Copy the token into `TELEGRAM_BOT_TOKEN`

**Your Telegram User ID:**
1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. It will reply with your user ID
3. Copy it into `TELEGRAM_ALLOWED_USER_ID`

**Replicate API Token (for image generation):**
1. Go to [replicate.com](https://replicate.com)
2. Sign up / log in
3. Go to Account Settings -> API tokens
4. Create a token and copy into `REPLICATE_API_TOKEN`

**X (Twitter) API Keys:**
1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Create a project and app
3. Under Keys and Tokens, generate:
   - API Key -> `X_API_KEY`
   - API Secret -> `X_API_SECRET`
   - Access Token -> `X_ACCESS_TOKEN`
   - Access Token Secret -> `X_ACCESS_SECRET`
   - Bearer Token -> `X_BEARER_TOKEN`
4. Make sure your app has Read and Write permissions

**LLM Provider (default: Anthropic):**
- Set `LLM_PROVIDER=anthropic` and fill `ANTHROPIC_API_KEY`
- Or use `openai` / `gemini` with their respective keys

### 3. Add brand guidelines

Edit `brand/guidelines.md` with your brand's voice, tone, colors, hashtags, and style rules. The agent reads this file before every generation to stay on-brand.

Add example posts as `.txt` files in `brand/examples/articles/` — the agent uses these as reference for tone and format.

### 4. Run

```bash
python main.py
```

## Usage

Just message the bot on Telegram:

- **"write our Token2049 recap post"** — generates a draft with image
- **/approve [N]** — approve the pending draft (option N if multiple images were generated)
- **/reject make it more urgent** — revises the draft with your feedback (re-runs full pipeline)
- **/edit make the background darker** — surgical img2img edit on the last generated image (strength 0.2, keeps composition)
- **/status** — shows the current pending draft
- **/cancel** — clears the pending draft
- **/refs** — show loaded reference materials
- **/feedback** — show approval/rejection stats
- **/learn** — trigger preference learning from feedback history
- **/style** — manage visual style profiles
- **/setup** — bootstrap guidelines from a PDF upload
- **/help** — shows available commands

Upload a photo to use as a reference image. Add a caption to immediately generate with it, or reply with `reference`, `finny`, `style <name>`, or `background`.

Only messages from your authorized Telegram user ID are processed. Everyone else is ignored.

## Style Profiles

Style profiles let you train the bot's visual identity from reference images. Upload references into named collections, activate a profile per content type, and every generation automatically applies that visual style.

### Quick start

```
/style create 3d_card Revolut-style 3D floating card visuals
```
Upload reference photos with caption `3d_card` to add them to the profile.

```
/style 3d_card announcement
```
Now every announcement request uses the 3D card references at 0.3 strength via img2img.

### Commands

| Command | Description |
|---------|-------------|
| `/style` | List all profiles with image counts and active mappings |
| `/style create <name> <description>` | Create a new profile |
| `/style <name> <content_type>` | Set profile as active for a content type |
| `/style <name> info` | Show profile details |
| `/style <name> remove` | Remove profile from all active mappings (keeps images) |

### How it works

1. Upload reference images with the profile name as caption — saved to `brand/references/styles/<name>/`
2. When a profile is active for a content type, `generate_image` stitches up to 3 refs into a grid and runs img2img at the profile's strength
3. `/approve` automatically saves the composed output back into the active profile, growing the collection over time
4. Profiles cap at 10 images, pruning the oldest automatically
5. Falls back to flat `approved_<type>_*.png` references if no profile is active

### Directory structure

```
brand/references/
  styles/
    3d_card/
      ref_1740350000.png
      ref_1740350100.png
    phone_mockup/
      ref_1740360000.png
  approved_announcement_*.png   <- existing flat approved images (fallback)
```

Profile metadata lives in `brand/styles.json`.
