# BrandMover Local

An autonomous AI marketing agent that runs via Telegram. Send a natural language request, get a branded post draft with a generated image, review it, and publish to X with one command.

**Pipeline:** Telegram message -> Read brand guidelines -> LLM generates caption + hashtags + image prompt -> Replicate Flux 1.1 Pro generates image -> Draft sent to Telegram for review -> /approve posts to X.

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
- **/approve** — posts the draft to X
- **/reject make it more urgent** — revises the draft with your feedback
- **/status** — shows the current pending draft
- **/cancel** — clears the pending draft
- **/help** — shows available commands

Only messages from your authorized Telegram user ID are processed. Everyone else is ignored.
