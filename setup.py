"""
BrandMover Local — Interactive Setup Wizard.
Run: python3 setup.py
"""

import os
import sys
from pathlib import Path


def _input(prompt: str, default: str = "") -> str:
    """Prompt for input with an optional default."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def _input_secret(prompt: str) -> str:
    """Prompt for a secret value (shows asterisks for length)."""
    value = input(f"{prompt}: ").strip()
    if value:
        print(f"  (set: {'*' * min(len(value), 8)}...)")
    return value


def _yes_no(prompt: str, default: bool = True) -> bool:
    """Yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{hint}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _test_anthropic(api_key: str) -> bool:
    """Quick connectivity test for Anthropic."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with OK"}],
        )
        return bool(resp.content)
    except Exception as e:
        print(f"  Anthropic test failed: {e}")
        return False


def _test_openai(api_key: str) -> bool:
    """Quick connectivity test for OpenAI."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with OK"}],
        )
        return bool(resp.choices)
    except Exception as e:
        print(f"  OpenAI test failed: {e}")
        return False


def _test_telegram(token: str) -> bool:
    """Quick connectivity test for Telegram bot token."""
    try:
        import httpx
        resp = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        data = resp.json()
        if data.get("ok"):
            bot_name = data["result"].get("username", "unknown")
            print(f"  Bot connected: @{bot_name}")
            return True
        print(f"  Telegram error: {data.get('description', 'unknown')}")
        return False
    except Exception as e:
        print(f"  Telegram test failed: {e}")
        return False


def main():
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"

    print("=" * 60)
    print("  BrandMover Local — Setup Wizard")
    print("=" * 60)
    print()
    print("This wizard will create your .env configuration file")
    print("and set up the brand folder structure.")
    print()

    if env_path.exists():
        if not _yes_no(".env already exists. Overwrite?", default=False):
            print("Setup cancelled.")
            return

    # --- Brand ---
    print("\n--- Brand Identity ---")
    brand_name = _input("Brand name", "MyBrand")
    brand_folder = _input("Brand folder path", "./brand")

    # Create brand folder structure
    brand_path = Path(brand_folder)
    if not brand_path.is_absolute():
        brand_path = project_root / brand_folder

    for subdir in ["examples/articles", "examples/images", "references", "assets"]:
        (brand_path / subdir).mkdir(parents=True, exist_ok=True)

    guidelines_path = brand_path / "guidelines.md"
    if not guidelines_path.exists():
        guidelines_path.write_text(
            f"# {brand_name} Brand Guidelines\n\n"
            "## Brand Voice\n"
            "- [Describe your brand's tone and personality]\n\n"
            "## Visual Style\n"
            "- [Colors, typography, imagery guidelines]\n\n"
            "## Hashtags\n"
            "- [List approved hashtags]\n\n"
            "## Never Use\n"
            "- [Words and phrases to avoid]\n",
            encoding="utf-8",
        )
        print(f"  Created template: {guidelines_path}")
    else:
        print(f"  Found existing: {guidelines_path}")

    # --- LLM Provider ---
    print("\n--- LLM Provider ---")
    print("  1. Anthropic (Claude) — recommended")
    print("  2. OpenAI (GPT-4o)")
    print("  3. Google (Gemini)")
    provider_choice = _input("Choose provider (1/2/3)", "1")
    provider_map = {"1": "anthropic", "2": "openai", "3": "gemini"}
    llm_provider = provider_map.get(provider_choice, "anthropic")

    anthropic_key = ""
    openai_key = ""
    gemini_key = ""

    if llm_provider == "anthropic":
        anthropic_key = _input_secret("Anthropic API key")
        if anthropic_key and _yes_no("Test Anthropic connection?"):
            if _test_anthropic(anthropic_key):
                print("  Anthropic API: OK")
            else:
                print("  Warning: Anthropic test failed. Check your key.")
    elif llm_provider == "openai":
        openai_key = _input_secret("OpenAI API key")
        if openai_key and _yes_no("Test OpenAI connection?"):
            if _test_openai(openai_key):
                print("  OpenAI API: OK")
            else:
                print("  Warning: OpenAI test failed. Check your key.")
    elif llm_provider == "gemini":
        gemini_key = _input_secret("Gemini API key")

    # --- Telegram ---
    print("\n--- Telegram Bot ---")
    print("  Create a bot via @BotFather on Telegram to get a token.")
    telegram_token = _input_secret("Telegram bot token")
    telegram_user_id = _input("Your Telegram user ID (numeric)", "0")

    if telegram_token and _yes_no("Test Telegram bot token?"):
        if _test_telegram(telegram_token):
            pass
        else:
            print("  Warning: Telegram test failed. Check your token.")

    # --- Image Generation ---
    print("\n--- Image Generation (Replicate) ---")
    print("  Get a token at https://replicate.com/account/api-tokens")
    replicate_token = _input_secret("Replicate API token (optional, press Enter to skip)")
    image_model = _input("Image model routing", "auto")

    # --- X / Twitter ---
    print("\n--- X / Twitter (optional) ---")
    setup_x = _yes_no("Configure X/Twitter posting?", default=False)
    x_api_key = ""
    x_api_secret = ""  # nosec B105 — empty default, overwritten by user input
    x_access_token = ""  # nosec B105
    x_access_secret = ""  # nosec B105
    x_bearer_token = ""  # nosec B105
    if setup_x:
        x_api_key = _input_secret("X API Key")
        x_api_secret = _input_secret("X API Secret")
        x_access_token = _input_secret("X Access Token")
        x_access_secret = _input_secret("X Access Secret")
        x_bearer_token = _input_secret("X Bearer Token")

    # --- Pipeline / Agent Mode ---
    print("\n--- Mode Configuration ---")
    print("  pipeline: 4-step generation (analyze > plan > verify > generate)")
    print("  agent: Claude tool-use loop (smarter, more flexible)")
    agent_mode = _input("Mode (pipeline/agent)", "agent")
    pipeline_mode = _input("Pipeline sub-mode (full/fast)", "full")
    agent_model = _input("Agent model", "claude-sonnet-4-6")

    # --- Figma (optional) ---
    print("\n--- Figma (optional) ---")
    setup_figma = _yes_no("Configure Figma integration?", default=False)
    figma_token = ""  # nosec B105 — empty default, overwritten by user input
    figma_file_key = ""
    figma_node_id = "0:5"
    if setup_figma:
        figma_token = _input_secret("Figma access token")
        figma_file_key = _input("Figma file key", "")
        figma_node_id = _input("Figma node ID", "0:5")

    # --- Write .env ---
    print("\n--- Writing .env ---")
    env_lines = [
        f"# BrandMover Local — generated by setup.py",
        f"",
        f"# --- LLM Provider ---",
        f"LLM_PROVIDER={llm_provider}",
        f"ANTHROPIC_API_KEY={anthropic_key}",
        f"OPENAI_API_KEY={openai_key}",
        f"GEMINI_API_KEY={gemini_key}",
        f"",
        f"# --- Image Generation (Replicate) ---",
        f"REPLICATE_API_TOKEN={replicate_token}",
        f'IMAGE_MODEL={image_model}',
        f"",
        f"# --- Telegram ---",
        f"TELEGRAM_BOT_TOKEN={telegram_token}",
        f"TELEGRAM_ALLOWED_USER_ID={telegram_user_id}",
        f"",
        f"# --- X / Twitter ---",
        f"X_API_KEY={x_api_key}",
        f"X_API_SECRET={x_api_secret}",
        f"X_ACCESS_TOKEN={x_access_token}",
        f"X_ACCESS_SECRET={x_access_secret}",
        f"X_BEARER_TOKEN={x_bearer_token}",
        f"",
        f"# --- Brand ---",
        f"BRAND_FOLDER={brand_folder}",
        f"BRAND_NAME={brand_name}",
        f"",
        f"# --- Pipeline ---",
        f"PIPELINE_MODE={pipeline_mode}",
        f"MAX_REFERENCE_CHARS=50000",
        f"",
        f"# --- Agent Mode ---",
        f"AGENT_MODE={agent_mode}",
        f"AGENT_MAX_TURNS=15",
        f"AGENT_MODEL={agent_model}",
        f"FEEDBACK_SUMMARIZE_EVERY=10",
        f"",
        f"# --- Figma ---",
        f"FIGMA_ACCESS_TOKEN={figma_token}",
        f"FIGMA_FILE_KEY={figma_file_key}",
        f"FIGMA_NODE_ID={figma_node_id}",
    ]

    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(f"  Wrote: {env_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print()
    print(f"  Brand:      {brand_name}")
    print(f"  Mode:       {agent_mode}")
    print(f"  LLM:        {llm_provider}")
    print(f"  Brand dir:  {brand_path}")
    print(f"  Telegram:   {'configured' if telegram_token else 'not set'}")
    print(f"  Replicate:  {'configured' if replicate_token else 'not set'}")
    print(f"  X/Twitter:  {'configured' if x_api_key else 'not set'}")
    print(f"  Figma:      {'configured' if figma_token else 'not set'}")
    print()
    print("Next steps:")
    print(f"  1. Edit {brand_path / 'guidelines.md'} with your brand guidelines")
    print(f"     (or use /setup in Telegram to bootstrap from a PDF)")
    print(f"  2. Drop reference PDFs into {brand_path / 'references/'}")
    print(f"  3. Run: python3 main.py")
    print()


if __name__ == "__main__":
    main()
