# Brand Configuration

This folder holds your brand-specific assets and configuration. Most files here are gitignored — each instance maintains its own brand identity.

## Getting Started

1. Copy the example files to create your brand config:

```bash
cp brand/guidelines.md.example brand/guidelines.md
cp brand/styles.json.example brand/styles.json
cp brand/personality/system_prompt.md.example brand/personality/system_prompt.md
cp brand/personality/memory.md.example brand/personality/memory.md
```

2. Or use the bootstrap script to extract brand identity from a PDF:

```bash
python scripts/bootstrap_brand.py path/to/brand-deck.pdf
```

## Structure

```
brand/
  guidelines.md          Brand voice, colors, fonts, visual effects (main config)
  styles.json            Compositor style profiles
  config.json            Optional brand metadata
  personality/
    system_prompt.md     Agent personality and behavior rules
    memory.md            Persistent memory across conversations
    voice_rules.md       Voice/tone guidelines
  prompts/               Custom image prompt templates per content type
  references/            Reference articles and content examples
  examples/
    articles/            Example posts for few-shot learning
    images/              Reference images
  assets/
    fonts/               Brand fonts (.ttf, .otf)
    images/              Brand images (logos, backgrounds)
    library/             Asset library items
  templates/             Post templates
  training_data/         LoRA training data
  loras/                 Trained LoRA weights
  archive/               Archived content
```

## Notes

- All files except examples and `.gitkeep` placeholders are gitignored
- Run `python scripts/bootstrap_brand.py` to auto-generate from a brand PDF
- The compositor reads visual config from `guidelines.md` (see `## VISUAL EFFECTS` section)
