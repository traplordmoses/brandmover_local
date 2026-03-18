---
name: repurpose-content
description: Transform one piece of content into multiple formats — thread, post, image caption, newsletter
category: repurposing
priority: 2
dependencies: [brain.py, unified_brain.py, content_types.py, image_gen.py, compositor.py, guidelines.py, brand_check.py]
---

# Repurpose Content

## Purpose
Take a single piece of source content (blog post, thread, announcement, talk transcript) and systematically transform it into multiple output formats. Maximizes content ROI while maintaining brand voice across every derivative.

## Trigger Conditions
- A long-form piece has been published and needs social distribution
- Operator provides source content and requests multiple formats
- Campaign plan requires the same message across different content types

## Inputs
- Source content (text, URL, or file reference)
- Source format (blog, thread, transcript, announcement)
- Target formats (default: all applicable)
- Priority ranking of outputs (which to produce first)
- Optional: specific angles to emphasize per format

## Execution Steps
1. Load and parse source content — extract key themes, quotes, data points
2. Load brand context via `guidelines.get_brand_context()`
3. Identify repurposing targets from `content_types.ALL_CONTENT_TYPES`
4. Generate single-post distillation (280 chars, strongest insight)
5. Generate thread version (5-7 posts, educational arc)
6. Generate image caption (punchy, pairs with visual)
7. Generate image prompt for `image_gen.py` based on core visual concept
8. Generate newsletter section (2-3 paragraphs, more formal)
9. Run all outputs through `brand_check.check_brand_compliance()`
10. Package outputs as array with format labels

## Output Format
```json
{
  "source_summary": "string",
  "outputs": [
    {"format": "single_post", "content": "string", "char_count": 240},
    {"format": "thread", "posts": ["string", "string", "..."]},
    {"format": "image_caption", "content": "string"},
    {"format": "image_prompt", "content": "string"},
    {"format": "newsletter_section", "content": "string"}
  ]
}
```

## Quality Checks
- Each output stands alone (no "as mentioned in the blog" references)
- Single post captures the single most compelling insight
- Thread follows generate-thread quality standards
- Image caption is under 200 chars and visually evocative
- Newsletter section has a different tone — slightly more formal, still FOID voice
- No output is a direct copy-paste from source (all are rewritten)

## Tools & Modules Used
- `brain.py` / `unified_brain.py` — LLM rewriting per format
- `content_types.py` — target format definitions
- `image_gen.py` — visual asset generation
- `compositor.py` — image composition if needed
- `guidelines.py` — brand voice consistency
- `brand_check.py` — compliance validation per output

## Edge Cases & Learnings
<!-- populated through use -->

## Examples

**Example: repurpose "What is Loreboard?" blog post**
- Single post: "loreboard isn't a feed. it's a grid. curated by humans. no algorithm decides what matters — you do."
- Thread: 6-post thread covering origin story, how it works, why curation > algorithms, cultural context, MiFOID tie-in, CTA
- Image caption: "curation is identity"
- Image prompt: "minimal dark grid layout with glowing meme thumbnails, purple accent lighting, foid.fun aesthetic"
- Newsletter: 3 paragraphs on loreboard's design philosophy, positioned as counter-narrative to algorithmic feeds
