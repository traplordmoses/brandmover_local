# Kinetic Typography Video — SKILL.md
---
name: kinetic-typography-video
description: Build premium kinetic typography promo videos using Playwright HTML→PNG frames + ffmpeg. Word-by-word reveals, staggered animations, crossfade transitions. Revolut/Coinbase quality.
---

## When to Use
- Operator wants a premium promo/explainer video
- Remotion generator produces flat fades or stuck frames
- Need word-by-word kinetic text reveals with real personality
- Target: 20–60 second square/portrait/landscape video

## How It Works

### Pipeline
1. **Plan scenes** — agree on scene table with Moses before building
2. **Build HTML frames** with Playwright — each scene = multiple keyframe PNGs
3. **Stitch with ffmpeg** — crossfade transitions between scenes
4. **Review** — check duration, typography, transitions
5. **Send**

---

## Core Technique: HTML → PNG Frames

Each scene is rendered as a series of PNG frames using Playwright with animated CSS.

### Frame Structure
```python
from playwright.sync_api import sync_playwright
import os, time

os.makedirs('state/outputs/frames', exist_ok=True)

# Each scene: render multiple frames at different animation states
scenes = [
    {
        "id": 1,
        "duration": 4,  # seconds
        "lines": ["remember when", "the internet", "was fun?"],
        "color": "#a78bfa",  # purple
        "size": "72px",
        "weight": "800",
    },
    # ... more scenes
]
```

### HTML Template (kinetic typography)
```html
<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800;900&display=swap');
  
  * { margin: 0; padding: 0; box-sizing: border-box; }
  
  body {
    width: 1080px;
    height: 1080px;
    background: #0a0a0a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Inter', sans-serif;
    overflow: hidden;
  }
  
  .scene {
    text-align: center;
    padding: 80px;
    max-width: 900px;
  }
  
  /* Word-by-word reveal */
  .word {
    display: inline-block;
    opacity: 0;
    transform: translateY(20px);
    animation: wordIn 0.4s ease forwards;
    margin: 0 8px;
  }
  
  @keyframes wordIn {
    to { opacity: 1; transform: translateY(0); }
  }
  
  /* Stagger each word */
  .word:nth-child(1) { animation-delay: 0.1s; }
  .word:nth-child(2) { animation-delay: 0.25s; }
  .word:nth-child(3) { animation-delay: 0.4s; }
  /* etc */
  
  /* Tag/label above main text */
  .tag {
    font-size: 14px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 24px;
  }
  
  /* Subtitle below main text */
  .subtitle {
    font-size: 22px;
    color: #888;
    margin-top: 20px;
    line-height: 1.5;
  }
  
  /* Logo image */
  .logo {
    width: 120px;
    margin-bottom: 32px;
    border-radius: 16px;
  }
</style>
</head>
<body>
<div class="scene">
  <div class="tag">🕯️ PROOF OF PRAYER</div>
  <h1 style="font-size:72px; color:#a78bfa; font-weight:900; line-height:1.1;">
    <span class="word">pray</span>
    <span class="word">daily</span>
    <br>
    <span class="word">with</span>
    <span class="word" style="color:#f472b6;">Foid Mommy.</span>
  </h1>
  <p class="subtitle">she remembers. it's on-chain.</p>
</div>
</body>
</html>
```

### Capturing Frames at Different Animation States
```python
with sync_playwright() as p:
    browser = p.chromium.launch()
    
    for scene in scenes:
        page = browser.new_page(viewport={'width': 1080, 'height': 1080})
        page.set_content(scene['html'])
        
        # Frame 1: early state (words mid-reveal)
        page.wait_for_timeout(300)
        page.screenshot(path=f'state/outputs/frames/scene_{scene["id"]}_f1.png')
        
        # Frame 2: mid state
        page.wait_for_timeout(400)
        page.screenshot(path=f'state/outputs/frames/scene_{scene["id"]}_f2.png')
        
        # Frame 3: full reveal (hold)
        page.wait_for_timeout(800)
        page.screenshot(path=f'state/outputs/frames/scene_{scene["id"]}_f3.png')
        
        page.close()
    
    browser.close()
```

---

## Color Palette Per Scene Type

| Scene Type | Primary Color | Accent |
|-----------|---------------|--------|
| Hook | `#a78bfa` (purple) | `#f472b6` (pink) |
| Problem | `#f87171` (red) | `#fbbf24` (amber) |
| Answer/Logo | `#ffffff` (white) | brand color |
| Feature 1 (Prayer) | `#a78bfa` (purple) | `#f472b6` |
| Feature 2 (Loreboard) | `#34d399` (green) | `#6ee7b7` |
| Feature 3 (Swipe) | `#f472b6` (pink) | `#fb7185` |
| Feature 4 (Gallery) | `#fbbf24` (amber) | `#f59e0b` |
| Feature 5 (MiFOID) | `#60a5fa` (blue) | `#a78bfa` |
| CTA | `#ffffff` | `#a78bfa` |

Background: always `#0a0a0a` (near-black, not pure black)

---

## ffmpeg Stitching with Crossfades

```python
import subprocess

# Build concat file with duration per frame
concat_lines = []
for frame_path, duration in frame_durations:
    concat_lines.append(f"file '{frame_path}'")
    concat_lines.append(f"duration {duration}")

with open('state/outputs/frames/concat.txt', 'w') as f:
    f.write('\n'.join(concat_lines))

# Stitch frames
subprocess.run([
    'ffmpeg', '-y',
    '-f', 'concat', '-safe', '0',
    '-i', 'state/outputs/frames/concat.txt',
    '-vf', 'fps=30,scale=1080:1080',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-crs', '18',
    'state/outputs/raw_video.mp4'
], check=True)

# Add crossfade transitions between scenes
# Use xfade filter for smooth transitions
# Example: 2-scene crossfade
subprocess.run([
    'ffmpeg', '-y',
    '-i', 'scene1.mp4', '-i', 'scene2.mp4',
    '-filter_complex',
    '[0][1]xfade=transition=fade:duration=0.4:offset=3.6',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    'output.mp4'
], check=True)
```

### Crossfade for N scenes (loop approach)
```python
# Build individual scene clips first
# Then chain xfade filters
# offset = cumulative duration - fade_duration

scene_clips = ['scene1.mp4', 'scene2.mp4', 'scene3.mp4', ...]
fade_dur = 0.4

# Build filter_complex string
inputs = ' '.join([f'-i {c}' for c in scene_clips])
filter_parts = []
last_output = '[0]'

for i in range(1, len(scene_clips)):
    offset = sum(durations[:i]) - fade_dur * i
    out = f'[v{i}]' if i < len(scene_clips)-1 else ''
    filter_parts.append(
        f'{last_output}[{i}]xfade=transition=fade:duration={fade_dur}:offset={offset}{out}'
    )
    last_output = f'[v{i}]'

filter_complex = '; '.join(filter_parts)
```

---

## Scene Duration Guidelines (30s video)

| Scene | Duration |
|-------|----------|
| Hook | 4s |
| Problem | 4s |
| Answer/Logo | 5s |
| Feature 1 | 4s |
| Feature 2 | 4s |
| Feature 3 | 4s |
| Feature 4 | 3s |
| CTA | 2s |
| **Total** | **30s** |

---

## Typography Rules

- Font: Inter (Google Fonts) — weights 400, 700, 800, 900
- Main headline: 64–80px, weight 800–900
- Subtitle/body: 20–26px, weight 400, color #888
- Tag/label: 13–15px, letter-spacing 3px, uppercase, color #555–#666
- Line height: 1.1 for headlines, 1.5 for body
- Max width: 900px centered in 1080px canvas
- Word spacing: 8–12px margin between words for kinetic reveal

## Animation Timings

- Word reveal: 0.4s ease, staggered 0.15s per word
- Subtitle fade: 0.6s ease, delay = last word delay + 0.3s
- Tag fade: 0.3s ease, delay 0s (appears first)
- Logo scale: transform scale(0.8→1.0), 0.5s ease

---

## Quality Checklist

- [ ] Duration within ±1s of target
- [ ] No blank/stuck frames
- [ ] Each scene has tag + headline + subtitle
- [ ] Color changes per scene type
- [ ] Crossfade transitions present
- [ ] Text fully readable (not mid-animation in final frame)
- [ ] Logo appears in scene 3 (Answer)
- [ ] CTA scene ends clean

---

## Moses Preferences

- Agreed plan before building — always show scene table first
- "Revolut/Coinbase quality" = premium dark, kinetic text, no Remotion flat fades
- Brand voice: warm nostalgic, not tech-bro
- Schedule at 11:11am (on-brand prayer time)
- This technique was first approved on: FOID Foundation 30s promo "remember when the internet was fun?"
