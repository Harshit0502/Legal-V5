---
title: Guarded Section-wise LongT5 v6
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.30.0
python_version: 3.11
app_file: app.py
pinned: false
---

# Guarded Section-wise LongT5 v6

Quality-first LongT5 summarizer for 20–30 page legal/document PDFs.

## Architecture

```text
PDF
→ PyMuPDF text extraction
→ header/footer cleanup
→ input quality guard
→ conservative section detection
→ fallback to ordered token windows if section detection is weak
→ extractive factual skeleton
→ LongT5 section/window summaries
→ guarded retry if output is bad
→ final LongT5 re-summary
→ extractive fallback if final output is still bad
```

## Key upgrades

1. Detects repeated garbage output such as `us us us`, `kids kids`, `wood wood`.
2. Uses a factual skeleton to ground LongT5.
3. Keeps K-Means optional and OFF by default.
4. Uses section detection only when conservative rules are satisfied.
5. Automatically falls back to ordered windows.
6. Returns extractive fallback instead of hallucinated/repetitive text.

## Recommended settings

Balanced:
- Mode: Balanced
- Use extractive factual skeleton: ON
- Anchor method: None
- Max section/window input tokens: 3072
- Section/window summary tokens: 260
- Final summary tokens: 576
- Force ordered-window fallback: OFF first; turn ON if section detection behaves badly.

Fast:
- Mode: Fast
- Anchor method: None
- Max input tokens: 2048
- Section summary tokens: 160–180
- Final summary tokens: 320–360

## Model

Default:

```bash
SUM_MODEL_NAME=pszemraj/long-t5-tglobal-base-16384-book-summary
```

For best results, replace with your fine-tuned legal LongT5 model:

```bash
SUM_MODEL_NAME=Harshit0502/legal-longt5-sectionwise
```

 ```bash
Gardio_link = https://huggingface.co/spaces/harshitmahour360/LonGT5Summarizer
```
Use T4 GPU or better for 20–30 page PDFs.
