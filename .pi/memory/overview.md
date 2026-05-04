---
description: >
  OCRfixr2 codebase overview — dual-verification OCR spellchecker using SymSpell + BERT, with unsplit module and CLI.
tags:
  - overview
  - architecture
  - python
  - ocr
  - spellcheck
---

# OCRfixr2 — Codebase Overview

## What It Does
Corrects OCR misreads in digitized book text (primarily for [Distributed Proofreaders](https://www.pgdp.net/)) using a **dual-verification** approach: SymSpell (dictionary-based) cross-referenced with BERT (contextual masked language modeling). Also includes an **unsplit** module for fixing words split across hyphenated line boundaries.

## Package Layout
```
src/ocrfixr2/
  __init__.py          → exports spellcheck, unsplit
  spellcheck.py        → core engine (SymSpell + BERT, interactive tkinter UI, lazy model loading)
  unsplit.py           → fixes words split across lines with hyphens (rule-based)
  run_ocrfixr.py       → CLI entry point (batch, parallel, dry-run, glob, custom dict)
  data/
    SCOWL_70.txt       → 116,884 words (primary word list)
    SCOWL_20.txt       → 12,777 common words (unsplit disambiguation)
    Scannos_Common.txt → ~4,700 direct OCR error→correction mappings (bypass BERT)
    Scannos_Stealth.txt → valid-but-wrong word mappings
    Ignore_These_Misspells.txt → words to skip (foreign, etc.)
    Ignore_These_Suggestions.txt → known bad suggestions (currently empty)
```

## Data Flow
```
Input text → unsplit (merge split words) → spellcheck per-paragraph
  → SCOWL check → SymSpell candidates → BERT [MASK] fill → intersection → apply corrections
```

## Key Design Decisions
- **Change-averse**: only corrects when SymSpell AND BERT agree on exactly ONE candidate
- **Lazy model loading**: BERT loads on first `fix()` call, not at import
- **Not thread-safe**: BERT pipeline is a module-level singleton; use separate processes for parallel work
- **Paragraph context**: text split by `\n` (configurable via `changes_by_paragraph`)
- **Common scannos bypass BERT**: words in Scannos_Common.txt corrected directly without context verification

## Toolchain
- **Package manager**: `uv` (lock file: `uv.lock`)
- **Build backend**: `uv_build`
- **Python**: 3.12+
- **Linting**: `ruff`
- **Type checking**: `ty`
- **Testing**: `pytest` with mock BERT fixtures (`tests/conftest.py`)
- **CI**: GitHub Actions on push/PR — ruff → ty → pytest

## Commands
```bash
uv sync
uv run pytest tests/ -v
uv run ruff check --fix src/ tests/
uv run ty check --fix src/ tests/
uv run ocrfixr2 input.txt output.txt
```

## Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `OCRFIXR_MODEL` | `bert-base-uncased` | BERT model name |
| `OCRFIXR_MODEL_CACHE` | `src/ocrfixr2/model_cache/` | Model cache directory |
| `HF_TOKEN` | (unset) | Hugging Face token for faster downloads |
