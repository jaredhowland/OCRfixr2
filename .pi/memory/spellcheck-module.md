---
description: >
  Core spellcheck module — SymSpell + BERT dual verification, interactive mode, lazy loading, confidence thresholds.
tags:
  - spellcheck
  - BERT
  - SymSpell
  - core-module
---

# spellcheck.py — Core Engine

## Lazy Model Loading
BERT pipeline is a module-level singleton (`_unmasker`) loaded on first `fix()` call via `_get_unmasker()`. Includes warm-up inference on first load.

## Class: `spellcheck(text, **kwargs)`

### Constructor Parameters
| Param | Default | Purpose |
|-------|---------|---------|
| `changes_by_paragraph` | `"F"` | Return per-paragraph suggestions |
| `return_fixes` | `"F"` | Also return dict of corrections made |
| `ignore_words` | `[]` | Words to skip |
| `interactive` | `"F"` | Enable tkinter GUI for accept/reject |
| `common_scannos` | `"T"` | Enable direct scanno corrections |
| `top_k` | 15 | BERT suggestions to consider |
| `return_context` | `"F"` | Include local context in output |
| `suggest_unsplit` | `"T"` | Suggest splitting mashed words |
| `full_paragraphs` | `"T"` | Use full paragraph context for BERT |
| `custom_dict` | `None` | Extra valid words |
| `confidence_threshold` | 0.0 | Min BERT confidence to accept |
| `ignore_first_word` | `False` | Skip first word (split word detection) |
| `progress_file` | `None` | Save/load interactive decisions |

### Key Methods
- `_SPLIT_PARAGRAPHS(text)` → split text into paragraphs (respects BERT 500-word limit)
- `_LIST_MISREADS()` → find unrecognized words (drops caps, hyphens, numbers, footnotes, Roman numerals, etc.)
- `_FIND_REPLACEMENTS(misreads)` → SymSpell ∩ BERT intersection logic
- `SINGLE_STRING_FIX()` → fix a single paragraph
- `fix()` → main entry point (splits → per-paragraph fix → reassemble)

### `_LIST_MISREADS` Filtering Pipeline
1. Split on spaces/newlines
2. Drop: hyphenated words, apostrophes, ellipsis, numbers, leading caps, footnotes, Roman numerals, -eth/-est endings, format tags, list items
3. Strip trailing punctuation (keep contractions)
4. Drop 1-char tokens
5. Check against SCOWL_70 + custom dict
6. Skip paragraphs where >30% words are unrecognized
7. Remove ignore list words
8. Add common/stealth scannos if enabled

### `_FIND_REPLACEMENTS` Logic
1. Common scannos → direct correction (bypass BERT)
2. Stealth scannos → check BERT for target word
3. Other misreads → SymSpell candidates + BERT `[MASK]` fill
4. Intersection: single match = accepted, 0 or >1 = rejected
5. Confidence threshold filter (if > 0)
6. Homophone filter (double metaphone) — with o→e exception
7. Remove trivial s-dropping fixes
8. Interactive mode: show dialog for each suggestion

### Interactive Mode (`_CREATE_DIALOGUE`)
- Centered 700×350 window, non-resizable
- Left: scrollable context area with highlighted misspelled word
- Right: found word, suggested replacement, buttons (Update, Ignore, Save, Cancel All)
- Esc key = cancel all
- Decisions saved per-word (repeated misreads only prompt once)
- Progress file: `<input>.ocrfixr_progress.json`
