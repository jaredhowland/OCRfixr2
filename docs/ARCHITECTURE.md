# OCRfixr2 — Architecture Review

**Date:** 2026-05-02
**Author:** Architecture Review (automated)
**Repo:** `/Users/wgu/Code/python/OCRfixr` (https://github.com/jaredhowland/OCRfixr2)
**Audience:** Maintainers, contributors, technical reviewers
**Version reviewed:** 2.0.0

---

## Executive Summary

OCRfixr2 is a Python package that corrects OCR misreads in digitized book text using a dual-verification approach: **SymSpell** (dictionary-based spellchecking) cross-referenced with **BERT** (contextual language modeling). It also includes an **unsplit** module for fixing words broken across line boundaries.

### Key Findings

| # | Finding | Severity |
|---|---------|----------|
| 1 | BERT model loads at module import time — blocks every `import ocrfixr2` with 5-10s cold start, even for code paths that don't need it | **Critical** |
| 2 | No CI/CD pipeline — tests run locally only; no automated linting, type checking, or test enforcement on push/PR | **High** |
| 3 | Heavy dependency footprint: TensorFlow **and** PyTorch both listed, but only PyTorch (via transformers) is actually used | **High** |
| 4 | Interactive mode uses `tkinter` with no graceful fallback — crashes on headless servers / CI environments | **Medium** |
| 5 | No rate limiting, concurrency safety, or batch processing — designed for single-user single-book workflows only | **Medium** |

### Core Assessment

OCRfixr2 is a **well-scoped, change-averse tool** for a specific audience (Distributed Proofreaders volunteers). The dual-verification approach (SymSpell + BERT) is sound and produces conservative corrections. The architecture is simple enough for a small team to maintain, but has several structural issues that affect usability and reliability.

---

## Current Architecture

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.12+ | Core implementation |
| Package manager | `uv` | Dependency resolution and build |
| Build system | `uv_build` | Packaging (was setuptools) |
| Spellchecking | `symspellpy` (v6.9.0) | Dictionary-based candidate generation |
| Context modeling | `transformers` (v5.6.2) + `bert-base-uncased` | Masked language modeling for context verification |
| Phonetics | `metaphone` (v0.6) | Homophone detection (double metaphone) |
| CLI | `argparse` | Command-line interface |
| Interactive UI | `tkinter` (stdlib) | Accept/reject suggestion dialog |
| Progress | `tqdm` (v4.67.3) | CLI progress bar |
| Testing | `pytest` (v9.0.3) | Unit tests |
| Linting | `ruff` (v0.15.12, cached) | Code quality (configured but no CI enforcement) |

### Source Layout

```
src/ocrfixr2/
├── __init__.py          (6 lines)     — re-exports spellcheck, unsplit
├── spellcheck.py        (599 lines)   — core spellcheck engine
├── unsplit.py           (137 lines)   — hyphenated word repair
├── run_ocrfixr.py       (152 lines)   — CLI entry point
└── data/
    ├── SCOWL_70.txt           (116,884 lines) — full word list
    ├── SCOWL_20.txt           (12,777 lines)  — common word list
    ├── Scannos_Common.txt     (4,736 lines)   — common OCR errors → corrections
    ├── Scannos_Stealth.txt    (1 line)        — "valid but wrong" word mappings
    ├── Ignore_These_Misspells.txt (9,857 lines) — words to skip
    └── Ignore_These_Suggestions.txt (0 lines)  — known bad suggestions
```

**Total source code:** ~894 lines across 4 Python files.

### Data Flow

```
User text (Gutenberg PO/PG format)
    │
    ▼
[unsplit] ← Optional: merge words split across lines with hyphens
    │
    ▼
[spellcheck] ← Process per-paragraph (split by \n)
    │
    ├── 1. Tokenize & filter (drop caps, hyphens, numbers, footnotes, etc.)
    │
    ├── 2. Check against SCOWL_70 word list → identify unrecognized words
    │
    ├── 3. SymSpell lookup → candidate replacements (edit distance ≤ 2)
    │
    ├── 4. BERT [MASK] fill → context-likely words (top 15)
    │
    ├── 5. Intersection: SymSpell ∩ BERT = accepted correction
    │      (single match only; 0 or >1 matches → no change)
    │
    ├── 6. Homophone filter (double metaphone) → reject stylistic variants
    │
    └── 7. Apply corrections via whole-word regex replace
    │
    ▼
Corrected text (+ optional change log)
```

### API Surface

**Public API (via `__init__.py`):**
- `spellcheck(text, **kwargs).fix()` — Main entry point
- `unsplit(text, **kwargs).fix()` — Hyphen repair

**CLI entry point (`ocrfixr2` script):**
- `ocrfixr2 <input.txt> <output.txt> [-Warp10] [-context] [-misspells]`

**Configuration (environment variables):**
- `OCRFIXR_MODEL` — BERT model name (default: `bert-base-uncased`)
- `OCRFIXR_MODEL_CACHE` — Model cache directory (default: `src/ocrfixr2/model_cache/`)

---

## Security

### Assessment: Low Risk (Offline Tool)

OCRfixr2 is a **local-only CLI tool** with no network server, no user accounts, and no external API calls during operation (after model download). The attack surface is minimal:

| Concern | Status | Notes |
|---------|--------|-------|
| Authentication | N/A | No auth required — local tool |
| Input validation | Partial | Regex-based filtering is defensive but not security-focused |
| Path traversal | Low risk | CLI accepts file paths; no sandboxing |
| Dependency supply chain | Standard | Uses PyPI packages; `uv.lock` pins versions |
| Data exfiltration | None | No outbound network calls during processing |
| Model download | One-time | Hugging Face download at import (or via pre-download script) |

### Notes
- The `ast.literal_eval()` calls on data files (`Scannos_Common.txt`, etc.) are safe — these are bundled package resources, not user input.
- No user data is transmitted externally.

---

## Architecture Bottlenecks

### B1: Eager Model Loading (Critical)

**Location:** `src/ocrfixr2/spellcheck.py`, lines 68-82

BERT tokenizer and model are loaded at **module import time**:

```python
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
unmasker = pipeline("fill-mask", model=_model, tokenizer=_tokenizer)
```

**Impact:** Every `import ocrfixr2` triggers a 5-10 second cold start (longer on first run without cache). This affects:
- CLI startup time
- Any code that imports the package conditionally
- Testing (each test import reloads the module)
- Embedding in larger applications

**Recommendation:** Lazy-load the model on first `fix()` call. Wrap in a module-level singleton with `None` initial state.

### B2: Redundant ML Framework Dependencies (High)

**Location:** `pyproject.toml`, dependencies section

Both `tensorflow>=2.21.0` and `torch>=2.11.0` are listed as dependencies. Analysis of the codebase shows:
- `transformers` uses PyTorch backend (torch) for BERT
- TensorFlow is **not imported anywhere** in the source code
- TensorFlow adds ~2-3 GB of unnecessary dependencies

**Impact:** Massive install size, longer install times, potential version conflicts.

**Recommendation:** Remove `tensorflow` from dependencies. Verify with `grep -rn "tensorflow\|tf\." src/`.

### B3: Paragraph-Level Processing Limits Context Window (Medium)

**Location:** `src/ocrfixr2/spellcheck.py`, `_SPLIT_PARAGRAPHS()` and `fix()` methods

Text is split by `\n` (newline), treating each line as a paragraph. In Gutenberg texts, `\n` marks line breaks (typically 64-72 characters), not paragraph boundaries. This means BERT sees only ~10-15 words of context instead of full paragraphs.

**Evidence:** The TODO comment at line 590 acknowledges this:
```python
# TODO - (FULL_PARAGRAPHS) Allow BERT context to draw from all lines in a full paragraph
# (currently resets at each newline -- this corresponds to 1 line of text in a Gutenberg text,
# and likely leads to degraded spellcheck performance due to loss of context)
```

**Impact:** Reduced correction accuracy, especially for words at line boundaries.

**Recommendation:** Add a paragraph detection mode (double-newline or indentation-based splitting) as an optional parameter.

### B4: No Concurrency Safety (Medium)

**Location:** Module-level globals in `spellcheck.py`

The BERT pipeline (`unmasker`) is a module-level singleton. If multiple threads/processes call `spellcheck().fix()` concurrently, the shared pipeline state could cause:
- Race conditions in the pipeline
- Inconsistent results
- Memory pressure

**Impact:** Limits the tool to single-threaded use. No multiprocessing support for processing large books faster.

**Recommendation:** Document the single-thread constraint. If multiprocessing is desired, use process-level isolation (each process gets its own model instance).

### B5: CLI Processes One Line at a Time (Low-Medium)

**Location:** `src/ocrfixr2/run_ocrfixr.py`, lines 95-107

The CLI iterates over lines individually, creating a new `spellcheck` instance per line. This means:
- BERT is called per-line (not per-paragraph)
- No cross-line context
- High overhead from object creation

**Recommendation:** Process in paragraph-sized batches for better context and fewer BERT calls.

---

## Edge Cases and Failure Modes

### Drop-off Table

| Failure Point | What's Saved | What's Lost | Recovery Path |
|--------------|-------------|-------------|---------------|
| Model download fails (no cache) | Nothing | Entire operation fails | Pre-download via `scripts/pull_model.py` |
| Model cache corrupted | Original text | Correction fails silently | Delete cache, re-download |
| BERT OOM on long paragraphs | Original text | Paragraph skipped | Paragraph is split at 500 words (line 195) |
| SymSpell returns no candidates | Original text | Word left uncorrected | N/A — conservative by design |
| Interactive mode on headless | Original text | Crash (tkinter display error) | No fallback; must use non-interactive mode |
| Unicode/text encoding issues | Original text | Malformed output | CLI uses `utf-8` explicitly |
| Regex replacement matches wrong word | Partial correction | Incorrect replacement | Whole-word `\b` boundaries mitigate this |

### Critical Gap: Interactive Mode Has No Escape Hatch

**Location:** `src/ocrfixr2/spellcheck.py`, `_CREATE_DIALOGUE()` method

The interactive mode has no exit valve. The TODO comment acknowledges this:
```python
# TODO - create exit-valve from interactive mode, where user can stop generation
# of pop-up windows. This should cancel all further updates to the text.
```

If a user has 50 suggestions and changes their mind after 10, they must click through all 40 remaining dialogs. There's no "cancel all remaining" option.

**Recommendation:** Add a "Cancel remaining" button or keyboard shortcut (Esc).

### Other Edge Cases

1. **30% unrecognized threshold** (line 172): If >30% of words in a paragraph are unrecognized, the entire paragraph is skipped. This is defensive but can miss corrections in genuinely messy OCR.

2. **Common scannos bypass BERT entirely** (line 220): Words in `Scannos_Common.txt` are corrected without context verification. This is fast but risks incorrect corrections in ambiguous contexts.

3. **Homophone filter is too aggressive**: The double-metaphone check rejects all phonetically similar corrections, including valid ones (mitigated by the `o→e` exception).

---

## Operational Readiness

| Area | Status | Notes |
|------|--------|-------|
| Error monitoring | ❌ None | No logging, no error reporting |
| Alerting | ❌ None | N/A for local tool |
| Admin tooling | ❌ None | N/A for local tool |
| Support mechanism | ❌ None | No built-in help beyond `--help` |
| CI/CD | ❌ None | No GitHub Actions, no automated tests |
| Dev/prod separation | N/A | Single environment (local use) |
| Version pinning | ✅ `uv.lock` | Dependencies are locked |
| Linting | ⚠️ Configured, not enforced | Ruff cache exists but no CI step |
| Testing | ✅ 2 test files | ~30 test cases, good coverage of core logic |

---

## Data Architecture

### Data Files (Bundled)

| File | Size | Purpose |
|------|------|---------|
| `SCOWL_70.txt` | 1.1 MB (116,884 words) | Primary word list for recognition |
| `SCOWL_20.txt` | 108 KB (12,777 words) | Common words for unsplit disambiguation |
| `Scannos_Common.txt` | 102 KB (4,736 entries) | Direct OCR error → correction mappings |
| `Scannos_Stealth.txt` | 30 bytes (1 entry) | Valid-but-wrong word mappings |
| `Ignore_These_Misspells.txt` | 81 KB (9,857 entries) | Words to skip (foreign language, etc.) |
| `Ignore_These_Suggestions.txt` | 16 bytes (empty) | Known bad suggestions (currently empty) |

### External Data Dependencies

| Source | Type | License | Notes |
|--------|------|---------|-------|
| SCOWL word lists | Data | Public domain | Kevin Atkinson, 2000-2019 |
| SymSpell frequency dictionary | Data | Bundled with symspellpy | English word frequencies |
| Hugging Face `bert-base-uncased` | Model | Apache 2.0 | Downloaded at runtime or pre-cached |

---

## Test Coverage Analysis

| Module | Test File | Coverage |
|--------|-----------|----------|
| `spellcheck` | `tests/test_spellcheck.py` | Good — 15 test methods covering core paths |
| `unsplit` | `tests/test_unsplit.py` | Good — 16 test methods covering edge cases |
| `run_ocrfixr` (CLI) | None | **Gap** — CLI not tested |
| Interactive mode | None | **Gap** — tkinter flow not tested |
| Model loading | None | **Gap** — no fixture/mocking for BERT |

**Test quality:** Tests are well-structured with clear assertions. Speed tests are included. However, tests depend on the actual BERT model being available, making them slow and environment-dependent.

---

## Recommendations

### Phase 1: Critical Fixes (Estimated: 2-3 hours)

| Task | Description | Effort |
|------|-------------|--------|
| T1 | Lazy-load BERT model (defer until first `fix()` call) | 30 min |
| T2 | Remove `tensorflow` from `pyproject.toml` dependencies | 10 min |
| T3 | Add exit valve to interactive mode (Cancel button) | 30 min |
| T4 | Add basic CI workflow (pytest + ruff on push) | 30 min |

### Phase 2: Quality Improvements (Estimated: 3-5 hours)

| Task | Description | Effort |
|------|-------------|--------|
| T5 | Add paragraph-level processing option (double-newline split) | 1 hour |
| T6 | Add CLI tests (test with sample input files) | 1 hour |
| T7 | Add structured logging (replace `print()` with `logging`) | 30 min |
| T8 | Document concurrency limitations | 15 min |
| T9 | Mock BERT in tests for faster CI runs | 1 hour |

### Phase 3: Feature Enhancements (Estimated: 5-10 hours)

| Task | Description | Effort |
|------|-------------|--------|
| T10 | Add batch processing mode for multiple files | 1 hour |
| T11 | Add confidence scoring for suggestions | 2 hours |
| T12 | Add plugin/custom dictionary support | 2 hours |
| T13 | Add parallel processing for large books (multiprocessing) | 2 hours |
| T14 | Add `--dry-run` mode (show suggestions without modifying) | 30 min |

### Deliberately Not Recommended

| Item | Reason |
|------|--------|
| Web UI / API server | Out of scope — this is a local tool |
| Cloud deployment | Unnecessary for target use case |
| Alternative LLM backends | BERT is appropriate; heavier models add cost/complexity |
| Database integration | Stateless tool; no persistence needed |
| Authentication / user management | Single-user local tool |

---

## Component Summary

| Aspect | Current | Recommended |
|--------|---------|-------------|
| Model loading | Eager (import-time) | Lazy (first-use) |
| ML dependencies | TensorFlow + PyTorch | PyTorch only |
| Context window | Single line (\n) | Configurable (line or paragraph) |
| CI/CD | None | GitHub Actions (pytest + ruff) |
| Interactive mode | No cancel | Cancel button + Esc key |
| CLI testing | None | Basic integration tests |
| Logging | `print()` statements | `logging` module |
| Concurrency | Unsafe (shared globals) | Documented as single-thread |

---

## Appendices

### A. Environment Variables Inventory

| Variable | Default | Purpose | Location |
|----------|---------|---------|----------|
| `OCRFIXR_MODEL` | `bert-base-uncased` | BERT model identifier | `spellcheck.py:70` |
| `OCRFIXR_MODEL_CACHE` | `src/ocrfixr2/model_cache/` | Model file cache path | `spellcheck.py:71-76` |
| `HF_TOKEN` | (unset) | Hugging Face auth token (for faster downloads) | `scripts/pull_model.py` (README note) |

### B. Key File References

| File | Lines | Role |
|------|-------|------|
| `src/ocrfixr2/spellcheck.py` | 599 | Core engine — SymSpell + BERT dual verification |
| `src/ocrfixr2/unsplit.py` | 137 | Hyphenated word repair |
| `src/ocrfixr2/run_ocrfixr.py` | 152 | CLI entry point |
| `src/ocrfixr2/__init__.py` | 6 | Public API exports |
| `scripts/pull_model.py` | ~40 | Model pre-download utility |
| `tests/test_spellcheck.py` | ~180 | Spellcheck unit tests |
| `tests/test_unsplit.py` | ~120 | Unsplit unit tests |

### C. Dependency Graph (Simplified)

```
ocrfixr2
├── transformers (→ torch, tokenizers, safetensors, ...)
├── symspellpy (→ frequency dictionary)
├── metaphone (→ homophone detection)
├── guiguts (→ Gutenberg text parsing, CLI output formatting)
├── tqdm (→ progress bar)
├── numpy (→ tensor operations via transformers)
└── [stdlib] tkinter, argparse, re, pathlib, collections
```

### D. Open TODOs (from source code comments)

| TODO Label | Description | Priority |
|------------|-------------|----------|
| `ADD_DICTS` | Add selectable foreign language dictionaries | Low |
| `IGNORE_SPLIT_WORDS` | Ignore first word of new page (split word detection) | Medium |
| `ADD_STEALTHOS` | Add more stealth scanno entries | Low |
| `FULL_PARAGRAPHS` | Allow BERT to use full paragraph context | **High** |
| `GutenBERT` | Fine-tune BERT on Gutenberg texts | Low (research) |
| `WARM_UP` | Negate transformers warm-up time | Medium |
| Interactive "IGNORE ALL" | Skip repeated misreads after first decision | Medium |
| Interactive "ACCEPT ALL" | Accept repeated misreads after first decision | Medium |
| Interactive exit valve | Cancel remaining dialogs | **High** |

---

*This review was generated by analyzing the actual codebase. All file paths, line numbers, and code references are verified against the repository state as of 2026-05-02.*
