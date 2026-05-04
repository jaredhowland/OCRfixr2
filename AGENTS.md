# OCRfixr2 — Codebase Instructions

## What It Is
OCRfixr2 is a Python 3.12+ package that corrects OCR misreads in digitized book text (primarily for [Distributed Proofreaders](https://www.pgdp.net/)). It uses a **dual-verification** approach: SymSpell (dictionary-based spellcheck) cross-referenced with BERT (masked language modeling for context). Also includes an **unsplit** module for fixing words broken across line boundaries.

## Architecture

### Core Modules (src/ocrfixr2/)
| File | Role |
|------|------|
| `spellcheck.py` | Core engine — SymSpell + BERT dual verification, interactive tkinter UI, lazy model loading |
| `unsplit.py` | Fixes words split across lines with hyphens (rule-based, no ML) |
| `run_ocrfixr.py` | CLI entry point — supports batch, parallel, dry-run, glob patterns, custom dicts |
| `__init__.py` | Public API: exports `spellcheck` and `unsplit` |

### Data Files (src/ocrfixr2/data/)
- `SCOWL_70.txt` — Primary word list (116,884 words)
- `SCOWL_20.txt` — Common words (12,777) for unsplit disambiguation
- `Scannos_Common.txt` — Direct OCR error→correction mappings (bypass BERT, ~4,700 entries)
- `Scannos_Stealth.txt` — Valid-but-wrong word mappings
- `Ignore_These_Misspells.txt` — Words to skip (foreign, etc.)
- `Ignore_These_Suggestions.txt` — Known bad suggestions (currently empty)

### Data Flow
```
Input text → unsplit (merge split words) → spellcheck per-paragraph
  → SCOWL check → SymSpell candidates → BERT [MASK] fill → intersection → apply corrections
```

### Key Design Decisions
- **Change-averse**: Only corrects when SymSpell AND BERT agree on exactly ONE candidate
- **Lazy model loading**: BERT loads on first `fix()` call, not at import
- **Not thread-safe**: BERT pipeline is a module-level singleton; use separate processes for parallel work
- **Paragraph context**: Splits by `\n` (configurable via `changes_by_paragraph`)

## Setup & Development

### Toolchain
- **Package manager**: `uv` (lock file: `uv.lock`)
- **Build backend**: `uv_build`
- **Python**: 3.12 (`.python-version`)
- **Linting**: `ruff`
- **Type checking**: `ty`
- **Testing**: `pytest` with mock BERT fixtures (`tests/conftest.py`)

### Commands
```bash
uv sync                               # Install deps
uv run pytest tests/ -v               # Run tests
uv run ruff check --fix src/ tests/   # Lint
uv run ty check --fix src/ tests/     # Type check
uv run ocrfixr2 input.txt output.txt  # CLI
```

### CI
GitHub Actions on push/PR to `main`/`master`: ruff lint → ty type check → pytest (Python 3.12, ubuntu-latest).

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `OCRFIXR_MODEL` | `bert-base-uncased` | BERT model name |
| `OCRFIXR_MODEL_CACHE` | `src/ocrfixr2/model_cache/` | Model cache directory |
| `HF_TOKEN` | (unset) | Hugging Face token for faster model downloads |

### Model Cache
The BERT model is pre-cached in `src/ocrfixr2/model_cache/`. To pre-download on a new machine:
```bash
uv run scripts/pull_model.py --model bert-base-uncased --cache src/ocrfixr2/model_cache
```

## Practices

### Testing
- Tests use mock BERT fixtures (`mock_bert`, `mock_bert_context_match`) from `conftest.py` — **always patch `_get_unmasker`** when testing spellcheck logic to avoid loading the real model
- Test files: `test_spellcheck.py` (core logic), `test_unsplit.py` (hyphen repair), `test_cli.py` (CLI)
- Smoke test data: `tests/input_smoke.txt` / `tests/output_smoke.txt`

### Code Style
- `ruff` for linting (check `pyproject.toml` for config)
- `ty` for type checking
- Logging via `logging.getLogger("ocrfixr2")` — no bare `print()` in production code

### Recent Work (commit history)
The repo has gone through an architecture review and implemented Phase 1-3 improvements: lazy model loading, CI pipeline, parallel processing, dry-run mode, confidence thresholds, custom dictionaries, interactive mode with save/resume progress, and full paragraph context.

## Helpful Hints

1. **BERT is heavy** — any test or tool that imports `ocrfixr2.spellcheck` will trigger model loading on first use. Use the `mock_bert` fixture to avoid this.
2. **Interactive mode requires a display** — tkinter crashes on headless servers/CI. The CLI flag `-i/--interactive` should only be used locally.
3. **Parallel processing uses separate processes** — each worker process gets its own BERT model instance (memory intensive but safe).
4. **The unsplit module runs first** — both in the CLI and recommended usage, always run unsplit before spellcheck.
5. **Common scannos bypass BERT** — words in `Scannos_Common.txt` are corrected directly without context verification (fast but can be wrong in ambiguous contexts).
6. **Paragraph splitting** — text is split by `\n`. In Gutenberg texts this is a line break (~70 chars), not a full paragraph. Use `changes_by_paragraph="T"` for paragraph-level processing.
7. **Upstream remote** — there's an `upstream` remote pointing to the original OCRfixr repo; `origin` is the fork at `jaredhowland/OCRfixr2`.
