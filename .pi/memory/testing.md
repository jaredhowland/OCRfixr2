---
description: >
  Testing setup — mock BERT fixtures, test files, smoke test data, CI pipeline.
tags:
  - testing
  - pytest
  - CI
  - fixtures
---

# Testing & CI

## Test Structure
| File | Coverage |
|------|----------|
| `tests/test_spellcheck.py` | 15 test methods — core logic, misread detection, corrections, mashed words, speed |
| `tests/test_unsplit.py` | 16 test methods — hyphen repair, edge cases, punctuation, proper nouns |
| `tests/test_cli.py` | 7 test methods — CLI flags, output files, error handling |
| `tests/conftest.py` | `mock_bert` and `mock_bert_context_match` fixtures |
| `tests/input_smoke.txt` / `output_smoke.txt` | Smoke test data |

## Mock BERT Fixtures (`conftest.py`)
- `mock_bert()` → factory returning mock pipeline with default suggestions
- `mock_bert_context_match(words)` → factory returning specific words for overlap testing
- **Always patch `_get_unmasker`** when testing spellcheck logic to avoid loading the real model

## CI Pipeline (`.github/workflows/ci.yml`)
- Triggers: push/PR to `main`/`master`
- Steps: uv setup → deps → ruff check → ty check → pytest (Python 3.12, ubuntu-latest)

## Key Notes
- Tests use unittest style (not pytest style) despite pytest runner
- Speed tests included (acceptable thresholds defined)
- Interactive mode not tested (requires display)
