---
description: >
  Unsplit module — fixes words split across lines with hyphens using rule-based logic.
tags:
  - unsplit
  - hyphen
  - rule-based
---

# unsplit.py — Hyphen Repair

## Class: `unsplit(text, return_fixes="F")`

### Key Methods
- `_LIST_SPLIT_WORDS()` → find words with `-\n` pattern (excludes end-of-page hyphens)
- `__DECIDE_HYPHEN(text)` → determine hyphen disposition
- `_FIND_REPLACEMENTS(splits)` → build correction dict
- `fix()` → main entry point

### Hyphen Decision Logic
For each split word (W0 = unhyphenated, W1 = first half, W2 = second half):

| Condition | Action |
|-----------|--------|
| End of page (`--File` or W2 is digit) | Keep hyphen + `*` flag |
| W0 recognized + common word | Remove hyphen |
| W0 recognized + both halves recognized | Keep hyphen + `*` flag (uncertain) |
| W0 recognized + halves NOT recognized | Remove hyphen |
| W0 NOT recognized + both halves recognized | Keep hyphen |
| W0 NOT recognized + halves NOT recognized | Remove hyphen (assumed proper noun) |
| Numbers in halves | Keep hyphen |
| Second half titlecased | Keep hyphen (proper noun) |
