---
description: >
  CLI entry point — batch processing, parallel workers, dry-run, custom dictionaries, interactive mode.
tags:
  - CLI
  - batch
  - parallel
  - entry-point
---

# run_ocrfixr.py — CLI Entry Point

## Usage
```bash
ocrfixr2 <input> [output] [options]
```

## Input Resolution
Supports: single file, glob patterns, `.lst` files (one path per line), directories (recursive `.txt` search)

## Options
| Flag | Purpose |
|------|---------|
| `-Warp10` | Ignore words appearing 10+ times |
| `-context` | Include local context in suggestions |
| `-misspells` | List unrecognized words only |
| `-i` / `--interactive` | tkinter GUI (single file only) |
| `--resume` | Resume interactive from saved progress |
| `--dry-run` | Show suggestions without writing files |
| `--confidence <0-1>` | Min BERT confidence threshold |
| `--dict <path>` | Custom word list file |
| `--parallel <N>` | Parallel worker count |

## Processing Flow
1. Unsplit → add line numbers → Warp10 filter (if enabled) → spellcheck per line → collect suggestions → write output
2. Parallel mode uses `multiprocessing.Pool` with `_process_single_file` worker function
3. Interactive mode processes full book as single unit with progress file support

## Output Format
- Default: `<input>.suggestions` (line number + position + suggestion per line)
- Interactive: `<input>.fixed` (corrected text)
- Custom: `output_%(basename)s.txt` pattern for batch
