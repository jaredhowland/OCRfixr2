#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OCRfixr2 CLI entry point with batch, parallel, and dry-run support."""

import argparse
import glob
import logging
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

logger = logging.getLogger("ocrfixr2")

# Configure logger with a default handler so CLI output is visible
# Users can override by configuring the "ocrfixr2" logger externally
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _process_single_file(args_tuple):
    """Worker function for processing a single file.

    Returns (input_path, output_path, suggestions, error) tuple.
    """
    input_path, output_path, warp10, context_fl, ignored_words, \
        dry_run, confidence_threshold, custom_dict_words = args_tuple

    try:
        from ocrfixr2 import spellcheck, unsplit

        # Read input file
        with open(input_path, "r", encoding="utf-8") as f:
            full_book = f.read()

        # Always run unsplitter first to merge words split across lines
        fixed_text = unsplit(full_book).fix()

        # Add line numbers
        q = []
        for number, line in enumerate(fixed_text.split("\n")):
            q.append("%d:  %s" % (number + 1, line))

        # Warp10: ignore words appearing 10+ times
        if warp10:
            M = spellcheck(full_book)._LIST_MISREADS()
            M = [word for word in M if len(word) > 3]
            counts = dict(Counter(M))
            counts = dict(sorted(counts.items(), key=lambda item: -item[1]))
            over_ten = {k: v for (k, v) in counts.items() if v >= 10}
            if over_ten:
                logger.info(
                    f"  Ignoring {len(over_ten)} frequent unrecognized words in "
                    f"{os.path.basename(input_path)}"
                )
                ignored_words = ignored_words + list(over_ten.keys())

        # Run spellcheck on each line
        suggestions = []
        for line_entry in q:
            fixes = spellcheck(
                line_entry,
                changes_by_paragraph="T",
                return_context=context_fl,
                ignore_words=ignored_words,
                confidence_threshold=confidence_threshold,
                custom_dict=custom_dict_words,
            ).fix()
            if fixes == "NOTE: No changes made to text":
                continue
            for x in fixes.split("\n"):
                suggestions.append(
                    "".join((" ".join(re.findall("^[0-9]+:", line_entry)), x))
                )

        # Write output file (unless dry-run)
        if not dry_run:
            with open(output_path, "w", encoding="utf-8") as f:
                for item in suggestions:
                    f.write(item + "\n")

        return (input_path, output_path, suggestions, None)

    except Exception as e:
        logger.error(f"  Error processing {input_path}: {e}")
        return (input_path, output_path, [], str(e))


def _resolve_input_files(text_arg):
    """Resolve input file(s) from argument.

    Supports:
    - Single file path
    - Glob patterns (*.txt, dir/*.txt)
    - File list (one path per line)
    - Directory (recursively finds all .txt files)
    """
    path = Path(text_arg)

    # If it's a file listing paths, expand it
    if path.is_file() and path.suffix == ".lst":
        with open(path, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
        return files

    # Directory - find all .txt files (check before glob)
    if path.is_dir():
        return [str(p) for p in path.rglob("*.txt")]

    # Try glob expansion (e.g., *.txt, dir/*.txt)
    expanded = glob.glob(str(path))
    if expanded:
        # Filter to only files (not directories)
        return [f for f in expanded if Path(f).is_file()]

    # Single file
    if path.is_file():
        return [str(path)]

    return [str(path)]


def main():
    parser = argparse.ArgumentParser(
        prog="ocrfixr2",
        description="Provides context-based spellcheck suggestions for input text.",
    )

    parser.add_argument(
        "text",
        help="path to text file(s) to spellcheck. Supports glob patterns, "
        "directories, or .lst files containing paths.",
    )
    parser.add_argument(
        "outfile",
        nargs="?",
        default=None,
        help="path to output file. If multiple inputs, use pattern like "
        "'output_%%(basename)s.txt'. For dry-run, output goes to stdout.",
    )
    parser.add_argument(
        "-Warp10",
        action="store_const",
        const=True,
        default=False,
        dest="Warp10",
        help="option to ignore the most common misspells, which are likely correct words.",
    )
    parser.add_argument(
        "-context",
        action="store_const",
        const=True,
        default=False,
        dest="context",
        help="option to add local context of suggested change.",
    )
    parser.add_argument(
        "-misspells",
        action="store_const",
        const=True,
        default=False,
        dest="misspells",
        help="option to return all of the words OCRfixr didn't recognize.",
    )
    # Interactive mode (tkinter GUI)
    parser.add_argument(
        "-i",
        "-interactive",
        "--interactive",
        action="store_const",
        const=True,
        default=False,
        dest="interactive",
        help="open tkinter dialog to accept/reject each suggestion. "
        "Requires a display (will not work headless).",
    )
    # Save/load progress for interactive mode
    parser.add_argument(
        "-resume",
        "--resume",
        action="store_const",
        const=True,
        default=False,
        dest="resume",
        help="resume interactive spellcheck from saved progress. "
        "Loads decisions from <input>.ocrfixr_progress.json.",
    )
    # T14: Dry-run mode
    parser.add_argument(
        "-dry-run",
        "--dry-run",
        action="store_const",
        const=True,
        default=False,
        dest="dry_run",
        help="show suggestions without writing output files.",
    )
    # T11: Confidence threshold
    parser.add_argument(
        "-confidence",
        "--confidence",
        type=float,
        default=0.0,
        dest="confidence_threshold",
        help="minimum BERT confidence score to accept a suggestion (0.0-1.0). "
        "Default: 0.0 (accept all).",
    )
    # T12: Custom dictionary
    parser.add_argument(
        "-dict",
        "--dict",
        type=str,
        default=None,
        dest="custom_dict",
        help="path to custom word list file (one word per line). "
        "Words in this file are treated as valid.",
    )
    # T13: Parallel processing
    parser.add_argument(
        "-parallel",
        "--parallel",
        type=int,
        default=1,
        dest="parallel",
        help="number of parallel workers for processing. "
        "Default: 1 (sequential).",
    )

    args = parser.parse_args()

    # T12: Load custom dictionary if provided
    custom_dict_words = None
    if args.custom_dict:
        dict_path = Path(args.custom_dict)
        if not dict_path.is_file():
            logger.error(f"Custom dictionary file not found: {args.custom_dict}")
            sys.exit(1)
        with open(dict_path, "r", encoding="utf-8") as f:
            custom_dict_words = [line.strip().lower() for line in f if line.strip()]
        logger.info(f"Loaded {len(custom_dict_words)} words from custom dictionary")

    # T10: Resolve input files
    input_files = _resolve_input_files(args.text)
    if not input_files:
        logger.error(f"No input files found matching: {args.text}")
        sys.exit(1)

    logger.info(f"Processing {len(input_files)} file(s)...")

    # Interactive mode: only works with single file
    if args.interactive:
        if len(input_files) > 1:
            logger.error("Interactive mode requires a single input file.")
            sys.exit(1)

        from ocrfixr2 import spellcheck, unsplit

        input_path = input_files[0]
        with open(input_path, "r", encoding="utf-8") as f:
            full_book = f.read()

        # Always run unsplitter first to merge words split across lines
        full_book = unsplit(full_book).fix()

        # Progress file for save/resume
        progress_file = str(Path(input_path).with_suffix(".ocrfixr_progress.json"))

        logger.info(f"Opening interactive dialog for {os.path.basename(input_path)}...")
        result = spellcheck(
            full_book,
            interactive="T",
            common_scannos="T",
            custom_dict=custom_dict_words,
            confidence_threshold=args.confidence_threshold,
            progress_file=progress_file,
        ).fix()

        # Write corrected text
        if args.outfile:
            output_path = args.outfile
        else:
            output_path = str(Path(input_path).with_suffix("")) + ".fixed"

        if not args.dry_run:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            logger.info(f"Corrected text written to {output_path}")
        else:
            logger.info("Dry-run: corrected text not written")
        return

    # Build output paths
    def _make_output_path(input_path):
        """Generate output path from input path."""
        if args.outfile:
            # Use pattern with basename substitution
            basename = Path(input_path).stem
            return args.outfile.replace("%%(basename)s", basename)
        else:
            # Default: same name with .suggestions suffix
            return str(Path(input_path).with_suffix("")) + ".suggestions"

    # Handle -misspells flag (legacy behavior: single file, list all unrecognized words)
    if args.misspells:
        from ocrfixr2 import spellcheck

        for input_path in input_files:
            with open(input_path, "r", encoding="utf-8") as f:
                full_book = f.read()
            M = spellcheck(full_book)._LIST_MISREADS()
            counts = dict(Counter(M))
            counts = dict(sorted(counts.items(), key=lambda item: -item[1]))

            output_path = _make_output_path(input_path) if args.outfile else None
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    for key, value in counts.items():
                        f.write(f"{key}:{value}\n")
                logger.info(f"File has been written to {output_path}")
            else:
                for key, value in counts.items():
                    print(f"{key}:{value}")
        return

    # Build output paths
    def _make_output_path(input_path):
        """Generate output path from input path."""
        if args.outfile:
            # Use pattern with basename substitution
            basename = Path(input_path).stem
            return args.outfile.replace("%%(basename)s", basename)
        else:
            # Default: same name with .suggestions suffix
            return str(Path(input_path).with_suffix("")) + ".suggestions"

    # Prepare arguments for each file
    file_args = []
    for input_path in input_files:
        output_path = _make_output_path(input_path)
        file_args.append((
            input_path,
            output_path,
            args.Warp10,
            "T" if args.context else "F",
            [],  # ignored_words (populated per-file for Warp10)
            args.dry_run,
            args.confidence_threshold,
            custom_dict_words,
        ))

    # T13: Process files (sequentially or in parallel)
    if args.parallel > 1 and len(file_args) > 1:
        logger.info(f"Using {args.parallel} parallel workers...")
        with mp.Pool(processes=min(args.parallel, len(file_args))) as pool:
            results = pool.map(_process_single_file, file_args)
    else:
        # Sequential processing with progress bar
        results = []
        for args_tuple in tqdm(file_args, desc="Processing files"):
            result = _process_single_file(args_tuple)
            results.append(result)

    # Report results
    total_suggestions = 0
    errors = 0
    for input_path, output_path, suggestions, error in results:
        if error:
            logger.error(f"  {input_path}: {error}")
            errors += 1
        else:
            total_suggestions += len(suggestions)
            status = "dry-run" if args.dry_run else f"written to {output_path}"
            logger.info(
                f"  {os.path.basename(input_path)}: "
                f"{len(suggestions)} suggestion(s) {status}"
            )

    logger.info(
        f"Complete: {total_suggestions} total suggestion(s), "
        f"{errors} error(s)"
    )


if __name__ == "__main__":
    main()
