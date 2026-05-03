#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import re
from tqdm import tqdm
from collections import Counter
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


def main():

    parser = argparse.ArgumentParser(
        prog="ocrfixr2",
        description="Provides context-based spellcheck suggestions for input text.",
    )

    parser.add_argument("text", help="path to text you want to spellcheck")
    parser.add_argument("outfile", help="path to output file")
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

    args = parser.parse_args()

    ### Read in file ============================================================
    # read in full file to check if the text has split words (which will cause false misreads to show up)
    # -- do this first to throw error early if file is invalid
    logger.info("Loading text....")
    Full_Book = open(args.text, "r", encoding="utf-8").read()

    if len(re.findall("[A-z]-\n", Full_Book)) > 30:
        logger.warning(
            "This file appears to have words split across lines, which can cause issues with the spellchecker"
        )
        logger.info("Merging split words back together...")
        from ocrfixr2 import unsplit

        fixed_text = unsplit(Full_Book).fix()
        data = fixed_text.split("\n")
    else:
        data = Full_Book.split("\n")

    from ocrfixr2 import spellcheck

    # Add line numbers
    q = []
    for number, line in enumerate(data):
        q.append("%d:  %s" % (number + 1, line))

    # Define misspells counter function
    # Used by both -Warp10 and -misspells flags
    def ct_misspells(text, min_len):
        M = spellcheck(text)._LIST_MISREADS()
        M = [word for word in M if len(word) > min_len]
        counts = dict(Counter(M))
        counts = dict(sorted(counts.items(), key=lambda item: -item[1]))
        return counts

    ### Misspells Option ============================================================
    # Have OCRfixr just output a list of all the words it checked (ranked by frequency), rather than spellchecking
    # This is intended as a diagnostic measure to see if OCRfixr is missing a large number of suggestions for valid (fixable) words

    if args.misspells:
        counts = ct_misspells(Full_Book, 0)
        with open(args.outfile, "w", encoding="utf-8") as f:
            for key, value in counts.items():
                f.write("%s:%s\n" % (key, value))
        logger.info("File has been written to " + args.outfile)

        # for this path, don't continue any further
        return

    ### WARP10 Option ============================================================
    # Have OCRfixr ignore any word (>3 characters long) that pops up 10+ times
    # This allows for unrecognized words that are likely correct to be left alone, since they show up consistently in the text
    # OCRfixr runs fewer check cycles = faster execution

    if args.Warp10:
        logger.info("Engaging Warp10!")
        counts = ct_misspells(Full_Book, 3)
        over_ten = {key: value for (key, value) in counts.items() if value >= 10}

        logger.info(
            "To speed things up, OCRfixr will ignore the following unrecognized words that popped up 10 or more times in the text:"
        )
        if len(over_ten) == 0:
            logger.info("NO WORDS IGNORED!")
        else:
            for k, v in over_ten.items():
                logger.info(f"{k} --> {v}")
        ignored_words = list(over_ten.keys())

    else:
        ignored_words = []

    if args.context:
        context_fl = "T"
    else:
        context_fl = "F"

    ### Run spellcheck on each line ==================================================
    logger.info("Running spellcheck....")

    suggestions = []
    for i in tqdm(q):
        fixes = spellcheck(
            i,
            changes_by_paragraph="T",
            return_context=context_fl,
            ignore_words=ignored_words,
        ).fix()
        if fixes == "NOTE: No changes made to text":
            pass
        else:
            for x in fixes.split("\n"):
                suggestions.append("".join((" ".join(re.findall("^[0-9]+:", i)), x)))

    ### Output file =================================================================
    file = open(args.outfile, "w", encoding="utf-8")
    for items in suggestions:
        file.writelines(items + "\n")
    file.close()

    logger.info("File has been written to " + args.outfile)


if __name__ == "__main__":
    main()
