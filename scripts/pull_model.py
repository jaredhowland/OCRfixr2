#!/usr/bin/env python3
"""Pre-download a Hugging Face transformer model into the project-local cache.

Usage:
  python scripts/pull_model.py --model bert-base-uncased --cache src/ocrfixr2/model_cache

This saves model files into the given cache dir so runs of ocrfixr2 don't need to re-download the model.
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Pull HF model into a local cache directory")
    p.add_argument("--model", default="bert-base-uncased", help="Model name or path")
    p.add_argument(
        "--cache",
        default="src/ocrfixr2/model_cache",
        help="Cache directory to store model files (project-local by default)",
    )
    args = p.parse_args()

    cache_dir = Path(args.cache).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model {args.model} into {cache_dir}...")

    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        AutoTokenizer.from_pretrained(args.model, cache_dir=str(cache_dir))
        AutoModelForMaskedLM.from_pretrained(args.model, cache_dir=str(cache_dir))

        print("Model download complete.")
    except Exception as e:
        print("Error while downloading model:", e)
        raise


if __name__ == '__main__':
    main()
