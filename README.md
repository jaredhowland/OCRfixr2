<img src=https://img.shields.io/badge/Python-3.6%2B-blue alt="python versions supported">

# OCRfixr2

## OVERVIEW 
This project aims to help automate the challenging work of manually correcting OCR output from Distributed Proofreaders' book digitization projects


## Correcting OCR Misreads
OCRs can sometimes mistake similar-looking characters when scanning a book. For example, "l" and "1" are easily confused, potentially causing the OCR to misread the word "learn" as "1earn".

As written in book: 
> _"The birds flevv south"_

Corrected text:
> _"The birds flew south"_

### How OCRfixr2 Works:
OCRfixr2 fixes misreads by checking __1) possible spell corrections__ against the __2) local context__ of the word. For example, here's how OCRfixr2 would evaluate the following OCR mistake:

As written in book: 
> _"Days there were when small trade came to the __stoie__. Then the young clerk read._"

| Method | Plausible Replacements |
| --------------- | --------------- | 
| Spellcheck (symspellpy) | stone, __store__, stoke, stove, stowe, stole, soie |
| Context (BERT) | market, shop, town, city, __store__, table, village, door, light, markets, surface, place, window, docks, area |

Since there is match for both a plausible spellcheck replacement and that word reasonably matches the context of the sentence, OCRfixr2 updates the word. 

Corrected text:
> _"Days there were when small trade came to the __store__. Then the young clerk read._"

For very common scanning errors where it is clear what the word should have been (ex: 'onlv' --> 'only'), OCRfixr2 skips the context check and relies solely on a static mapping of common corrections. This helps to maximize the number of successful edits \& decrease compute time. (You can disable this by setting common_scannos to "F").

### Using OCRfixr2

The package can be installed using [uv (preferred) or pip](https://pypi.org/project/OCRfixr2/). 

```bash
uv add ocrfixr2
```

or

```bash
pip install ocrfixr2
```

By default, OCRfixr2 only returns the original string, with all changes incorporated:
```python
>>> from ocrfixr2 import spellcheck

>>> text = "The birds flevv south"
>>> spellcheck(text).fix()
'The birds flew south'
```

Use __return_fixes__ to also include all corrections made to the text, with associated counts for each:
```python
>>> spellcheck(text, return_fixes = "T").fix()
['The birds flew south', {("flevv","flew"):1}]
```

_(Note: OCRfixr2 resets its BERT context window at the start of each new paragraph, so splitting by paragraph may be a useful debug feature)_


### Interactive Mode
OCRfixr2 also has an option for the user to interactively accept/reject suggested changes to the text:

```python
>>> text = "The birds flevv down\n south, but wefe quickly apprehended\n by border patrol agents"
>>> spellcheck(text, interactive = "T").fix()
```

<img width="723" alt="Suggestion 1" src="https://user-images.githubusercontent.com/67446041/107133270-7918c300-68b4-11eb-9de5-5b6282510c16.png">

Each suggestion provides the local context around the garbled text, so that the user can determine if the suggestion fits.

<img width="723" alt="Suggestion 2" src="https://user-images.githubusercontent.com/67446041/115068768-af7c4b00-9ec0-11eb-9c7a-65b518718ec4.png">

```python
>>> ### User accepts change to "flevv", but rejects change to "wefe" in GUI
'The birds flew down\n south, but wefe quickly apprehended\n by border patrol agents'
```

This returns the text with all accepted changes reflected. All rejected suggestions are left as-is in the text.

### Command-Line 
OCRfixr2 is also callable via command-line (intended for Guiguts use):

```python
>>> ocrfixr2 input_text.txt output_filename.txt
```

or

```bash
uv run ocrfixr2 input_text.txt output_filename.txt
```

The output file will list the line number and position of all suggested changes.


### Avoiding "Damn You, Autocorrect!"
By design, OCRfixr2 is change-averse:
- If spellcheck/context do not line up, no update is made.
- Likewise, if there is >1 word that lines up for spellcheck/context, no update is made.
- Only the top 15 context suggestions are considered, to limit low-probability matches.
- If the suggestion is a homophone of the original word, it is ignored  (original: coupla --> suggestion: couple). These are assumed to be 'stylistic' or phonetic misspellings
- Proper nouns (anything starting with a capital letter) are not evaluated for spelling.

Word context is drawn from all sentences in the current paragraph (designated by a '\n'), to maximize available information, while also not bogging down the BERT model. 



## Pre-downloading the model (optional)

To avoid downloading the transformer model on every run, pre-download it into the project-local cache. A helper script is provided to populate the cache directory used by OCRfixr2.

Usage (run once on the machine where you'll run the CLI):

```bash
python scripts/pull_model.py --model bert-base-uncased --cache src/ocrfixr2/model_cache
```

The default project-local cache directory is `src/ocrfixr2/model_cache`. You can also set the OCRFIXR_MODEL_CACHE environment variable to point to a different directory.

For fully offline operation after pre-download, set environment variables or use `local_files_only` options in transformers (see transformers docs). After downloading, subsequent runs of `uv run ocrfixr2 ...` will reuse the cached files.


## Credits

- __symspellpy__ powers spellcheck suggestions
- __transformers__ does the heavy lifting for BERT context modelling
- __DataMunging__ provided a very useful list of common scanning errors 
- __SCOWL__ word list is Copyright 2000-2019 by Kevin Atkinson.
- This project was created to help __Distributed Proofreaders__. Support them here: <https://www.pgdp.net/c/>