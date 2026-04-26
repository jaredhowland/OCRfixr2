"""Top-level package for OCRfixr2"""

from .spellcheck import spellcheck as spellcheck
from .unsplit import unsplit as unsplit

__all__ = ["spellcheck", "unsplit"]
