"""Shared pytest fixtures for OCRfixr2 tests."""

import pytest


@pytest.fixture
def mock_bert():
    """Mock the BERT fill-mask pipeline to avoid loading the real model in CI.

    Returns a context manager that patches `_get_unmasker()` to return a mock
    pipeline that yields predictable token suggestions.
    """

    def _make_mock_pipeline(suggestions=None):
        """Return a mock pipeline function callable like the real BERT pipeline."""
        if suggestions is None:
            suggestions = [
                {"token_str": "the", "score": 0.45},
                {"token_str": "a", "score": 0.20},
                {"token_str": "was", "score": 0.15},
                {"token_str": "is", "score": 0.10},
                {"token_str": "and", "score": 0.05},
                {"token_str": "in", "score": 0.03},
                {"token_str": "of", "score": 0.02},
                {"token_str": "to", "score": 0.02},
                {"token_str": "it", "score": 0.01},
                {"token_str": "he", "score": 0.01},
                {"token_str": "she", "score": 0.01},
                {"token_str": "they", "score": 0.005},
                {"token_str": "we", "score": 0.003},
                {"token_str": "you", "score": 0.002},
                {"token_str": "i", "score": 0.001},
            ]

        def mock_call(text):
            return suggestions

        return mock_call

    return _make_mock_pipeline


@pytest.fixture
def mock_bert_context_match():
    """Mock BERT to return specific words that overlap with SymSpell suggestions.

    Use this fixture when you need BERT to agree with a particular correction.
    Pass a list of words that should be returned from the mock pipeline.
    """

    def _make_matcher(words):
        """Return a mock pipeline that yields the given words as token suggestions."""
        suggestions = [{"token_str": w, "score": 0.1 / (i + 1)} for i, w in enumerate(words)]

        def mock_call(text):
            return suggestions

        return mock_call

    return _make_matcher
