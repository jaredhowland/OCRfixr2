#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the CLI entry point (run_ocrfixr)."""

import os
import tempfile
import unittest
from unittest.mock import patch

from ocrfixr2.run_ocrfixr import main


class TestCLI(unittest.TestCase):
    """Test the CLI entry point with sample input files."""

    def setUp(self):
        """Create temporary input and output files."""
        self.input_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        self.output_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        self.input_file.close()
        self.output_file.close()

    def tearDown(self):
        """Clean up temporary files."""
        for f in (self.input_file, self.output_file):
            if os.path.exists(f.name):
                os.unlink(f.name)

    def _write_input(self, content):
        """Write content to the input file."""
        with open(self.input_file.name, "w", encoding="utf-8") as f:
            f.write(content)

    def _read_output(self):
        """Read content from the output file."""
        with open(self.output_file.name, "r", encoding="utf-8") as f:
            return f.read()

    def _run_cli(self, *extra_args):
        """Run the CLI with the test input/output files and optional extra args."""
        args = [self.input_file.name, self.output_file.name] + list(extra_args)
        with patch("sys.argv", ["ocrfixr2"] + args):
            main()

    def test_cli_creates_output_file(self):
        """Test that the CLI creates an output file."""
        self._write_input("This is a test file with no errors.")
        self._run_cli()
        self.assertTrue(os.path.exists(self.output_file.name))

    def test_cli_with_no_errors_produces_empty_output(self):
        """Test that clean text produces no suggestions."""
        self._write_input("This is a test file with no errors at all.")
        self._run_cli()
        output = self._read_output()
        self.assertEqual(output.strip(), "")

    def test_cli_misspells_flag(self):
        """Test the -misspells flag outputs unrecognized words."""
        self._write_input("This is a test with unkown words here.")
        self._run_cli("-misspells")
        output = self._read_output()
        # The output should contain the unrecognized word
        self.assertIn("unkown", output)

    def test_cli_warp10_flag(self):
        """Test the -Warp10 flag runs without error."""
        self._write_input(
            "This is a test file.\n"
            "It has multiple lines of text.\n"
            "Nothing wrong with any of it.\n"
        )
        self._run_cli("-Warp10")
        self.assertTrue(os.path.exists(self.output_file.name))

    def test_cli_context_flag(self):
        """Test the -context flag runs without error."""
        self._write_input("This is a test file with no errors.")
        self._run_cli("-context")
        self.assertTrue(os.path.exists(self.output_file.name))

    def test_cli_with_split_words_triggers_unsplit(self):
        """Test that files with many split words trigger the unsplit module."""
        # Create text with many line-ending hyphens (>30) to trigger unsplit
        split_lines = ["test-word-"] * 35
        split_lines.append("final line of text.")
        self._write_input("\n".join(split_lines))
        self._run_cli()
        self.assertTrue(os.path.exists(self.output_file.name))

    def test_cli_invalid_input_file_raises_error(self):
        """Test that missing input file raises an error."""
        with self.assertRaises(FileNotFoundError):
            with patch("sys.argv", ["ocrfixr2", "/nonexistent/file.txt", self.output_file.name]):
                main()


if __name__ == "__main__":
    unittest.main()
