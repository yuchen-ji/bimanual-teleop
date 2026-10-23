"""Run tests: python scripts/run_tests.py discover -s tests -v."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    unittest.main(module=None)
