"""Pytest bootstrap: put the repository root on ``sys.path``.

The tests live in ``tests/`` without an ``__init__.py`` and import the
``spatial`` package from the repository root, so the root must be importable
regardless of the directory pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
