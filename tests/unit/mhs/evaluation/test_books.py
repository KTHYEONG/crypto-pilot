"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.books as books


def test_books_module_present() -> None:
    assert books.__name__ == "src.mhs.evaluation.books"
    assert callable(books._book_structure_trace)
