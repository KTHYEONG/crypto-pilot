"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.concurrency as concurrency


def test_concurrency_module_present() -> None:
    assert concurrency.__name__ == "src.mhs.evaluation.concurrency"
    assert callable(concurrency._run_books_concurrent)


def test_concurrency_post_book_base_panel_injection() -> None:
    from src.mhs.evaluation import concurrency
    import inspect

    sig = inspect.signature(concurrency._run_post_book_concurrently)
    assert "base_panel" in sig.parameters
