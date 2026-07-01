from __future__ import annotations

from pathlib import Path


class TestNoLegacyCoupling:
    """Scenario 6: src/ has no legacy import or public legacy name."""

    def test_src_has_no_legacy_import_or_public_legacy_name(self) -> None:
        src = Path("src")
        for py_file in sorted(src.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            content = py_file.read_text(encoding="utf-8")
            for lineno, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "from legacy" not in stripped, f"{py_file}:{lineno}: contains 'from legacy'"
                assert "import legacy" not in stripped, f"{py_file}:{lineno}: contains 'import legacy'"
                assert "legacy." not in stripped, f"{py_file}:{lineno}: contains 'legacy.'"
