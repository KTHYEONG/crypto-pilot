from __future__ import annotations

from unittest.mock import patch

from src.application.futures.runner.active_pipeline import _get_rss_mb


def test_get_rss_mb_returns_positive_on_linux() -> None:
    result = _get_rss_mb()
    assert isinstance(result, float)
    assert result > 0.0


def test_get_rss_mb_returns_neg_one_on_missing_proc() -> None:
    with patch("builtins.open", side_effect=FileNotFoundError):
        result = _get_rss_mb()
    assert result == -1.0


def test_log_mem_runs_without_exception() -> None:
    """_log_mem이 예외 없이 실행되고 None을 반환하는지 검증.

    Note:
        setup_logger의 FlushingStreamHandler가 sys.stdout을 모듈 로드 시점에
        직접 바인딩하므로, caplog/capsys로는 로그 캡처 불가.
        동작 검증은 예외 미발생(smoke test)으로 대체.
    """
    from src.application.futures.runner.active_pipeline import _log_mem

    result = _log_mem("test_stage", 100.0, extra="n_syms=5")
    assert result is None
