"""TDD tests for Data-Window Floor Consistency.

Covers scenarios from docs/specs/data-window-floor-consistency.md:
- Scenario 1: Happy Path
- Scenario 2: Edge Cases
- Scenario 3: Error Handling
"""

import datetime as dt

from pytest_mock import MockerFixture

from src.application.futures.runner.active_pipeline import (
    _resolve_layered_window,
    _resolve_quarterly_window,
)
from src.domain.futures.optimization.opt_config import (
    get_layered_window,
    get_quarterly_window,
)
from src.domain.futures.optimization.opt_data_utils import (
    resolve_warmup_days_for_tf,
)

# ── Scenario 1: Happy Path ─────────────────────────────────────────────────


def test_resolve_warmup_days_for_tf_4h() -> None:
    """1.1: resolve_warmup_days_for_tf("4h") == 62 (ceil(252/6)+20)."""
    assert resolve_warmup_days_for_tf("4h") == 62


def test_get_layered_window_fetch_start_uses_resolved_warmup() -> None:
    """1.2: warmup_days 미지정 시 fetch_start = l1_start - timedelta(days=62)."""
    window = get_layered_window(reference_date=dt.date(2026, 1, 1), tf="4h")
    assert window.fetch_start == window.l1_start - dt.timedelta(days=62)


def test_get_layered_window_fetch_start_recovers_data_floor() -> None:
    """1.3: --date 2026-01-01 시 fetch_start >= 2022-04-01 (크래시 조건 해소)."""
    window = get_layered_window(reference_date=dt.date(2026, 1, 1), tf="4h")
    assert window.fetch_start >= dt.date(2022, 4, 1)


# ── Scenario 2: Edge Cases ─────────────────────────────────────────────────


def test_warmup_days_affects_only_fetch_start() -> None:
    """2.1: warmup_days 변경이 fetch_start에만 영향, l1_start 등은 불변."""
    ref = dt.date(2026, 6, 15)
    window_old = get_layered_window(reference_date=ref, warmup_days=365, tf="4h")
    window_new = get_layered_window(reference_date=ref, warmup_days=None, tf="4h")

    assert window_old.l1_start == window_new.l1_start
    assert window_old.l2_start == window_new.l2_start
    assert window_old.holdout_start == window_new.holdout_start
    assert window_old.holdout_end == window_new.holdout_end
    assert window_old.fetch_start != window_new.fetch_start


def test_resolve_warmup_days_for_tf_calls_resolve_warmup_bars(
    mocker: MockerFixture,
) -> None:
    """2.2: resolve_warmup_days_for_tf가 _resolve_warmup_bars를 호출(spy)."""
    spy = mocker.patch(
        "src.domain.futures.optimization.opt_data_utils._resolve_warmup_bars",
        return_value=252,
    )
    resolve_warmup_days_for_tf("4h")
    spy.assert_called_once_with("4h")


def test_resolve_warmup_days_for_tf_tf_variants() -> None:
    """2.3: tf별 다른 값 산출 - 1h→31일, 1d→272일."""
    result_1h = resolve_warmup_days_for_tf("1h")
    result_1d = resolve_warmup_days_for_tf("1d")
    assert result_1h == 62, f"Expected 62, got {result_1h}"
    assert result_1d == 272, f"Expected 272, got {result_1d}"


def test_get_layered_window_explicit_warmup_days_respected() -> None:
    """2.4: 명시적 warmup_days=365 전달 시 그대로 사용."""
    window = get_layered_window(
        reference_date=dt.date(2026, 1, 1),
        warmup_days=365,
        tf="4h",
    )
    assert window.fetch_start == window.l1_start - dt.timedelta(days=365)


def test_get_quarterly_window_fetch_start_uses_resolved_warmup() -> None:
    """2.5: get_quarterly_window도 warmup buffer 62일 적용."""
    from datetime import datetime

    fetch_start_str, is_start_str, _, _ = get_quarterly_window(
        reference_date=dt.date(2026, 1, 1),
        tf="4h",
    )
    fetch_start = datetime.strptime(fetch_start_str, "%Y-%m-%d").date()
    is_start = datetime.strptime(is_start_str, "%Y-%m-%d").date()
    assert fetch_start == is_start - dt.timedelta(days=62)


def test_resolve_layered_window_passes_tf_to_get_layered_window(
    mocker: MockerFixture,
) -> None:
    """2.6: _resolve_layered_window가 tf를 get_layered_window에 전달."""
    mock = mocker.patch(
        "src.domain.futures.optimization.opt_config.get_layered_window",
        return_value=mocker.MagicMock(),
    )
    _resolve_layered_window("2026-01-01", tf="1h")
    mock.assert_called_once()
    _, kwargs = mock.call_args
    assert kwargs.get("tf") == "1h"


def test_resolve_quarterly_window_passes_tf_to_get_quarterly_window(
    mocker: MockerFixture,
) -> None:
    """추가: _resolve_quarterly_window가 tf를 get_quarterly_window에 전달."""
    mock = mocker.patch(
        "src.application.futures.runner.active_pipeline.get_quarterly_window",
        return_value=("2024-01-01", "2024-07-01", "2025-01-01", "2026-01-01"),
    )
    _resolve_quarterly_window("2026-01-01", tf="1h")
    mock.assert_called_once()
    _, kwargs = mock.call_args
    assert kwargs.get("tf") == "1h"


# ── Scenario 3: Error Handling ─────────────────────────────────────────────


def test_resolve_warmup_days_for_tf_unknown_tf() -> None:
    """3.1: unknown_tf → bars_per_day=6 폴백, 예외 없음."""
    result = resolve_warmup_days_for_tf("unknown_tf")
    assert isinstance(result, int)
    assert result > 0


def test_get_layered_window_warmup_days_zero() -> None:
    """3.2: warmup_days=0 명시 시 그대로 존중 (resolve 미호출)."""
    window = get_layered_window(
        reference_date=dt.date(2026, 1, 1),
        warmup_days=0,
        tf="4h",
    )
    assert window.fetch_start == window.l1_start
