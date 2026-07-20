"""[WS2][pipeline-runtime-memory-optimization] L1 evidence fold IPC payload 재검증 로깅.

_fit_and_predict_single_fold_from_globals가 반환 직전 DEBUG 가드 하에
실제 pickle 크기를 [SYS] stage=l1_evidence_ipc_payload로 로깅하는지 검증한다.

opt_main_futures 로거는 setup_logger()로 propagate=False + stdout 핸들러로
구성되는 프로세스 전역 싱글턴이라(caplog는 기본적으로 root logger를 통해
캡처하므로 propagate=False 로거의 레코드를 못 잡는다), capsys로 실제
stdout 출력을 직접 검증한다.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from src.domain.futures.strategy import candidate_workflow as cw
from src.domain.futures.strategy.walk_forward import WFFold

_FOLD = WFFold(fit_start=0, fit_end=10, cal_start=8, cal_end=10, oos_start=10, oos_end=14)


@pytest.fixture(autouse=True)
def _reset_globals_and_logger() -> Any:
    """전역 상태 및 opt_main_futures 로거 핸들러/레벨을 매 테스트 전/후 초기화한다."""
    logger = logging.getLogger("opt_main_futures")
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    logger.handlers.clear()

    yield

    cw._GLOBAL_LABELED_EVENTS = None
    cw._GLOBAL_PREPARED_EVENTS = None
    cw._GLOBAL_ALIGNED = None
    cw._GLOBAL_CFG = None
    cw._GLOBAL_PURGE_BARS = None
    logger.handlers.clear()
    logger.handlers.extend(saved_handlers)
    logger.setLevel(saved_level)


def _set_minimal_globals() -> None:
    cw._GLOBAL_LABELED_EVENTS = pd.DataFrame({"entry_idx": [1, 2]})
    cw._GLOBAL_PREPARED_EVENTS = None
    cw._GLOBAL_ALIGNED = SimpleNamespace()
    cw._GLOBAL_CFG = SimpleNamespace()
    cw._GLOBAL_PURGE_BARS = 1


def _fake_fold_output() -> SimpleNamespace:
    return SimpleNamespace(
        fold_id=0,
        fit_status="trained",
        timing_profile={"total": 0.01},
        model_output=SimpleNamespace(events=pd.DataFrame()),
        oos_set=None,
    )


def test_fit_and_predict_single_fold_logs_payload_size_when_debug_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _set_minimal_globals()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    with patch(
        "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold",
        return_value=_fake_fold_output(),
    ):
        cw._fit_and_predict_single_fold_from_globals(0, _FOLD, True, True)

    out = capsys.readouterr().out
    assert "[SYS] stage=l1_evidence_ipc_payload" in out, "l1_evidence_ipc_payload log must be emitted at DEBUG level"
    assert "fold_idx=0" in out
    assert "compact=True" in out
    assert "payload_mb=" in out


def test_fit_and_predict_single_fold_skips_payload_logging_when_debug_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _set_minimal_globals()
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    with patch(
        "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold",
        return_value=_fake_fold_output(),
    ):
        cw._fit_and_predict_single_fold_from_globals(0, _FOLD, True, True)

    out = capsys.readouterr().out
    assert "l1_evidence_ipc_payload" not in out, "l1_evidence_ipc_payload log must not be emitted when DEBUG disabled"
