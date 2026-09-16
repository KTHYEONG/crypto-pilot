# ruff: noqa
def test_block_bootstrap_paths_shape_and_determinism() -> None:
    import numpy as np
    import pytest

    from src.mhs.deploy_gate import block_bootstrap_paths

    # Given
    x = np.array([0.01, -0.02, 0.003, 0.015, -0.007, 0.02, -0.01, 0.005, 0.0, 0.011, -0.004, 0.009], dtype="float64")

    # When
    a = block_bootstrap_paths(x, n_paths=50, path_len=37, seed=1)
    b = block_bootstrap_paths(x, n_paths=50, path_len=37, seed=1)
    c = block_bootstrap_paths(x, n_paths=50, path_len=37, seed=2)

    # Then: shape, determinism, resampled-from-source
    assert a.shape == (50, 37)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.isin(a, x).all()

    # Then: fail-closed inputs
    with pytest.raises(ValueError):
        block_bootstrap_paths(np.array([], dtype="float64"), n_paths=10, path_len=5, seed=1)
    with pytest.raises(ValueError):
        block_bootstrap_paths(np.array([0.1, np.nan], dtype="float64"), n_paths=10, path_len=5, seed=1)
    with pytest.raises(ValueError):
        block_bootstrap_paths(x, n_paths=0, path_len=5, seed=1)
    with pytest.raises(ValueError):
        block_bootstrap_paths(x, n_paths=10, path_len=0, seed=1)


def test_annualized_log_growth_lcb_separates_edge_from_zero_edge() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.mhs.deploy_gate import annualized_log_growth_lcb

    # Given: 드리프트 있는 계열 vs 무엣지 잡음 계열
    rng = np.random.default_rng(101)
    idx = pd.date_range("2024-01-01", periods=1000, freq="1D", tz="UTC")
    edge = pd.Series(rng.normal(0.004, 0.02, 1000), index=idx, dtype="float64")
    noise = pd.Series(rng.normal(0.0, 0.02, 1000), index=idx, dtype="float64")

    # When
    edge_lcb, edge_point = annualized_log_growth_lcb(edge, n_paths=400, seed=5)
    noise_lcb, noise_point = annualized_log_growth_lcb(noise, n_paths=400, seed=5)

    # Then
    assert edge_lcb > 0.0
    assert noise_lcb <= 0.0
    assert edge_lcb <= edge_point
    assert noise_lcb <= noise_point
    assert edge_point == pytest.approx(float(np.log1p(edge.to_numpy()).mean() * 365.0), rel=0.0, abs=1e-12)

    # Then: fail-closed on undefined log1p and bad alpha
    with pytest.raises(ValueError):
        annualized_log_growth_lcb(pd.Series([-1.5, 0.01], index=idx[:2], dtype="float64"), n_paths=10, seed=1)
    with pytest.raises(ValueError):
        annualized_log_growth_lcb(edge, alpha=0.0, n_paths=10, seed=1)


def test_profitable_fold_critical_count_matches_exact_binomial() -> None:
    import pytest
    from scipy.stats import binom

    from src.common.errors import DataIntegrityError
    from src.mhs.deploy_gate import profitable_fold_critical_count

    # Given / When
    k = profitable_fold_critical_count(16, alpha=0.05)

    # Then: 정확한 이항 임계
    assert k == 12
    assert float(binom.sf(k - 1, 16, 0.5)) <= 0.05
    assert float(binom.sf(k - 2, 16, 0.5)) > 0.05
    assert float(binom.sf(11, 16, 0.5)) == pytest.approx(0.038406, abs=1e-6)
    assert float(binom.sf(10, 16, 0.5)) == pytest.approx(0.105057, abs=1e-6)

    # Then: 임계는 alpha에 대해 단조 비증가
    assert profitable_fold_critical_count(16, alpha=0.10) <= k

    # Then: 통과 불가능한 n은 조용히 통과시키지 않는다 (0.5**4 = 0.0625 > 0.05)
    with pytest.raises(DataIntegrityError):
        profitable_fold_critical_count(4, alpha=0.05)
    with pytest.raises(ValueError):
        profitable_fold_critical_count(0, alpha=0.05)


def test_survival_probabilities_are_distribution_not_path() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.mhs.deploy_gate import survival_probabilities

    # Given: 실현 경로 자체의 낙폭은 예산(60%) 안이지만 꼬리가 두꺼운 계열
    rng = np.random.default_rng(11)
    idx = pd.date_range("2022-01-01", periods=800, freq="1D", tz="UTC")
    r = pd.Series(rng.normal(0.003, 0.03, 800), index=idx, dtype="float64")
    eq = (1.0 + r).cumprod()
    realized_mdd = float((eq / eq.cummax() - 1.0).min())
    assert realized_mdd > -0.60

    # When
    p_mdd_1x, p_ruin_1x = survival_probabilities(r, max_drawdown=0.60, ruin_fraction=0.60, horizon_years=3.0, n_paths=400, seed=9)
    p_mdd_3x, p_ruin_3x = survival_probabilities(r * 3.0, max_drawdown=0.60, ruin_fraction=0.60, horizon_years=3.0, n_paths=400, seed=9)

    # Then: 유효 확률이며 레버리지에 단조
    for p in (p_mdd_1x, p_ruin_1x, p_mdd_3x, p_ruin_3x):
        assert 0.0 <= p <= 1.0
    assert p_mdd_3x > p_mdd_1x
    # Then: 실현 경로가 예산을 지켜도 분포 확률은 0이 아니다 (범주 오류의 직접 재현)
    assert p_mdd_1x > 0.0

    # Then: fail-closed bounds
    with pytest.raises(ValueError):
        survival_probabilities(r, max_drawdown=0.0, ruin_fraction=0.60, horizon_years=3.0, n_paths=50, seed=1)
    with pytest.raises(ValueError):
        survival_probabilities(r, max_drawdown=0.60, ruin_fraction=1.0, horizon_years=3.0, n_paths=50, seed=1)
    with pytest.raises(ValueError):
        survival_probabilities(r, max_drawdown=0.60, ruin_fraction=0.60, horizon_years=0.0, n_paths=50, seed=1)

def test_tail_concentration_detects_planted_time_structure() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.deploy_gate import tail_concentration

    def _folds(mus: list[float], seed: int) -> list[pd.Series]:
        rng = np.random.default_rng(seed)
        out: list[pd.Series] = []
        for i, mu in enumerate(mus):
            idx = pd.date_range("2022-01-01", periods=90, freq="1D", tz="UTC") + pd.Timedelta(days=90 * i)
            out.append(pd.Series(rng.normal(mu, 0.015, 90), index=idx, dtype="float64"))
        return out

    # Given: 균질 16 fold vs 마지막 4개에 성장을 몰아넣은 16 fold
    homogeneous = _folds([0.004] * 16, seed=31)
    concentrated = _folds([0.0002] * 12 + [0.02] * 4, seed=31)

    # When
    h_obs, h_null = tail_concentration(homogeneous, n_draws=400, seed=13)
    c_obs, c_null = tail_concentration(concentrated, n_draws=400, seed=13)

    # Then
    assert h_obs <= h_null
    assert c_obs > c_null
    assert c_obs > h_obs
    assert 0.25 < h_null < 1.0

    # Then: 총성장이 양수가 아니면 집중도는 정의되지 않는다
    losing = _folds([-0.01] * 16, seed=31)
    with pytest.raises(DataIntegrityError):
        tail_concentration(losing, n_draws=100, seed=13)
    with pytest.raises(ValueError):
        tail_concentration(homogeneous[:1], n_draws=100, seed=13)


def test_evaluate_deploy_gate_short_circuits_on_integrity() -> None:
    from src.mhs.deploy_gate import (
        GATE_FOLD_INTEGRITY,
        GATE_LIVE_PARITY_BLOCKED,
        evaluate_deploy_gate,
    )
    from src.mhs.params import GROWTH_RISK_ENVELOPES

    # Given: 무결성 실패 2건, fold 증거는 아예 비어 있음
    envelope = GROWTH_RISK_ENVELOPES["growth_extreme_budgeted"]

    # When
    result = evaluate_deploy_gate(
        fold_returns=(),
        fold_stress_returns=(),
        integrity_reasons=(GATE_LIVE_PARITY_BLOCKED, GATE_FOLD_INTEGRITY),
        envelope=envelope,
    )

    # Then: 축1/축2를 계산하지 않고 단락
    assert result.go is False
    assert result.reason_codes == (GATE_FOLD_INTEGRITY, GATE_LIVE_PARITY_BLOCKED)
    assert result.metrics == {}


def test_evaluate_deploy_gate_blocks_on_edge_axis_before_survival() -> None:
    import numpy as np
    import pandas as pd

    from src.mhs.deploy_gate import (
        GATE_EDGE_BREADTH,
        GATE_OOS_GROWTH,
        GATE_STRESS_GROWTH,
        evaluate_deploy_gate,
    )
    from src.mhs.params import GROWTH_RISK_ENVELOPES

    # Given: 16개 무엣지 fold
    rng = np.random.default_rng(77)
    folds: list[pd.Series] = []
    for i in range(16):
        idx = pd.date_range("2022-01-01", periods=90, freq="1D", tz="UTC") + pd.Timedelta(days=90 * i)
        folds.append(pd.Series(rng.normal(0.0, 0.02, 90), index=idx, dtype="float64"))
    stress = [s - 0.0006 for s in folds]

    # When
    result = evaluate_deploy_gate(
        fold_returns=folds,
        fold_stress_returns=stress,
        integrity_reasons=(),
        envelope=GROWTH_RISK_ENVELOPES["growth_extreme_budgeted"],
        n_paths=300,
        n_draws=200,
    )

    # Then: 엣지 축에서 차단
    assert result.go is False
    assert GATE_OOS_GROWTH in result.reason_codes or GATE_STRESS_GROWTH in result.reason_codes or GATE_EDGE_BREADTH in result.reason_codes
    assert result.metrics["n_folds"] == 16.0
    assert result.metrics["breadth_critical_count"] == 12.0
    assert "oos_ann_log_growth_lcb" in result.metrics
    # 축2는 계산되지 않는다
    assert "p_mdd_breach" not in result.metrics
    assert "tail_share_observed" not in result.metrics


def test_evaluate_deploy_gate_passes_all_three_axes() -> None:
    import numpy as np
    import pandas as pd

    from src.mhs.deploy_gate import evaluate_deploy_gate
    from src.mhs.params import GROWTH_RISK_ENVELOPES

    # Given: 16개 균질 + 뚜렷한 드리프트 fold
    rng = np.random.default_rng(4242)
    folds: list[pd.Series] = []
    for i in range(16):
        idx = pd.date_range("2022-01-01", periods=90, freq="1D", tz="UTC") + pd.Timedelta(days=90 * i)
        folds.append(pd.Series(rng.normal(0.005, 0.012, 90), index=idx, dtype="float64"))
    stress = [s - 0.0004 for s in folds]

    # When
    result = evaluate_deploy_gate(
        fold_returns=folds,
        fold_stress_returns=stress,
        integrity_reasons=(),
        envelope=GROWTH_RISK_ENVELOPES["growth_extreme_budgeted"],
        n_paths=400,
        n_draws=400,
    )

    # Then
    assert result.reason_codes == ()
    assert result.go is True
    for key in (
        "n_folds", "breadth_critical_count", "profitable_folds", "profitable_folds_stress",
        "oos_ann_log_growth", "oos_ann_log_growth_lcb", "stress_ann_log_growth", "stress_ann_log_growth_lcb",
        "p_mdd_breach", "mdd_budget", "mdd_budget_prob", "p_ruin", "ruin_fraction", "ruin_max_prob",
        "tail_share_observed", "tail_share_null_quantile",
    ):
        assert key in result.metrics, key
        assert isinstance(result.metrics[key], float)
    assert result.metrics["oos_ann_log_growth_lcb"] > 0.0
    assert result.metrics["profitable_folds"] >= result.metrics["breadth_critical_count"]


def test_integrity_reasons_from_report_is_fail_closed_on_missing_evidence() -> None:
    import dataclasses
    import types

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deploy_gate import (
        GATE_BLEND_LEDGER_INVALID,
        GATE_FOLD_INTEGRITY,
        GATE_INPUT_UNSEALED,
        GATE_LIVE_PARITY_BLOCKED,
        GATE_REPORT_NOT_COMPLETE,
        integrity_reasons_from_report,
    )
    from src.mhs.pipeline.config import MhsRunConfig

    plain = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    trim = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(name_drift_trim=True)))

    # Given: 아무 증거도 없는 리포트 -> 전부 fail-closed
    empty = types.SimpleNamespace()
    reasons = integrity_reasons_from_report(empty, plain)
    assert set(reasons) == {
        GATE_REPORT_NOT_COMPLETE, GATE_FOLD_INTEGRITY,
        GATE_BLEND_LEDGER_INVALID, GATE_INPUT_UNSEALED,
    }
    assert list(reasons) == sorted(reasons)

    # Given: 완전한 리포트
    good_fold = types.SimpleNamespace(fold_index=0, strict=object(), stress=object(), failures=())
    good = types.SimpleNamespace(
        status="COMPLETE",
        folds=(good_fold,),
        blend=types.SimpleNamespace(primary=types.SimpleNamespace(ledger=types.SimpleNamespace(primary_valid=True))),
        backtest_reliability=types.SimpleNamespace(input_manifest_digest="a" * 64),
    )
    assert integrity_reasons_from_report(good, plain) == ()

    # Then: 라이브 미지원 기능은 차단
    assert integrity_reasons_from_report(good, trim) == (GATE_LIVE_PARITY_BLOCKED,)

    # Then: fold 실패 코드는 무결성 위반
    bad_fold = types.SimpleNamespace(fold_index=0, strict=object(), stress=object(), failures=("RELEVANT_EXECUTION_DATA_GAP",))
    bad = types.SimpleNamespace(
        status="COMPLETE",
        folds=(bad_fold,),
        blend=good.blend,
        backtest_reliability=good.backtest_reliability,
    )
    assert integrity_reasons_from_report(bad, plain) == (GATE_FOLD_INTEGRITY,)


def test_deploy_gate_from_report_reads_strict_fold_ledgers() -> None:
    import dataclasses
    import types

    import numpy as np
    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deploy_gate import GATE_REPORT_NOT_COMPLETE, deploy_gate_from_report, fold_daily_returns
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(growth_envelope="growth_extreme_budgeted")))

    def _replay(mu: float, start: str, seed: int) -> types.SimpleNamespace:
        rng = np.random.default_rng(seed)
        idx = pd.date_range(start, periods=90, freq="1D", tz="UTC")
        eq = pd.Series((1.0 + rng.normal(mu, 0.012, 90)).cumprod(), index=idx, dtype="float64")
        return types.SimpleNamespace(ledger=types.SimpleNamespace(equity=eq))

    # Given: fold_index가 뒤섞여 들어온 16개 fold
    folds = []
    for i in range(16):
        start = (pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=90 * i)).strftime("%Y-%m-%d")
        folds.append(types.SimpleNamespace(
            fold_index=i, failures=(),
            strict=_replay(0.005, start, 1000 + i),
            stress=_replay(0.0045, start, 1000 + i),
        ))
    shuffled = tuple(folds[::-1])
    report = types.SimpleNamespace(
        status="COMPLETE", folds=shuffled,
        blend=types.SimpleNamespace(primary=types.SimpleNamespace(ledger=types.SimpleNamespace(primary_valid=True))),
        backtest_reliability=types.SimpleNamespace(input_manifest_digest="b" * 64),
    )

    # When
    result = deploy_gate_from_report(report, request, n_paths=300, n_draws=300)

    # Then: 원장이 실제로 소비되었다
    assert result.metrics["n_folds"] == 16.0
    assert result.metrics["oos_ann_log_growth_lcb"] > 0.0
    assert len(fold_daily_returns(folds[0].strict)) == 89

    # Then: 무결성 실패는 원장을 읽지 않고 단락
    incomplete = types.SimpleNamespace(status="RUNNING", folds=shuffled, blend=report.blend, backtest_reliability=report.backtest_reliability)
    blocked = deploy_gate_from_report(incomplete, request, n_paths=300, n_draws=300)
    assert blocked.go is False
    assert GATE_REPORT_NOT_COMPLETE in blocked.reason_codes
    assert blocked.metrics == {}


def test_deploy_gate_fail_closed_guards() -> None:
    """Supplementary: contracted fail-closed guards with no dedicated scenario skeleton."""
    import types

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.deploy_gate import (
        annualized_log_growth_lcb,
        evaluate_deploy_gate,
        fold_daily_returns,
        profitable_fold_critical_count,
    )
    from src.mhs.params import GROWTH_RISK_ENVELOPES

    with pytest.raises(ValueError, match="at least 2 rows"):
        annualized_log_growth_lcb(pd.Series([0.01], dtype="float64"), n_paths=10, seed=1)
    with pytest.raises(ValueError, match="finite"):
        annualized_log_growth_lcb(
            pd.Series([0.01, float("inf")], dtype="float64"), n_paths=10, seed=1
        )
    with pytest.raises(ValueError, match="alpha"):
        profitable_fold_critical_count(16, alpha=0.0)
    with pytest.raises(DataIntegrityError, match="replay is None"):
        fold_daily_returns(None)
    with pytest.raises(DataIntegrityError, match="no ledger"):
        fold_daily_returns(types.SimpleNamespace())
    with pytest.raises(DataIntegrityError, match="no equity"):
        fold_daily_returns(types.SimpleNamespace(ledger=types.SimpleNamespace()))
    s = pd.Series([0.01, -0.02], dtype="float64")
    with pytest.raises(ValueError, match="match in length"):
        evaluate_deploy_gate(
            fold_returns=(s,),
            fold_stress_returns=(s, s),
            integrity_reasons=(),
            envelope=GROWTH_RISK_ENVELOPES["growth_extreme_budgeted"],
        )

def test_evaluate_deploy_gate_blocks_on_mdd_and_ruin_budgets() -> None:
    import numpy as np
    import pandas as pd

    from src.mhs.deploy_gate import GATE_MDD_BUDGET, GATE_RUIN_BUDGET, GATE_TIME_CONCENTRATION, evaluate_deploy_gate
    from src.mhs.params import GrowthRiskEnvelope

    # Given: 축1을 여유롭게 통과하는 16개 균질 fold
    rng = np.random.default_rng(4242)
    folds: list[pd.Series] = []
    for i in range(16):
        idx = pd.date_range("2022-01-01", periods=90, freq="1D", tz="UTC") + pd.Timedelta(days=90 * i)
        folds.append(pd.Series(rng.normal(0.005, 0.012, 90), index=idx, dtype="float64"))
    stress = [s - 0.0004 for s in folds]

    # Given: 짧은 지평 + 엄격 예산 봉투(등록 정책 입력만 바꾼다)
    strict = GrowthRiskEnvelope(
        name="unit_test_strict",
        max_drawdown=0.02,
        max_drawdown_prob=0.001,
        ruin_fraction=0.99,
        max_ruin_prob=0.001,
        horizon_years=0.01,
        leverage_ceiling=1.0,
    )

    # When
    result = evaluate_deploy_gate(
        fold_returns=folds,
        fold_stress_returns=stress,
        integrity_reasons=(),
        envelope=strict,
        n_paths=400,
        n_draws=400,
    )

    # Then: 생존 축에서만 차단되고 엣지 축은 통과했다
    assert result.go is False
    assert result.reason_codes == (GATE_MDD_BUDGET, GATE_RUIN_BUDGET)
    assert GATE_TIME_CONCENTRATION not in result.reason_codes
    assert result.metrics["oos_ann_log_growth_lcb"] > 0.0
    assert result.metrics["profitable_folds"] >= result.metrics["breadth_critical_count"]
    assert result.metrics["p_mdd_breach"] > strict.max_drawdown_prob
    assert result.metrics["p_ruin"] > strict.max_ruin_prob
    assert result.metrics["tail_share_observed"] <= result.metrics["tail_share_null_quantile"]

def test_evaluate_deploy_gate_blocks_on_time_concentrated_growth() -> None:
    import numpy as np
    import pandas as pd

    from src.mhs.deploy_gate import (
        GATE_MDD_BUDGET,
        GATE_RUIN_BUDGET,
        GATE_TIME_CONCENTRATION,
        evaluate_deploy_gate,
    )
    from src.mhs.params import GROWTH_RISK_ENVELOPES

    # Given: 앞 12개는 약하지만 확실한 양(+), 마지막 4개에 성장이 몰린 16 fold
    rng = np.random.default_rng(909)
    mus = [0.0025] * 12 + [0.02] * 4
    folds: list[pd.Series] = []
    for i, mu in enumerate(mus):
        idx = pd.date_range("2022-01-01", periods=90, freq="1D", tz="UTC") + pd.Timedelta(days=90 * i)
        folds.append(pd.Series(rng.normal(mu, 0.008, 90), index=idx, dtype="float64"))
    stress = [s - 0.0002 for s in folds]

    # When
    result = evaluate_deploy_gate(
        fold_returns=folds,
        fold_stress_returns=stress,
        integrity_reasons=(),
        envelope=GROWTH_RISK_ENVELOPES["growth_extreme_budgeted"],
        n_paths=400,
        n_draws=400,
    )

    # Then: 엣지와 생존 예산은 통과, 시간집중만으로 차단
    assert result.go is False
    assert result.reason_codes == (GATE_TIME_CONCENTRATION,)
    assert GATE_MDD_BUDGET not in result.reason_codes
    assert GATE_RUIN_BUDGET not in result.reason_codes
    assert result.metrics["oos_ann_log_growth_lcb"] > 0.0
    assert result.metrics["stress_ann_log_growth_lcb"] > 0.0
    assert result.metrics["profitable_folds"] == 16.0
    assert result.metrics["profitable_folds_stress"] == 16.0
    assert result.metrics["tail_share_observed"] > result.metrics["tail_share_null_quantile"]

def test_integrity_reasons_accepts_terminal_only_blend_ledger() -> None:
    # Given: a COMPLETE report whose blend ledger is invalid only via UNKNOWN_TERMINATION
    from types import SimpleNamespace

    import pandas as pd

    from src.mhs.deploy_gate import GATE_BLEND_LEDGER_INVALID, integrity_reasons_from_report
    from src.mhs.execution import ExecutionDataGap

    terminal_gaps = (
        ExecutionDataGap(
            code="UNKNOWN_TERMINATION", symbol="AAAUSDT",
            timestamp=pd.Timestamp("2025-12-31", tz="UTC"),
        ),
    )

    def _report(gaps, fills):
        return SimpleNamespace(
            status="COMPLETE",
            folds=[SimpleNamespace(fold_index=0, strict=object(), failures=())],
            blend=SimpleNamespace(
                primary=SimpleNamespace(
                    ledger=SimpleNamespace(primary_valid=False, data_gaps=gaps),
                    simulated_fills=fills,
                )
            ),
            backtest_reliability=SimpleNamespace(input_manifest_digest="deadbeef"),
        )

    request = SimpleNamespace(name_drift_trim=False)
    recovering_fills = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-05", tz="UTC")],
            "symbol": ["AAAUSDT"],
            "quantity_delta": [1.0],
            "fill_price": [1.0],
            "fee_bps": [0.0],
            "reason": ["timeout_taker"],
            "pre_trade_equity": [1.0],
        }
    )
    recovering_gaps = (
        ExecutionDataGap(
            code="MISSING_HELD_FUNDING", symbol="AAAUSDT",
            timestamp=pd.Timestamp("2025-06-01", tz="UTC"),
        ),
    )

    # When
    certified = integrity_reasons_from_report(_report(terminal_gaps, pd.DataFrame()), request)
    uncertified = integrity_reasons_from_report(_report(recovering_gaps, recovering_fills), request)

    # Then: terminal inventory clears axis 0; a recovering gap still blocks
    assert GATE_BLEND_LEDGER_INVALID not in certified
    assert certified == ()
    assert GATE_BLEND_LEDGER_INVALID in uncertified
