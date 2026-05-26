"""Promotion research gates and validation blocks for futures strategies."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.domain.futures.optimization.opt_config import default_ev_hurdle_bps

# --- Atomic Blocks (6M non-overlapping) ---

# 6M ≈ 182.5일 (밀리초 단위)
_MS_PER_DAY: int = 86_400_000
_DAYS_PER_6M: float = 182.5
_6M_MS: int = int(_DAYS_PER_6M * _MS_PER_DAY)


@dataclass(frozen=True)
class AtomicBlockConfig:
    """Atomic block 평가 설정.

    Attributes:
        block_months: 블록 기간 (고정 6M).
        min_pass_ratio: 통과 블록 비율 최소값.
        required_min_blocks: 판정을 위한 최소 block 수.

    """

    block_months: int = 6
    min_pass_ratio: float = 0.70
    required_min_blocks: int = 3


@dataclass
class AtomicBlockResult:
    """Atomic block 평가 결과.

    Attributes:
        n_blocks: 총 블록 수.
        n_passed: 통과 블록 수.
        pass_ratio: n_passed / n_blocks.
        passed: pass_ratio >= min_pass_ratio AND n_blocks >= required_min_blocks.
        block_log_tws: 각 블록의 log Terminal Wealth.
        worst_block_mdd: 최악 블록의 최대 낙폭 (0~1 scale).
        median_log_growth: 중앙값 log TW.

    """

    n_blocks: int
    n_passed: int
    pass_ratio: float
    passed: bool
    block_log_tws: list[float]
    worst_block_mdd: float
    median_log_growth: float


def build_atomic_blocks(
    timestamps: np.ndarray,
    is_end_ts: int,
    block_months: int = 6,
) -> list[tuple[int, int]]:
    """Non-overlapping 6M 시작/끝 인덱스 쌍 반환.

    IS 이후 타임스탬프부터 block_months 단위로 non-overlap 분할한다.

    Args:
        timestamps: decision bar timestamps (UTC unix ms), shape [T].
        is_end_ts: IS 종료 시점 (이 값 이상인 bar부터 OOS).
        block_months: 블록 길이 (기본 6).

    Returns:
        list of (start_idx, end_idx) — end_idx는 exclusive.

    """
    ts = np.asarray(timestamps, dtype=np.int64)
    n = int(ts.size)
    if n == 0:
        return []

    # IS 종료 이후 첫 인덱스 탐색
    oos_start_idx = int(np.searchsorted(ts, is_end_ts, side="left"))
    if oos_start_idx >= n:
        return []

    # 6M 밀리초
    block_ms = int(_DAYS_PER_6M * _MS_PER_DAY * (block_months / 6))

    blocks: list[tuple[int, int]] = []
    cur_start_idx = oos_start_idx

    while cur_start_idx < n:
        block_end_ts = ts[cur_start_idx] + block_ms
        # block_end_ts 이상인 첫 인덱스
        cur_end_idx = int(np.searchsorted(ts, block_end_ts, side="left"))

        if cur_end_idx > cur_start_idx:
            blocks.append((cur_start_idx, cur_end_idx))
        elif cur_end_idx == cur_start_idx:
            # 전진 불가 → 종료
            break

        cur_start_idx = cur_end_idx

    return blocks


def _calc_block_log_tw(equity_curve: np.ndarray) -> float:
    """블록 equity curve에서 log Terminal Wealth 계산."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    start = float(eq[0])
    end = float(eq[-1])
    if start <= 1e-15:
        return -10.0
    ratio = end / start
    if ratio <= 1e-15:
        return -10.0
    return float(math.log(ratio))


def _calc_block_mdd(equity_curve: np.ndarray) -> float:
    """블록 equity curve의 최대 낙폭 (0~1)."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    running_max = np.where(running_max < 1e-15, 1e-15, running_max)
    dd = (eq - running_max) / running_max
    return float(abs(np.min(dd)))


def evaluate_atomic_blocks(
    equity_curves: list[np.ndarray],
    config: AtomicBlockConfig = AtomicBlockConfig(),
) -> AtomicBlockResult:
    """블록별 equity curve를 평가하여 AtomicBlockResult 반환.

    Args:
        equity_curves: 각 block별 equity curve. len == n_blocks.
        config: AtomicBlockConfig 설정.

    Returns:
        AtomicBlockResult with pass/fail 판정.

    """
    n_blocks = len(equity_curves)

    if n_blocks < config.required_min_blocks:
        return AtomicBlockResult(
            n_blocks=n_blocks,
            n_passed=0,
            pass_ratio=0.0,
            passed=False,
            block_log_tws=[],
            worst_block_mdd=0.0,
            median_log_growth=0.0,
        )

    block_log_tws: list[float] = []
    block_mdds: list[float] = []
    n_passed = 0

    for ec in equity_curves:
        log_tw = _calc_block_log_tw(ec)
        mdd = _calc_block_mdd(ec)
        block_log_tws.append(log_tw)
        block_mdds.append(mdd)
        if math.exp(log_tw) >= 1.0:
            n_passed += 1

    pass_ratio = float(n_passed) / float(n_blocks) if n_blocks > 0 else 0.0
    worst_mdd = float(max(block_mdds)) if block_mdds else 0.0
    arr_log_tw = np.array(block_log_tws, dtype=np.float64)
    median_log = float(np.median(arr_log_tw)) if arr_log_tw.size > 0 else 0.0

    passed = pass_ratio >= config.min_pass_ratio

    return AtomicBlockResult(
        n_blocks=n_blocks,
        n_passed=n_passed,
        pass_ratio=pass_ratio,
        passed=passed,
        block_log_tws=block_log_tws,
        worst_block_mdd=worst_mdd,
        median_log_growth=median_log,
    )


# --- Boundary Contract (Purge Bars) ---

@dataclass
class ModulePurgeBarsMeta:
    """모듈별 purge_bars 메타데이터.

    Attributes:
        module_name: 모듈 식별자.
        purge_bars: IS/OOS 경계 purge 길이 (바 단위).
        reason: purge_bars 값의 근거 (label_horizon / fit_window 등).

    """

    module_name: str
    purge_bars: int
    reason: str


class PurgeBarsRegistry:
    """모든 signal/feature 모듈이 purge_bars를 등록하는 중앙 레지스트리.

    빈 레지스트리에서 get_boundary_purge_bars() 호출 시 RuntimeError를 발생시켜
    미등록 상태의 백테스트 진입을 Fail-fast로 차단한다.
    """

    def __init__(self) -> None:
        self._registry: dict[str, ModulePurgeBarsMeta] = {}

    def register(self, meta: ModulePurgeBarsMeta) -> None:
        """모듈 purge_bars 등록. 동일 모듈명 재등록 시 최신값 갱신.

        Args:
            meta: ModulePurgeBarsMeta 인스턴스.

        """
        self._registry[meta.module_name] = meta

    def get_boundary_purge_bars(self) -> int:
        """등록된 모든 모듈의 purge_bars 중 최대값 반환.

        Returns:
            max(all registered purge_bars).

        Raises:
            RuntimeError: 등록된 모듈이 없을 경우 Fail-fast.

        """
        if not self._registry:
            raise RuntimeError(
                "No modules registered purge_bars. Fail-fast. "
                "백테스트 진입 전 모든 signal/feature 모듈이 purge_bars를 등록해야 합니다."
            )
        return max(m.purge_bars for m in self._registry.values())

    def validate(self) -> None:
        """미등록 상태면 backtest 진입 거부.

        Raises:
            RuntimeError: 등록된 모듈이 없을 경우.

        """
        _ = self.get_boundary_purge_bars()  # RuntimeError 전파

    def list_modules(self) -> list[ModulePurgeBarsMeta]:
        """등록된 모든 모듈 목록 반환."""
        return list(self._registry.values())


# --- Research Gates ---

@dataclass(frozen=True)
class V3HardGates:
    """v3.0 확정 상수 — 8-gate 평가 기준."""

    MIN_POSITIVE_LEG_RATIO: float = 0.55
    WORST_LEG_TW_FLOOR: float = 0.85
    MEAN_LEG_TW_FLOOR: float = 1.015
    ERGODICITY_PCT: float = 15.0
    EV_COST_FLOOR: float = 3.0
    DSR_FLOOR: float = 0.60
    FUNDING_DRAG_CEILING: float = 0.30
    CAPACITY_REQUIRED_TIERS: tuple[int, ...] = (50_000, 100_000, 250_000)


@dataclass
class GateResult:
    """Gate 평가 결과."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def evaluate_v3_hard_gates(
    leg_log_tw: np.ndarray,
    worst_mdd: float,
    dsr: float,
    ev_cost: float,
    funding_drag_ratio: float,
    ergodicity_dev_pct: float,
    capacity_results: dict[int, bool],
    gates: V3HardGates = V3HardGates(),
) -> GateResult:
    """8-gate v3.0 평가.

    Args:
        leg_log_tw: shape [K] — leg별 log Terminal Wealth.
        worst_mdd: 최대 낙폭 (0~1).
        dsr: Deflated Sharpe Ratio (0~1).
        ev_cost: EV/Cost 비율.
        funding_drag_ratio: funding_drag / gross_return (0~1).
        ergodicity_dev_pct: ergodicity deviation (%).
        capacity_results: {aum: pass/fail} — CAPACITY_REQUIRED_TIERS 전부 필요.
        gates: V3HardGates 상수 컨테이너.

    Returns:
        GateResult(passed, failures, metrics).

    """
    arr = np.asarray(leg_log_tw, dtype=np.float64)
    tw_arr = np.exp(arr)

    failures: list[str] = []

    # Gate 1: min positive leg ratio
    pos_ratio = float(np.mean(tw_arr >= 1.0))
    if pos_ratio < gates.MIN_POSITIVE_LEG_RATIO:
        failures.append("WF_POSITIVE_LEG_RATIO")

    # Gate 2: worst leg TW floor
    worst_tw = float(np.min(tw_arr)) if arr.size > 0 else 0.0
    if worst_tw < gates.WORST_LEG_TW_FLOOR:
        failures.append("WF_WORST_LEG_TW")

    # Gate 3: mean leg TW floor
    mean_tw = float(np.mean(tw_arr)) if arr.size > 0 else 0.0
    if mean_tw < gates.MEAN_LEG_TW_FLOOR:
        failures.append("WF_MEAN_LEG_TW")

    # Gate 4: DSR floor
    if float(dsr) < gates.DSR_FLOOR:
        failures.append("DSR_FLOOR")

    # Gate 5: funding drag ceiling
    if float(funding_drag_ratio) > gates.FUNDING_DRAG_CEILING:
        failures.append("FUNDING_DRAG")

    # Gate 6: capacity — CAPACITY_REQUIRED_TIERS 전부 통과 필요
    required_tiers = set(gates.CAPACITY_REQUIRED_TIERS)
    cap_fail = any(
        not capacity_results.get(tier, False)
        for tier in required_tiers
        if tier in capacity_results
    ) or any(
        tier not in capacity_results
        for tier in required_tiers
    )
    if cap_fail:
        failures.append("CAPACITY")

    # Gate 7: ergodicity
    if float(ergodicity_dev_pct) > gates.ERGODICITY_PCT:
        failures.append("WF_ERGODICITY")

    # Gate 8: EV/Cost
    if float(ev_cost) < gates.EV_COST_FLOOR:
        failures.append("EV_COST")

    metrics: dict[str, float] = {
        "pos_ratio": pos_ratio,
        "worst_tw": worst_tw,
        "mean_tw": mean_tw,
        "dsr": float(dsr),
        "funding_drag_ratio": float(funding_drag_ratio),
        "ergodicity_dev_pct": float(ergodicity_dev_pct),
        "ev_cost": float(ev_cost),
    }

    return GateResult(
        passed=len(failures) == 0,
        failures=failures,
        metrics=metrics,
    )


@dataclass(frozen=True)
class FuturesResearchGateInput:
    """Thresholds and observations for one-shot research gate evaluation."""

    phase3_enabled: bool
    pbo_max: float
    dsr_min: float
    is_precision: float
    oos_port: dict[str, Any]
    pbo_obs: float
    dsr_obs: float
    wf_failures: tuple[str, ...]
    min_is_net_alpha_pct: float
    is_net_alpha_pct: float
    min_long_pf: float
    min_short_pf: float
    oos_long_pf: float
    oos_short_pf: float
    is_cagr_pct: float
    is_sharpe: float
    is_survival_min_cagr: float
    is_survival_min_sharpe: float
    worst_leg_log_tw: float
    awf_p10_log_tw_floor: float
    # V3.1 Mechanical Additions
    oos_mdd_duration: float = 0.0
    max_mdd_duration: float = 180.0
    oos_expectancy: float = 0.0
    min_expectancy: float = 0.40
    is_expectancy: float = 0.0
    min_oos_retention_expectancy_pct: float = 50.0
    # Auxiliary diagnostics only (non-blocking): legacy CAGR retention.
    oos_cagr_pct: float = 0.0
    is_cagr_ref_pct: float = 0.0


def evaluate_research_gates(inp: FuturesResearchGateInput) -> tuple[bool, list[str]]:
    """Evaluate ordered research gates; return (ok, failure codes).

    Short-circuits on first blocking group.
    """
    failures: list[str] = []

    if inp.phase3_enabled:
        if not check_hard_gates_ml(
            inp.oos_port,
            float(inp.pbo_obs),
            float(inp.dsr_obs),
            inp.is_precision,
            pbo_max_override=inp.pbo_max,
            dsr_min_override=inp.dsr_min,
        ):
            failures.append("PHASE3_HARD_GATE")
            return False, failures

    if inp.wf_failures:
        failures.extend(inp.wf_failures)
        return False, failures

    # V3.1 Mechanical: Expectancy Gate
    if inp.oos_expectancy < float(inp.min_expectancy):
        failures.append("EXPECTANCY_GATE")
        return False, failures

    # V4.3: OOS retention gate based on expectancy (not CAGR)
    if abs(float(inp.is_expectancy)) > 1e-9:
        _exp_ret = float(inp.oos_expectancy) / float(inp.is_expectancy) * 100.0
    else:
        _exp_ret = 0.0
    if _exp_ret < float(inp.min_oos_retention_expectancy_pct):
        failures.append("OOS_RETENTION_EXPECTANCY_GATE")
        return False, failures

    # V3.1 Mechanical: MDD Duration Gate
    if inp.oos_mdd_duration > float(inp.max_mdd_duration):
        failures.append("MDD_DURATION_GATE")
        return False, failures

    if inp.is_net_alpha_pct <= float(inp.min_is_net_alpha_pct):
        failures.append("IS_ALPHA_GATE")
        return False, failures

    if inp.oos_long_pf < float(inp.min_long_pf) or inp.oos_short_pf < float(inp.min_short_pf):
        failures.append("DIRECTIONAL_PF_GATE")
        return False, failures

    if not (
        inp.is_cagr_pct > float(inp.is_survival_min_cagr)
        and inp.is_sharpe > float(inp.is_survival_min_sharpe)
    ):
        failures.append("IS_SURVIVAL_GATE")
        return False, failures

    if inp.worst_leg_log_tw <= float(inp.awf_p10_log_tw_floor):
        failures.append("AWF_HARDENING_GATE")
        return False, failures

    return True, []


# Human-readable map for logs / ops automation
GATE_CODE_DESCRIPTIONS: dict[str, str] = {
    "PHASE3_HARD_GATE": "PBO/DSR/WR/Mdd/Dir-PF Phase-3 composite gate failed",
    "WF_POSITIVE_LEG_RATIO": "Walk-forward: insufficient fraction of legs with TW>=1",
    "WF_WORST_LEG_TW": "Walk-forward: worst-leg terminal wealth below floor",
    "WF_MEAN_LEG_TW": "Walk-forward: mean leg TW below floor",
    "WF_ERGODICITY": "Walk-forward: path ergodicity deviation above guideline",
    "WF_EMPTY_LEGS": "Walk-forward: no evaluable legs",
    "WF_INVALID_SPAN": "Walk-forward: OOS span or n_legs invalid",
    "IS_ALPHA_GATE": "In-sample net alpha vs BTC below policy floor",
    "DIRECTIONAL_PF_GATE": "OOS long/short profit factor below policy minima",
    "IS_SURVIVAL_GATE": "IS CAGR or IS Sharpe below survival cut",
    "AWF_HARDENING_GATE": "Worst AWF leg log-TW below distributional floor",
    "EXPECTANCY_GATE": "OOS Mean Return per Trade below 0.40% mechanical hurdle",
    "OOS_RETENTION_EXPECTANCY_GATE": "OOS/IS expectancy retention below policy floor",
    "MDD_DURATION_GATE": "OOS Max Drawdown Duration exceeds 180 days",
}


def check_hard_gates_ml(
    oos_result: dict[str, Any],
    pbo_val: float,
    dsr_val: float,
    is_precision: float,
    *,
    pbo_max_override: float | None = None,
    dsr_min_override: float | None = None,
) -> bool:
    """Check if the OOS results pass all mandatory research and stability gates.

    Evaluates PBO, DSR, win rate, MDD, and profit factor against configurable
    thresholds.

    Args:
        oos_result: Dictionary containing out-of-sample performance metrics.
        pbo_val: Observed Probability of Backtest Overfitting.
        dsr_val: Observed Deflated Sharpe Ratio.
        is_precision: In-sample precision (win rate) for comparison.
        pbo_max_override: Optional override for the PBO threshold.
        dsr_min_override: Optional override for the DSR threshold.

    Returns:
        True if all gates are passed, False otherwise.

    """
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
    _logger = logging.getLogger(__name__)

    cfg = OPT_FUTURES_CONFIG
    pbo_lim = float(
        pbo_max_override if pbo_max_override is not None else cfg.get("FUTURES_PBO_MAX", 0.45)
    )
    pbo_ok = pbo_val < pbo_lim
    dsr_floor = float(
        dsr_min_override if dsr_min_override is not None else (
            cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.20)
        )
    )
    dsr_ok = dsr_val >= dsr_floor
    wr_pct = float(oos_result.get("win_rate_pct", oos_result.get("win_rate", 0.0)))
    wr_frac = wr_pct / 100.0 if wr_pct > 1.0 else wr_pct
    wr_ok = wr_frac >= is_precision * 0.85
    mdd_v = float(oos_result.get("mdd_pct", oos_result.get("mdd", 100.0)))
    mdd_ok = abs(mdd_v) < float(cfg.get("FUTURES_MAX_MDD", 25.0))
    l_pf = float(oos_result.get("long_profit_factor", oos_result.get("oos_long_pf", 1.0)))
    s_pf = float(oos_result.get("short_profit_factor", oos_result.get("oos_short_pf", 1.0)))
    combined_pf = float(oos_result.get("profit_factor", (l_pf + s_pf) / 2.0))
    dir_ok = combined_pf >= 1.05

    # V3.1 Mechanical Hurdle: Mean Return per Trade (Expectancy) >= 0.40%
    ev_pct = float(oos_result.get("mean_ret_pct", oos_result.get("expectancy", 0.0)))
    ev_ok = ev_pct >= float(default_ev_hurdle_bps(cfg)) / 100.0
    trades = float(
        oos_result.get(
            "trade_count",
            oos_result.get("n_trades", oos_result.get("oos_trade_count", 0.0)),
        )
    )
    if trades <= 0.0:
        _logger.info(
            " [FINAL-FLAT-DIAG] oos_zero_trades=1 wr_ok=%s mdd_ok=%s pf_ok=%s ev_ok=%s "
            "wr=%.4f mdd=%.4f pf=%.4f ev_pct=%.6f",
            wr_ok,
            mdd_ok,
            dir_ok,
            ev_ok,
            wr_frac,
            mdd_v,
            combined_pf,
            ev_pct,
        )

    return bool(pbo_ok and dsr_ok and wr_ok and mdd_ok and dir_ok and ev_ok)
