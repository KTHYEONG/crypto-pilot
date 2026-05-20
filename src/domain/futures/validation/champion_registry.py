from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChampionMetricsV3:
    """v3.0 multi-dimensional benchmark metrics for Champion promotion."""

    atomic_oos_pass_ratio: float
    capacity_ceiling_usdt: float
    median_log_growth: float
    worst_block_mdd: float
    absolute_decay_bps_yr: float
    dsr: float
    # Backward compatibility fields
    cagr: float = 0.0
    mdd: float = 0.0
    sharpe: float = 0.0


@dataclass
class PromotionGateResult:
    """Result of the sequential promotion gate evaluation."""

    passed: bool
    gate_failures: list[str]
    promoted_to_champion: bool


@dataclass(frozen=True)
class ChampionMetrics:
    """Benchmark metrics loaded from champion JSON for guard comparison."""

    cagr: float = 0.0
    mdd: float = 100.0
    net_alpha: float = 0.0
    sharpe: float = 0.0
    pbo: float = 1.0


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _metrics_from_payload(payload: dict[str, Any]) -> ChampionMetrics:
    raw = payload.get("metrics")
    if isinstance(raw, dict):
        m = raw
    else:
        m = payload
    if not isinstance(m, dict):
        return ChampionMetrics()
    return ChampionMetrics(
        cagr=_to_float(m.get("oos_cagr_pct", 0.0), 0.0),
        mdd=abs(_to_float(m.get("oos_mdd_pct", 100.0), 100.0)),
        net_alpha=_to_float(m.get("oos_net_alpha_pct", 0.0), 0.0),
        sharpe=_to_float(m.get("oos_sharpe_ratio", 0.0), 0.0),
        pbo=_to_float(m.get("pbo_paired", m.get("pbo", 1.0)), 1.0),
    )


def default_baseline_file(project_root: Path) -> Path:
    """Canonical baseline path (may not exist on disk)."""
    return project_root / "config" / "champion_baseline.json"


def resolve_baseline_file(project_root: Path) -> Path | None:
    """Prefer champion_baseline.json, then champion_baseline_v2.json (compat)."""
    for name in ("champion_baseline.json", "champion_baseline_v2.json"):
        p = project_root / "config" / name
        if p.exists():
            return p
    return None


def resolve_champion_record_path(logs_dir: Path) -> Path | None:
    """Prefer champion_v2.json, then champion.json (avoids stale legacy champion.json)."""
    for name in ("champion_v2.json", "champion.json"):
        p = logs_dir / name
        if p.exists():
            return p
    return None


def load_champion_metrics(path: Path) -> ChampionMetrics:
    """Load metrics from a single JSON file (ChampionRecordV2 or bare metrics)."""
    if not path.exists():
        return ChampionMetrics()
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return _metrics_from_payload(payload)
    except Exception:
        return ChampionMetrics()


def load_champion_metrics_for_guard(logs_dir: Path, project_root: Path) -> ChampionMetrics:
    """Resolve logs champion file; fallback to tracked config baseline; else empty metrics."""
    resolved = resolve_champion_record_path(logs_dir)
    if resolved is not None:
        return load_champion_metrics(resolved)
    baseline = resolve_baseline_file(project_root)
    if baseline is not None:
        return load_champion_metrics(baseline)
    return ChampionMetrics()


def should_promote_candidate(
    champion: ChampionMetrics,
    candidate: ChampionMetrics,
    pbo_strict_max: float,
    bypass: bool = False,
) -> tuple[bool, str]:
    if bypass:
        return True, "bypass"
    if candidate.pbo > float(pbo_strict_max):
        return False, "pbo_strict_fail"
    champ_romad = champion.cagr / max(champion.mdd, 1e-9)
    cand_romad = candidate.cagr / max(candidate.mdd, 1e-9)
    alpha_ok = candidate.net_alpha >= (champion.net_alpha - 2.0)
    better = (
        (candidate.sharpe > champion.sharpe + 0.05 and alpha_ok)
        or (cand_romad > champ_romad * 1.02 and alpha_ok)
        or (candidate.net_alpha > champion.net_alpha + 0.5)
        or (candidate.cagr > champion.cagr + 2.0 and cand_romad >= champ_romad * 0.95 and alpha_ok)
    )
    return (better, "improved" if better else "no_meaningful_improvement")


def champion_history_path(project_root: Path) -> Path:
    return project_root / "logs" / "champion_history.jsonl"


def append_champion_history(project_root: Path, record: dict[str, Any]) -> None:
    """Append one JSON line for champion lineage (append-only)."""
    path = champion_history_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_champion_record(
    project_root: Path,
    champion_data: dict[str, Any],
) -> Path:
    """Write canonical logs/champion.json (caller builds payload)."""
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "champion.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(champion_data, f, indent=2, ensure_ascii=False)
    return path


def write_champion_v2_record(
    project_root: Path,
    champion_data: dict[str, Any],
) -> Path:
    """Forward to write_champion_record; kept for backward compatibility."""
    return write_champion_record(project_root, champion_data)


def run_champion_promotion_guard(
    logs_dir: Path,
    project_root: Path,
    candidate: ChampionMetrics,
    pbo_strict_max: float,
    bypass: bool,
) -> tuple[bool, str]:
    """Load baseline/champion and decide promotion vs economic guard."""
    champ = load_champion_metrics_for_guard(logs_dir, project_root)
    return should_promote_candidate(champ, candidate, pbo_strict_max, bypass=bypass)


def build_champion_record_payload(
    *,
    run_id: str,
    promoted_at: str,
    note: str,
    architecture: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    gates: dict[str, Any],
    schema_version: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "id": run_id,
        "promoted_at": promoted_at,
        "note": note,
        "architecture": architecture,
        "parameters": parameters,
        "metrics": metrics,
        "gates": gates,
    }


def should_promote_candidate_v3(
    candidate: ChampionMetricsV3,
    champion: ChampionMetricsV3,
) -> bool:
    """Compare candidate against champion using multi-dimensional v3.0 logic.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    # 1. atomic_oos_pass_ratio가 5%p 이상 높으면 승격
    if candidate.atomic_oos_pass_ratio >= champion.atomic_oos_pass_ratio + 0.05:
        return True
    elif candidate.atomic_oos_pass_ratio <= champion.atomic_oos_pass_ratio - 0.05:
        return False

    # 2. capacity_ceiling 더 높음 (10% 이상 차이)
    if candidate.capacity_ceiling_usdt >= champion.capacity_ceiling_usdt * 1.10:
        return True
    elif candidate.capacity_ceiling_usdt <= champion.capacity_ceiling_usdt * 0.90:
        return False

    # 3. median_log_growth 더 높음
    if candidate.median_log_growth > champion.median_log_growth + 1e-9:
        return True
    elif candidate.median_log_growth < champion.median_log_growth - 1e-9:
        return False

    # 4. worst_block_mdd 더 낮음 (5%p 이상 차이)
    if candidate.worst_block_mdd <= champion.worst_block_mdd - 0.05:
        return True
    elif candidate.worst_block_mdd >= champion.worst_block_mdd + 0.05:
        return False

    # 5. absolute_decay_bps 덜 부정적
    if candidate.absolute_decay_bps_yr > champion.absolute_decay_bps_yr + 1e-9:
        return True
    elif candidate.absolute_decay_bps_yr < champion.absolute_decay_bps_yr - 1e-9:
        return False

    # 동률: dsr 더 높음
    return candidate.dsr > champion.dsr


def evaluate_sequential_promotion_gate(
    candidate: ChampionMetricsV3,
    champion: ChampionMetricsV3 | None,
    wf_result: Any,
    dual_decay: Any,
    atomic_result: Any,
    capacity_results: dict[int, bool],
    intrabar_tw: float,
    intrabar_mdd: float,
    mdd_hard_limit: float = 0.50,
) -> PromotionGateResult:
    """Evaluate sequential promotion gates for backtest v3.0 champion.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    failures: list[str] = []

    # 1. Inner AWF hard gates (WalkForwardResult.passed)
    if not getattr(wf_result, "passed", False):
        failures.append("AWF_HARD_GATES")

    # 2. Atomic 6M blocks pass_ratio >= 0.70
    if getattr(atomic_result, "pass_ratio", 0.0) < 0.70:
        failures.append("ATOMIC_PASS_RATIO")

    # 3. Intrabar 1m: TW > 1.0 AND MDD < mdd_hard_limit
    if intrabar_tw <= 1.0:
        failures.append("INTRABAR_TW")
    if intrabar_mdd >= mdd_hard_limit:
        failures.append("INTRABAR_MDD")

    # 4. Dual decay 통과
    if not getattr(dual_decay, "passed", False):
        failures.append("DUAL_DECAY")

    # 5. AUM ladder 50k, 100k, 250k 전부 pass
    for tier in (50000, 100000, 250000):
        if not capacity_results.get(tier, False):
            failures.append("CAPACITY_LADDER")
            break

    passed = len(failures) == 0
    promoted_to_champion = False

    if passed:
        if champion is None:
            promoted_to_champion = True
        else:
            promoted_to_champion = should_promote_candidate_v3(candidate, champion)

    return PromotionGateResult(
        passed=passed,
        gate_failures=failures,
        promoted_to_champion=promoted_to_champion,
    )
