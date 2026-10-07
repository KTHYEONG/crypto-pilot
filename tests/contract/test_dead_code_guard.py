"""Standing dead-code contract: retired modules, symbols and private cross-package seams stay gone.

Registries are append-only records of deliberate deletions. A deletion change
appends its entries in the same commit; an entry is removed only if the
retirement itself is reversed by an explicit decision. The private-import
allowlist is shrink-only: every entry must still be observed, so fixing a seam
forces its removal here.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCANNED_ROOTS: Final[tuple[str, ...]] = ("src", "tools")
DELETED_MODULES: Final[tuple[str, ...]] = (
    "src.application.facades",
    "src.application.ops.recorder_fingerprint",
    "src.application.research",
    "src.cli.adapters",
    "src.core",
    "src.market_data.streams.recorder",
    "src.market_data.streams.recorder_main",
    "src.mhs.deployed_weights_ledger",
    "src.mhs.deployment_bundle",
    "src.mhs.live_runtime",
    "src.mhs.live_signal_step",
    "src.mhs.process_backtest",
    "src.mhs.signal_refresh",
    "src.mhs.signal_runtime",
    "src.mhs.signal_state",
    "src.mhs.stage_services",
    "src.research",
    "src.market_data.storage.schemas",
    "src.quant.baseline",
    "src.quant.baseline.backtest",
    "src.quant.baseline.signal",
    "src.quant.contracts",
    "src.quant.evaluation.metrics",
    "src.quant.evaluation.gate_feasibility",
    "src.quant.evaluation.promotion",
    "src.quant.technical_experts.catalog",
    "src.quant.technical_experts.contracts",
    "src.common.settings",
    "src.live.signal_step_result",
    "src.mhs.pipeline.stages.diagnostic",
    "src.quant.technical_experts.trend_screen_catalog",
)
RETIRED_SYMBOLS: Final[tuple[tuple[str, str], ...]] = (
    ("src.application.mhs_supervisor", "PROCESS_INVENTORY_REPORT_PATH"),
    ("src.application.mhs_supervisor", "PROCESS_POLICY_REPORT_PATH"),
    ("src.application.mhs_supervisor", "PROCESS_REPORT_PATH"),
    ("src.cli.commands.data", "_bookdepth"),
    ("src.cli.commands.data", "_indicator_klines"),
    ("src.cli.commands.data", "_metrics"),
    ("src.cli.commands.data", "_refresh_one_symbol_tail"),
    ("src.common.paths", "bookdepth_path"),
    ("src.common.paths", "indicator_kline_path"),
    ("src.common.paths", "metrics_path"),
    ("src.live.runner", "apply_ruin_guard"),
    ("src.live.scheduler", "_load_last_processed"),
    ("src.live.scheduler", "_save_last_processed"),
    ("src.market_data.binance.futures", "BinanceClient.fetch_futures_data_metric"),
    ("src.market_data.binance.futures", "BinanceClient.fetch_mark_price_klines"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader._normalize_metrics_frame"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_bookdepth_daily"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_daily_metrics"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_funding_monthly"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_indicator_klines_daily"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_indicator_klines_monthly"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_klines_archive"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_metrics_daily"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_premiumindex_daily"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.fetch_range_metrics"),
    ("src.market_data.binance.vision", "BinanceVisionDownloader.verify_checksum"),
    ("src.market_data.binance.vision", "_OI_ADV_METRICS_START"),
    ("src.market_data.binance.vision", "fetch_metrics_bulk"),
    ("src.market_data.retention", "MHS_RETIRED_FEEDS"),
    ("src.market_data.retention", "RETIRED_MHS_CLEANUP_SUFFIXES"),
    ("src.market_data.retention", "enumerate_retired_mhs_feed_files"),
    ("src.market_data.retention", "quarantine_retired_mhs_feeds"),
    ("src.market_data.retention", "retired_feed_active_readers"),
    ("src.market_data.services.collection", "collect_bookdepth"),
    ("src.market_data.services.collection", "collect_indicator_klines"),
    ("src.market_data.services.collection", "collect_metrics"),
    ("src.market_data.services.futures_collection", "DataCollector._bookdepth_coverage_report"),
    ("src.market_data.services.futures_collection", "DataCollector._load_bookdepth_cache"),
    ("src.market_data.services.futures_collection", "DataCollector._load_indicator_kline_cache"),
    ("src.market_data.services.futures_collection", "DataCollector._load_mark_price_cache"),
    ("src.market_data.services.futures_collection", "DataCollector._load_metrics_cache"),
    ("src.market_data.services.futures_collection", "DataCollector._mark_price_coverage"),
    ("src.market_data.services.futures_collection", "DataCollector._merge_metrics_frames"),
    ("src.market_data.services.futures_collection", "DataCollector._metrics_coverage_report"),
    ("src.market_data.services.futures_collection", "DataCollector._normalize_bookdepth_frame"),
    ("src.market_data.services.futures_collection", "DataCollector._normalize_indicator_kline_frame"),
    ("src.market_data.services.futures_collection", "DataCollector._safe_symbol"),
    ("src.market_data.services.futures_collection", "DataCollector._save_bookdepth_cache"),
    ("src.market_data.services.futures_collection", "DataCollector._save_indicator_kline_cache"),
    ("src.market_data.services.futures_collection", "DataCollector._save_mark_price_coverage"),
    ("src.market_data.services.futures_collection", "DataCollector._save_metrics_cache"),
    ("src.market_data.services.futures_collection", "DataCollector._validate_bookdepth_frame"),
    ("src.market_data.services.futures_collection", "DataCollector._validate_metrics_frame"),
    ("src.market_data.services.futures_collection", "DataCollector.ensure_bookdepth_data"),
    ("src.market_data.services.futures_collection", "DataCollector.ensure_indicator_kline_data"),
    ("src.market_data.services.futures_collection", "DataCollector.ensure_mark_price_data"),
    ("src.market_data.services.futures_collection", "DataCollector.ensure_metrics_data"),
    ("src.market_data.services.futures_collection", "DataCollector.ensure_metrics_live_tail"),
    ("src.market_data.services.futures_collection", "DataCollector.load_mark_price_panel"),
    ("src.market_data.services.futures_collection", "DataValidator"),
    ("src.market_data.services.futures_collection", "MarkPriceCoverage"),
    ("src.market_data.services.futures_collection", "_BOOKDEPTH_CANONICAL_COLUMNS"),
    ("src.market_data.services.futures_collection", "_INDICATOR_KLINE_CANONICAL_COLUMNS"),
    ("src.market_data.services.futures_collection", "_METRICS_CANONICAL_COLUMNS"),
    ("src.market_data.services.futures_collection", "_METRICS_MERGE_TOLERANCE"),
    ("src.market_data.services.futures_collection", "_METRICS_NUMERIC_COLUMNS"),
    ("src.market_data.services.futures_collection", "_METRICS_RELEASE_LAG"),
    ("src.market_data.services.futures_collection", "_empty_bookdepth_frame"),
    ("src.market_data.services.futures_collection", "_empty_metrics_frame"),
    ("src.market_data.services.futures_collection", "_mark_price_manifest_path"),
    ("src.market_data.services.futures_collection", "_mark_price_path"),
    ("src.market_data.services.mhs_execution", "_MARK_AVAILABILITY_LAG_HOURS"),
    ("src.market_data.services.mhs_execution", "_apply_causal_gap_exclusion"),
    ("src.market_data.services.mhs_execution", "_mark_availability_index"),
    ("src.market_data.services.mhs_execution", "_mark_covers_grid"),
    ("src.market_data.services.mhs_execution", "_read_mark_labels"),
    ("src.market_data.services.mhs_execution", "_row_group_datetime_min_ns"),
    ("src.market_data.services.mhs_execution", "apply_dynamic_mark_gap_exclusion"),
    ("src.market_data.services.mhs_execution", "assert_execution_data_coverage"),
    ("src.market_data.services.mhs_execution", "assert_relevant_mark_price_coverage"),
    ("src.market_data.services.mhs_execution", "refresh_mhs_execution_manifest"),
    ("src.market_data.streams.liquidations", "BinanceForceOrderFeed"),
    ("src.market_data.streams.liquidations", "_normalize_symbol"),
    ("src.market_data.streams.liquidations", "load_liquidation_events"),
    ("src.market_data.streams.liquidations", "run_liquidation_stream"),
    ("src.mhs.evaluation.integrity", "_assert_cache_required_marks"),
    ("src.mhs.marks", "_cached_mark_panel"),
    ("src.mhs.marks", "_compact_mark_series_for_path"),
    ("src.mhs.marks", "_contemporaneous_mark_close_panel"),
    ("src.mhs.marks", "_fill_mark_parity_eligibility"),
    ("src.mhs.marks", "_get_symbol_mark_frame"),
    ("src.mhs.params", "FOLD_GROWTH_CONCENTRATION_MAX_SHARE"),
    ("src.mhs.report.persist", "mhs_horizon_diagnostic_report_path"),
    ("src.quant.technical_experts.cross_sectional", "_BARS_PER_YEAR"),
    ("src.quant.technical_experts.cross_sectional", "apply_no_trade_band"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_neutral_weights"),
    ("src.quant.technical_experts.cross_sectional", "XsAlphaCompositeSpec"),
    ("src.quant.technical_experts.cross_sectional", "_validate_alpha_panels"),
    ("src.quant.technical_experts.cross_sectional", "_cross_sectional_zscore"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_family_scores"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_composite_score"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_weights"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_family_weights"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_dual_family_weights"),
    ("src.quant.technical_experts.cross_sectional", "_causal_family_inverse_vol_weights"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_vol_weighted_weights"),
    ("src.quant.technical_experts.cross_sectional", "_positioning_score"),
    ("src.quant.technical_experts.cross_sectional", "_basis_score"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_positioning_weights"),
    ("src.quant.technical_experts.cross_sectional", "build_xs_alpha_positioning_only_weights"),
    ("src.quant.technical_experts.cross_sectional", "_true_realized_net"),
    ("src.quant.technical_experts.cross_sectional", "XsAdmissionConfig"),
    ("src.quant.technical_experts.cross_sectional", "XsAdmissionResult"),
    ("src.quant.technical_experts.cross_sectional", "XsReliabilityResult"),
    ("src.quant.technical_experts.cross_sectional", "_annualized_sharpe"),
    ("src.quant.technical_experts.cross_sectional", "_realized_beta"),
    ("src.quant.technical_experts.cross_sectional", "evaluate_xs_admission"),
    ("src.quant.technical_experts.cross_sectional", "evaluate_xs_reliability"),
    ("src.quant.technical_experts.cross_sectional", "size_xs_alpha_growth_optimal"),
    ("src.quant.technical_experts.cross_sectional", "select_vol_target_window"),
    ("src.quant.evaluation.reliability", "_logger"),
    ("src.quant.evaluation.reliability", "_SECONDS_PER_YEAR"),
    ("src.quant.evaluation.reliability", "ReliabilityGateConfig"),
    ("src.quant.evaluation.reliability", "ReliabilityGateResult"),
    ("src.quant.evaluation.reliability", "FoldDistributionResult"),
    ("src.quant.evaluation.reliability", "HoldoutSegment"),
    ("src.quant.evaluation.reliability", "block_size_search_hit_cap"),
    ("src.quant.evaluation.reliability", "_block_bootstrap_cagr"),
    ("src.quant.evaluation.reliability", "compute_reliability_gate"),
    ("src.quant.evaluation.reliability", "equity_span_years"),
    ("src.quant.evaluation.reliability", "count_closed_trades"),
    ("src.quant.evaluation.reliability", "derive_cost_multiple_hurdle_rate"),
    ("src.quant.evaluation.reliability", "derive_realized_weights_cost_total"),
    ("src.quant.evaluation.reliability", "compute_equity_reliability_gate"),
    ("src.quant.evaluation.reliability", "compute_portfolio_reliability_gate"),
    ("src.quant.evaluation.reliability", "derive_fold_concentration_threshold"),
    ("src.quant.evaluation.reliability", "_year_log_return_contributions"),
    ("src.quant.evaluation.reliability", "split_holdout_segment"),
    ("src.quant.evaluation.reliability", "_fold_equity_metrics"),
    ("src.quant.evaluation.reliability", "compute_fold_distribution"),
    ("src.quant.evaluation.reliability", "_equal_duration_fold_labels"),
    ("src.quant.evaluation.reliability", "compute_equal_duration_fold_distribution"),
    ("src.quant.evaluation.reliability", "compute_turnover_fold_upper_bound"),
    ("src.quant.evaluation.reliability", "compute_stress_test_gate"),
    ("src.quant.evaluation.reliability", "_check_contract"),
    ("src.quant.risk.growth_sizing", "apply_realised_risk_overlay"),
    ("src.quant.risk.growth_sizing", "compute_discovery_target_vol"),
    ("src.quant.risk.growth_sizing", "apply_vol_target_overlay"),
    ("src.quant.universe.pit_universe", "PitUniverseSpec"),
    ("src.quant.universe.pit_universe", "SymbolCoverage"),
    ("src.quant.universe.pit_universe", "_validate_rebalance_dates"),
    ("src.quant.universe.pit_universe", "_eligible_at"),
    ("src.quant.universe.pit_universe", "_eligible_count"),
    ("src.quant.universe.pit_universe", "earliest_admissible_start"),
    ("src.quant.universe.pit_universe", "build_universe_schedule"),
    ("src.quant.universe.pit_universe", "derive_backfill_candidates"),
    ("src.market_data.storage.loaders", "_logger"),
    ("src.market_data.storage.loaders", "RESEARCH_TIMEFRAMES"),
    ("src.market_data.storage.loaders", "_TIMEFRAME_BARS"),
    ("src.market_data.storage.loaders", "_TIMEFRAME_RULE"),
    ("src.market_data.storage.loaders", "validate_timeframe"),
    ("src.market_data.storage.loaders", "timeframe_period"),
    ("src.market_data.storage.loaders", "timeframe_scale_factor"),
    ("src.market_data.storage.loaders", "_taker_buy_quote_series"),
    ("src.market_data.storage.loaders", "load_ohlcv_1h_as"),
    ("src.market_data.storage.loaders", "load_ohlcv_1h_as_4h"),
    ("src.market_data.storage.loaders", "load_ohlcv_4h"),
    ("src.mhs.pipeline.stages.committee", "QUALIFICATION_END"),
)
RETIRED_LITERALS: Final[tuple[tuple[str, frozenset[str]], ...]] = (
    ("markPriceKlines", frozenset({"src/application/ops/gdrive_cleanup.py"})),
)
CROSS_PACKAGE_PRIVATE_ALLOWLIST: Final[frozenset[tuple[str, str, str]]] = frozenset(
    {
        ("src/application/mhs_frozen_account.py", "src.mhs.resources", "_current_tree_swap_bytes"),
        ("src/backtests/migration.py", "src.mhs.run_history", "_sparse_identity_key"),
        ("src/cli/commands/live.py", "src.live.liveness", "_default_state_path"),
        ("src/cli/commands/live.py", "src.live.scheduler", "_default_frozen_step"),
        ("src/cli/commands/live.py", "src.live.scheduler", "_resolve_heartbeat_path"),
        (
            "src/market_data/services/source_gap_audit.py",
            "src.mhs.source_gaps",
            "_default_registry_path",
        ),
        (
            "src/market_data/services/source_gap_audit.py",
            "src.mhs.source_gaps",
            "_parse_registry_bytes",
        ),
    }
)
DAEMON_IMPORT_CHAIN: Final[tuple[str, ...]] = (
    "src.cli.main",
    "src.live.scheduler",
    "src.mhs.execution.pnl",
)
DAEMON_IMPORT_TIMEOUT_S: Final[int] = 120


def _iter_python_files(root: str) -> Iterator[Path]:
    for path in sorted((REPO_ROOT / root).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _resolve_import_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    parts = list(rel.with_suffix("").parts)
    package = ".".join(parts[:-1])
    return importlib.util.resolve_name(
        "." * node.level + (node.module or ""), package
    )


def _top_level_package(module_or_relpath: str) -> str:
    if "/" in module_or_relpath or module_or_relpath.endswith(".py"):
        rel = module_or_relpath.removeprefix("./")
        if rel.startswith("src/"):
            rest = rel.removeprefix("src/")
            if "/" not in rest:
                return "<root>"
            return rest.split("/", 1)[0]
        return "<root>"
    parts = module_or_relpath.split(".")
    if len(parts) >= 2 and parts[0] == "src":
        return parts[1]
    return "<root>"


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_deleted_match(reference: str, deleted_path: str) -> bool:
    return reference == deleted_path or reference.startswith(deleted_path + ".")


def _deleted_reference_hits(path: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    rel = path.relative_to(REPO_ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_module(path, node)
            if resolved:
                candidates.append(resolved)
                candidates.extend(
                    f"{resolved}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidates.append(node.value)
        for candidate in candidates:
            for deleted in DELETED_MODULES:
                if _is_deleted_match(candidate, deleted):
                    hits.append((rel, node.lineno, deleted))
                    break
    return hits


def _collect_cross_package_private_imports() -> dict[tuple[str, str, str], list[int]]:
    observed: dict[tuple[str, str, str], list[int]] = {}
    for path in _iter_python_files("src"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        importer_top = _top_level_package(rel)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = _resolve_import_module(path, node)
                if not target.startswith("src."):
                    continue
                if _top_level_package(target) == importer_top:
                    continue
                module_private = any(
                    _is_private(segment) for segment in target.split(".")[1:]
                )
                if module_private:
                    observed.setdefault((rel, target, "*"), []).append(node.lineno)
                for alias in node.names:
                    if alias.name != "*" and _is_private(alias.name):
                        observed.setdefault((rel, target, alias.name), []).append(
                            node.lineno
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if not target.startswith("src."):
                        continue
                    if _top_level_package(target) == importer_top:
                        continue
                    if any(
                        _is_private(segment) for segment in target.split(".")[1:]
                    ):
                        observed.setdefault((rel, target, "*"), []).append(node.lineno)
    return observed


@pytest.mark.parametrize("module", DELETED_MODULES, ids=DELETED_MODULES)
def test_deleted_modules_absent_on_disk(module: str) -> None:
    rel = module.replace(".", "/")
    offenders = [
        candidate
        for candidate in (REPO_ROOT / f"{rel}.py", REPO_ROOT / rel)
        if candidate.exists()
    ]
    assert offenders == [], f"deleted module reappeared: {module} -> {offenders}"


def test_deleted_modules_unreferenced_from_src_and_tools() -> None:
    offenders: list[str] = []
    for root in SCANNED_ROOTS:
        for path in _iter_python_files(root):
            for rel, lineno, deleted in _deleted_reference_hits(path):
                offenders.append(f"{rel}:{lineno} -> {deleted}")
    assert sorted(offenders) == [], (
        f"references to deleted modules from {SCANNED_ROOTS}: {sorted(offenders)}"
    )


def test_deleted_module_matcher_is_boundary_aware() -> None:
    assert not _is_deleted_match("src.research_foo", "src.research")
    assert _is_deleted_match("src.research.sub", "src.research")


@pytest.mark.parametrize(
    ("relpath", "statement", "expected"),
    [
        ("src/mhs/execution/__init__.py", "from .accumulator import x", "src.mhs.execution.accumulator"),
        ("src/mhs/execution/pnl.py", "from ..marks import x", "src.mhs.marks"),
        ("tools/checks/__init__.py", "from .helpers import x", "tools.checks.helpers"),
    ],
)
def test_relative_imports_resolve_against_importer_package(
    relpath: str, statement: str, expected: str
) -> None:
    node = ast.parse(statement).body[0]
    assert isinstance(node, ast.ImportFrom)
    assert _resolve_import_module(REPO_ROOT / relpath, node) == expected


@pytest.mark.parametrize(
    "owner_attr",
    RETIRED_SYMBOLS,
    ids=[f"{owner}:{attr}" for owner, attr in RETIRED_SYMBOLS],
)
def test_retired_symbols_stay_absent(owner_attr: tuple[str, str]) -> None:
    owner, dotted = owner_attr
    try:
        module = importlib.import_module(owner)
    except ModuleNotFoundError:
        pytest.fail(
            f"owner module {owner!r} no longer imports; "
            "move the entry to DELETED_MODULES"
        )
    segments = dotted.split(".")
    obj = module
    for segment in segments[:-1]:
        try:
            obj = getattr(obj, segment)
        except AttributeError:
            return
    assert not hasattr(obj, segments[-1]), (
        f"retired symbol reappeared: {owner}:{dotted}"
    )


@pytest.mark.parametrize(
    ("literal", "allowed"),
    RETIRED_LITERALS,
    ids=[literal for literal, _ in RETIRED_LITERALS],
)
def test_retired_literals_confined_to_allowlist(
    literal: str, allowed: frozenset[str],
) -> None:
    hits: dict[str, list[int]] = {}
    for path in _iter_python_files("src"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            if literal in line:
                hits.setdefault(rel, []).append(lineno)
    offenders = sorted(
        f"{rel}:{lineno}"
        for rel, linenos in hits.items()
        if rel not in allowed
        for lineno in linenos
    )
    assert offenders == [], f"retired literal {literal!r} escaped allowlist: {offenders}"
    stale = sorted(rel for rel in allowed if rel not in hits)
    assert stale == [], f"stale literal allowance for {literal!r}: {stale}"


def test_no_new_cross_package_private_imports() -> None:
    observed = _collect_cross_package_private_imports()
    new = sorted(
        f"{rel}:{sorted(lines)} -> {module} :: {name}"
        for (rel, module, name), lines in observed.items()
        if (rel, module, name) not in CROSS_PACKAGE_PRIVATE_ALLOWLIST
    )
    assert new == [], f"new cross-package private imports: {new}"


def test_private_import_allowlist_is_shrink_only() -> None:
    observed = _collect_cross_package_private_imports()
    stale = sorted(
        f"{rel} -> {module} :: {name}"
        for (rel, module, name) in CROSS_PACKAGE_PRIVATE_ALLOWLIST
        if (rel, module, name) not in observed
    )
    assert stale == [], f"stale allowlist entries (remove from allowlist): {stale}"


def test_daemon_import_chain_imports_cleanly_in_fresh_interpreter() -> None:
    statement = "import " + ", ".join(DAEMON_IMPORT_CHAIN)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(REPO_ROOT),
    }
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", statement],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=DAEMON_IMPORT_TIMEOUT_S,
    )
    assert completed.returncode == 0, (
        f"daemon import chain failed (rc={completed.returncode}): {completed.stderr}"
    )
    assert "Traceback" not in completed.stderr, (
        f"daemon import chain raised: {completed.stderr}"
    )
