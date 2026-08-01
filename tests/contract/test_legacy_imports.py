from __future__ import annotations

import ast
from pathlib import Path

"""Compatibility contract tests for the src/ -> feature-canonical refactor.

RF-COMPAT-01: every existing public ``src.core`` / ``src.data`` /
``src.engine`` / ``src.strategy`` / ``src.validation`` symbol resolves to its
canonical implementation, e.g. ``src.engine.backtest.run_backtest`` must be the
same object as ``src.research.baseline.backtest.run_backtest``. No production
canonical module imports a compatibility façade.
"""


def legacy_import_contract() -> None:
    """Execute the frozen legacy-import compatibility contract."""
    test_legacy_imports_reexport_canonical_symbols()
    test_no_canonical_module_imports_a_facade()



def test_legacy_imports_reexport_canonical_symbols() -> None:
    """RF-COMPAT-01: every legacy facade resolves to the canonical object."""
    from src.engine.backtest import run_backtest as legacy_run_backtest
    from src.research.baseline.backtest import run_backtest as canonical_run_backtest

    assert legacy_run_backtest is canonical_run_backtest

    from src.core.config import borrow_path, funding_path, ohlcv_path, spot_ohlcv_path
    from src.common.config import borrow_path as c_borrow, funding_path as c_funding
    from src.common.config import ohlcv_path as c_ohlcv, spot_ohlcv_path as c_spot

    assert borrow_path is c_borrow
    assert funding_path is c_funding
    assert ohlcv_path is c_ohlcv
    assert spot_ohlcv_path is c_spot

    from src.core.logging_setup import setup_logger as legacy_setup_logger
    from src.common.logging import setup_logger as canonical_setup_logger

    assert legacy_setup_logger is canonical_setup_logger

    from src.core.types import CostModel, StrategySpec
    from src.research.contracts import CostModel as C, StrategySpec as S

    assert CostModel is C
    assert StrategySpec is S

    from src.core.types import CarryCostModel, CashCarrySpec
    from src.research.cash_carry.contracts import CarryCostModel as CC, CashCarrySpec as CS

    assert CarryCostModel is CC
    assert CashCarrySpec is CS

    from src.data.loader import DataIntegrityError as legacy_error
    from src.common.errors import DataIntegrityError as canonical_error

    assert legacy_error is canonical_error

    from src.data.loader import load_ohlcv_4h as legacy_load
    from src.market_data.storage.loaders import load_ohlcv_4h as canonical_load

    assert legacy_load is canonical_load

    from src.data.ohlcv_store import write_ohlcv as legacy_write
    from src.market_data.storage.ohlcv import write_ohlcv as canonical_write

    assert legacy_write is canonical_write

    from src.data.binance import BinanceSpotClient as legacy_spot
    from src.market_data.binance.spot import BinanceSpotClient as canonical_spot

    assert legacy_spot is canonical_spot

    from src.data.carry_data import CarryMarketData, load_carry_market_data
    from src.research.cash_carry.contracts import CarryMarketData as CM
    from src.research.cash_carry.market_data import load_carry_market_data as clmd

    assert CarryMarketData is CM
    assert load_carry_market_data is clmd

    from src.strategy.donchian import generate_signals as legacy_signals
    from src.research.baseline.signal import generate_signals as canonical_signals

    assert legacy_signals is canonical_signals

    from src.strategy.cash_carry import generate_cash_carry_target as legacy_target
    from src.research.cash_carry.signal import generate_cash_carry_target as canonical_target

    assert legacy_target is canonical_target

    from src.engine.portfolio_backtest import run_portfolio_backtest as legacy_portfolio
    from src.research.portfolio.backtest import run_portfolio_backtest as canonical_portfolio

    assert legacy_portfolio is canonical_portfolio

    from src.engine.cash_carry_backtest import run_cash_carry_backtest as legacy_carry
    from src.research.cash_carry.backtest import run_cash_carry_backtest as canonical_carry

    assert legacy_carry is canonical_carry

    from src.engine.results_log import record_run as legacy_record
    from src.research.provenance.results import record_run as canonical_record

    assert legacy_record is canonical_record

    from src.validation.metrics import compute_metrics as legacy_metrics
    from src.research.evaluation.metrics import compute_metrics as canonical_metrics

    assert legacy_metrics is canonical_metrics

    from src.validation.reliability_gate import ReliabilityGateResult as legacy_gate
    from src.research.evaluation.reliability import ReliabilityGateResult as canonical_gate

    assert legacy_gate is canonical_gate

    from src.validation.candidate_promotion import compose_promotion_verdict as legacy_promo
    from src.research.evaluation.promotion import compose_promotion_verdict as canonical_promo

    assert legacy_promo is canonical_promo

    from src.validation.candidate_registry import register_candidate as legacy_register
    from src.research.provenance.candidates import register_candidate as canonical_register

    assert legacy_register is canonical_register

    from src.data.portfolio_universe import select_liquid_universe as legacy_universe
    from src.research.portfolio.universe import select_liquid_universe as canonical_universe

    assert legacy_universe is canonical_universe

    from src.data.collector import DataCollector as legacy_collector
    from src.market_data.services.futures_collection import DataCollector as canonical_collector

    assert legacy_collector is canonical_collector

    from src.data.spot_collector import SpotDataCollector as legacy_spot_collector
    from src.market_data.services.spot_collection import SpotDataCollector as canonical_spot_collector

    assert legacy_spot_collector is canonical_spot_collector

    from src.data.spot_collector import import_quote_borrow_history as legacy_borrow
    from src.market_data.services.borrow_collection import import_quote_borrow_history as canonical_borrow

    assert legacy_borrow is canonical_borrow

    from src.data.vision import BinanceVisionDownloader as legacy_vision
    from src.market_data.binance.vision import BinanceVisionDownloader as canonical_vision

    assert legacy_vision is canonical_vision

    from src.core.constants import TAKER_FEE_BPS as LEGACY_FEE
    from src.common.constants import TAKER_FEE_BPS as CANONICAL_FEE

    assert LEGACY_FEE is CANONICAL_FEE


_FACADE_ROOTS = ("src/core", "src/data", "src/engine", "src/strategy", "src/validation")
_CANONICAL_ROOTS = (
    "src/common",
    "src/market_data",
    "src/research",
    "src/application",
    "src/cli",
)


def _iter_py_files(root: str) -> list[Path]:
    return sorted(Path(root).rglob("*.py"))


def _imports_facade(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == root or alias.name.startswith(root + ".") for root in _FACADE_ROOTS):
                    return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and any(node.module == root or node.module.startswith(root + ".") for root in _FACADE_ROOTS)
        ):
            return True
    return False


def test_no_canonical_module_imports_a_facade() -> None:
    """RF-COMPAT-01: canonical production modules never import legacy facades."""
    offending: list[str] = []
    for canonical_root in _CANONICAL_ROOTS:
        for path in _iter_py_files(canonical_root):
            if path.name == "__init__.py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            if _imports_facade(tree):
                offending.append(str(path))
    assert not offending, f"canonical modules import façades: {offending}"
