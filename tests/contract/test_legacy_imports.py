from __future__ import annotations

import ast
import importlib
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


def test_application_facades_reexport_canonical_evaluation_functions() -> None:
    """Legacy application module names resolve to the canonical evaluation objects."""
    from src.application.admission_backtest import (
        run_technical_library_admission_backtest as canonical_admission_backtest,
    )
    from src.application.expert_evaluation import (
        run_technical_expert_evaluation as canonical_expert_evaluation,
    )
    from src.application.expert_portfolio_evaluation import (
        run_expert_portfolio_evaluation as legacy_expert_portfolio,
    )
    from src.application.library_admission import (
        run_technical_library_admission as canonical_library_admission,
    )
    from src.application.library_evaluation import (
        run_expert_portfolio_evaluation as canonical_expert_portfolio,
    )
    from src.application.technical_expert_evaluation import (
        run_technical_expert_evaluation as legacy_expert_evaluation,
    )
    from src.application.technical_library_admission import (
        run_technical_library_admission as legacy_library_admission,
    )
    from src.application.technical_library_admission_backtest import (
        run_technical_library_admission_backtest as legacy_admission_backtest,
    )

    assert legacy_expert_portfolio is canonical_expert_portfolio
    assert legacy_expert_evaluation is canonical_expert_evaluation
    assert legacy_library_admission is canonical_library_admission
    assert legacy_admission_backtest is canonical_admission_backtest


_APPLICATION_FACADE_TARGETS = {
    "collection.py": "src.application.data.collection",
    "baseline_evaluation.py": "src.application.research.baseline.evaluation",
    "portfolio_evaluation.py": "src.application.research.portfolio.evaluation",
    "cash_carry_evaluation.py": "src.application.research.cash_carry.evaluation",
    "sleeve_blend_evaluation.py": "src.application.research.sleeve_blend.evaluation",
    "oi_deleveraging_evaluation.py": "src.application.research.oi_deleveraging.evaluation",
    "expert_evaluation.py": "src.application.research.technical_experts.evaluation",
    "library_evaluation.py": "src.application.research.expert_portfolio.evaluation",
    "library_admission.py": "src.application.research.expert_portfolio.admission",
    "admission_backtest.py": "src.application.research.expert_portfolio.admission_backtest",
    "technical_expert_evaluation.py": "src.application.research.technical_experts.evaluation",
    "expert_portfolio_evaluation.py": "src.application.research.expert_portfolio.evaluation",
    "technical_library_admission.py": "src.application.research.expert_portfolio.admission",
    "technical_library_admission_backtest.py": (
        "src.application.research.expert_portfolio.admission_backtest"
    ),
}
_CANONICAL_APPLICATION_PREFIXES = ("src.application.data.", "src.application.research.")


def _iter_flat_application_facades() -> list[Path]:
    return sorted(
        path for path in Path("src/application").glob("*.py") if path.name != "__init__.py"
    )


def _facade_exports(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                assert isinstance(node.value, (ast.List, ast.Tuple))
                return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
    return []


def test_flat_application_facade_map_is_complete() -> None:
    """RF-COMPAT-01: the facade map covers every flat application module."""
    flat = {path.name for path in _iter_flat_application_facades()}
    assert flat == set(_APPLICATION_FACADE_TARGETS), (
        "facade map drift: "
        f"unmapped={sorted(flat - set(_APPLICATION_FACADE_TARGETS))}, "
        f"stale={sorted(set(_APPLICATION_FACADE_TARGETS) - flat)}"
    )


def test_flat_application_modules_are_import_only_facades() -> None:
    """RF-COMPAT-01: a facade imports only its canonical target(s) and ``__all__``."""
    for path in _iter_flat_application_facades():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        exports = _facade_exports(tree)
        assert exports, f"{path}: facade must define a non-empty __all__"
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # module docstring
            if isinstance(node, ast.Import):
                raise AssertionError(f"{path}: facade must not use `import`")
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.ImportFrom):
                assert node.module is not None, f"{path}: facade uses a relative import"
                assert node.module.startswith(_CANONICAL_APPLICATION_PREFIXES), (
                    f"{path}: facade imports non-canonical {node.module}"
                )
                continue
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                continue
            raise AssertionError(f"{path}: facade body contains a non-import statement")


def test_flat_application_facades_reexport_canonical_objects() -> None:
    """RF-COMPAT-01: every facade export is the identical canonical object."""
    for filename, canonical in _APPLICATION_FACADE_TARGETS.items():
        facade = importlib.import_module(f"src.application.{Path(filename).stem}")
        canonical_module = importlib.import_module(canonical)
        for name in facade.__all__:
            assert getattr(facade, name) is getattr(canonical_module, name), (
                f"{facade.__name__}.{name} is not canonical {canonical_module.__name__}.{name}"
            )
