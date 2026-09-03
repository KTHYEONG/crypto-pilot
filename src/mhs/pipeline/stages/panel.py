"""S1: Panel loading + funding alignment + eligibility scaffolding.

Extracted verbatim from ``evaluation.py`` lines 3532-3610 (load_base_panel
call, funding series load/align, the base_1h_panel and funding_alignment
``telemetry.record`` + ``[DATA]`` ``debug_log.log`` calls, and the
``guards._guard_stage_or_breach`` calls at both checkpoints).

Byte-identity (I-IDENTITY-v2): every local in the source block is threaded
through ``ctx``; ``request`` -> ``ctx.config``, ``debug_log`` -> the
``telemetry`` (StageTelemetry) parameter, and the original ``telemetry``
(``_StageRecorder``) -> ``ctx.recorder``.  No branching added.
"""

from __future__ import annotations

from src.mhs.evaluation import (
    FUTURES_DATA_DIR,
    SOURCE_GAP_EXCLUDED_SYMBOLS,
    bar_funding_panel,
    guards,
    load_base_panel,
)
from src.mhs.marks import _load_funding_series
from src.mhs.pipeline.context import PipelineContext
from src.mhs.resources import _resolve_ram_budget, _StageRecorder
from src.mhs.telemetry import StageTelemetry, Tag


def load_panel(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Load the 1h base panel, align funding, and guard both checkpoints."""
    ctx.rss_budget_bytes, ctx.rss_reserve_bytes = _resolve_ram_budget(
        ctx.config.max_rss_bytes, ctx.config.ram_guard,
    )
    # Structured [TAG] debug logging (I-OBSERVE: additive only, never read back
    # into computed values). Distinct from `ctx.recorder` (report resource_measurements).
    # `telemetry` is the StageTelemetry passed in; the _StageRecorder is `ctx.recorder`.
    ctx.recorder = _StageRecorder(log_run=ctx.config.log_run)

    ctx.root = ctx.config.data_root or str(FUTURES_DATA_DIR / "ohlcv")
    panel = load_base_panel(
        ctx.root, "1h",
        (
            ("close", "open", "quote_vol", "taker_buy_quote")
            if ctx.config.committee_capital
            else ("close", "open", "quote_vol")
        ),
        ctx.start, ctx.end, partition="dev", min_bars=2000,
    )
    ctx.close, ctx.opens, ctx.quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    ctx.taker_buy_quote = panel["taker_buy_quote"] if ctx.config.committee_capital else None
    ctx.grid_1h = ctx.close.index
    ctx.symbols = list(ctx.close.columns)
    ctx.recorder.record("base_1h_panel", grid_bars=len(ctx.grid_1h), n_symbols=len(ctx.symbols))
    telemetry.log(
        Tag.DATA, "base_1h_panel",
        grid_start=str(ctx.grid_1h[0]), grid_end=str(ctx.grid_1h[-1]),
        symbols=sorted(ctx.symbols),
    )
    _terminal = guards._guard_stage_or_breach(
        "base_1h_panel", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
        ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
    )
    if _terminal is not None:
        ctx._terminal_report = _terminal
        return

    ctx.funding_by_symbol, ctx.funding_dropped = _load_funding_series(ctx.symbols)
    ctx.fold_funding = dict(ctx.funding_by_symbol)
    ctx.funded = [
        s for s in ctx.symbols
        if s in ctx.funding_by_symbol and s not in SOURCE_GAP_EXCLUDED_SYMBOLS
    ]
    if not ctx.funded:
        raise RuntimeError("no dev symbol has funding coverage; the MHS ledger requires funding")
    ctx.close = ctx.close[ctx.funded]
    ctx.opens = ctx.opens[ctx.funded]
    ctx.quote_vol = ctx.quote_vol[ctx.funded]
    if ctx.taker_buy_quote is not None:
        ctx.taker_buy_quote = ctx.taker_buy_quote[ctx.funded]
    ctx.bar_period = ctx.grid_1h[1] - ctx.grid_1h[0]
    ctx.funding_window = {
        s: ctx.funding_by_symbol[s].loc[
            (ctx.funding_by_symbol[s].index >= ctx.grid_1h[0])
            & (ctx.funding_by_symbol[s].index < ctx.grid_1h[-1] + ctx.bar_period)
        ]
        for s in ctx.funded
    }
    ctx.bar_funding = bar_funding_panel(ctx.funding_window, ctx.grid_1h)
    ctx.aligned_symbols = list(ctx.bar_funding.columns)
    if not ctx.aligned_symbols:
        raise RuntimeError("no dev symbol has causally aligned funding coverage")
    ctx.close = ctx.close[ctx.aligned_symbols]
    ctx.opens = ctx.opens[ctx.aligned_symbols]
    ctx.quote_vol = ctx.quote_vol[ctx.aligned_symbols]
    if ctx.taker_buy_quote is not None:
        ctx.taker_buy_quote = ctx.taker_buy_quote[ctx.aligned_symbols]
    ctx.funding_by_symbol = {s: ctx.funding_by_symbol[s] for s in ctx.aligned_symbols}
    ctx.bar_funding = ctx.bar_funding[ctx.aligned_symbols]
    ctx.recorder.record("funding_alignment", grid_bars=len(ctx.grid_1h), n_symbols=len(ctx.aligned_symbols))
    telemetry.log(
        Tag.DATA, "funding_alignment",
        aligned_symbols=len(ctx.aligned_symbols), dropped=sorted(ctx.funding_dropped),
    )
    _terminal = guards._guard_stage_or_breach(
        "funding_alignment", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
        ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
    )
    if _terminal is not None:
        ctx._terminal_report = _terminal
        return
