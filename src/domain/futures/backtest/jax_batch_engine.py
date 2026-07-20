"""JAX vmap GPU-accelerated batch backtest engine (Phase 1, gated).

[LIMIT-04] L2_JAX_BATCH_ENABLED=False → fallback to numba CPU path.

[ADR_20260720_HYBRID_COMPILATION_JAX_PHASE0_REJECTED] Phase 0 profiling gate
([LIMIT-01]) failed for both L1 (numba <2% of trial wall-clock) and L2 (no
dedicated numba kernel exists to port). Sealed at L2_JAX_BATCH_ENABLED=False;
not adopted. See docs/specs/hybrid_compilation_opt.md.
"""

# mypy: ignore-errors
# JAX lax.scan heavily uses dynamic tuple types that cannot be statically typed.
from __future__ import annotations

import logging

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array

_logger = logging.getLogger(__name__)


class JaxBatchUnavailableError(RuntimeError):
    """[LIMIT-03] GPU unavailable / OOM → caller should fall back to numba CPU path."""


def _init_state(n_syms: int, initial_balance: float) -> tuple:
    z = jnp.zeros(n_syms, dtype=jnp.float32)
    zi = jnp.zeros(n_syms, dtype=jnp.int32)
    return (
        jnp.float32(initial_balance),
        jnp.zeros(n_syms, dtype=jnp.bool_),
        jnp.zeros(n_syms, dtype=jnp.int8),
        z, zi, z,
        jnp.int32(0),
    )


def _single_backtest(
    target_weights: Array,
    atr_mult: Array,
    trail_mult: Array,
    close_2d: Array,
    high_2d: Array,
    low_2d: Array,
    open_2d: Array,
    funding_rate: Array,
    atr_2d: Array,
    initial_balance: float,
    maker_fee: float,
    taker_fee: float,
    slippage_rate: float,
    rebalance_bars: int,
    max_hold_bars: int,
) -> tuple[Array, Array, Array]:
    n_bars, n_syms = close_2d.shape
    rb = jnp.where(rebalance_bars > 0, rebalance_bars, 999_999_999)
    max_exposure_v = 10.0
    max_exp_per_coin_v = 100.0
    mc = 100

    bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count = _init_state(n_syms, initial_balance)
    equity_curve = jnp.zeros(n_bars, dtype=jnp.float32).at[0].set(jnp.float32(initial_balance))
    is_liq = jnp.bool_(False)

    def _bar_step(carry: tuple, i: int) -> tuple:
        bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count, equity_curve, is_liq = carry  # noqa: PLW2901

        prev_i = jnp.maximum(0, i - 1)
        op_i = open_2d[i]
        hi_i = high_2d[i]
        lo_i = low_2d[i]
        cl_i = close_2d[i]
        fr_i = funding_rate[i]
        atr_prev = atr_2d[prev_i]

        def _entry_exit(carry: tuple) -> tuple:
            bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count, _, _ = carry  # noqa: PLW2901

            tw = target_weights[i]
            tw = jnp.clip(tw, -1.0, 1.0)
            tw = jnp.where(jnp.isfinite(tw), tw, 0.0)
            gross = jnp.sum(jnp.abs(tw))
            tw = jnp.where(gross > max_exposure_v + 1e-12, tw * (max_exposure_v / gross), tw)

            # Sort by abs weight, keep top mc
            abs_tw = jnp.abs(tw)
            ranks = jnp.argsort(jnp.argsort(abs_tw))
            tw = jnp.where(ranks >= jnp.maximum(0, n_syms - mc), tw, 0.0)

            used_margin_bal = jnp.sum(jnp.where(in_pos, amount * entry_p / 1.0, 0.0))
            unreal_bal = jnp.sum(jnp.where(in_pos, (op_i - entry_p) * amount * pos_side.astype(jnp.float32), 0.0))
            eq_snap = bal + used_margin_bal + unreal_bal
            tgt_notional = eq_snap * tw
            ts = jnp.sign(tgt_notional)
            desired_amt = jnp.where(ts != 0, jnp.abs(tgt_notional) / op_i, 0.0)
            min_notional = jnp.maximum(0.01, eq_snap * 0.0001)
            need_exit = in_pos & ((ts == 0) | (ts != pos_side.astype(jnp.float32)) | (jnp.abs(amount - desired_amt) * op_i > min_notional))

            # Exit
            exit_price = op_i * (1.0 - slippage_rate * pos_side.astype(jnp.float32))
            pnl_e = (exit_price - entry_p) * amount * pos_side.astype(jnp.float32)
            fee_e = amount * exit_price * taker_fee
            ret_e = amount * entry_p / 1.0 + (pnl_e - fee_e)
            bal = bal + jnp.sum(jnp.where(need_exit, ret_e, 0.0))
            t_count = t_count + jnp.sum(need_exit.astype(jnp.int32))

            in_pos = jnp.where(need_exit, jnp.zeros_like(in_pos), in_pos)
            pos_side = jnp.where(need_exit, jnp.zeros_like(pos_side), pos_side)
            entry_p = jnp.where(need_exit, jnp.zeros_like(entry_p), entry_p)
            amount = jnp.where(need_exit, jnp.zeros_like(amount), amount)

            # After-exit equity
            used_after = jnp.sum(jnp.where(in_pos, amount * entry_p / 1.0, 0.0))
            unreal_after = jnp.sum(jnp.where(in_pos, (op_i - entry_p) * amount * pos_side.astype(jnp.float32), 0.0))
            eq_snap2 = bal + used_after + unreal_after

            # Enter new
            entry_mask = (~in_pos) & (ts != 0)
            fill_p = op_i * (1.0 + slippage_rate * ts.astype(jnp.float32))
            dt = jnp.where(entry_mask, jnp.abs(tgt_notional) / fill_p, 0.0)
            max_qty = jnp.where(max_exp_per_coin_v > 0.0, (eq_snap2 * max_exp_per_coin_v) / fill_p, jnp.inf)
            dt = jnp.where(entry_mask & (max_qty > 0.0), jnp.minimum(dt, max_qty), dt)
            dust_skip = (dt * fill_p) < jnp.maximum(0.01, eq_snap2 * 0.0001)
            free_margin = jnp.maximum(eq_snap2 - used_after, 0.0)
            max_qty_margin = jnp.maximum((free_margin * 0.97) / fill_p, 0.0)
            fq = jnp.minimum(dt, max_qty_margin)
            req_m = fq * fill_p
            margin_fail = (fq <= 1e-12) | (free_margin < req_m + fq * fill_p * taker_fee)
            enter_ok = entry_mask & ~dust_skip & ~margin_fail

            bal = bal + jnp.sum(jnp.where(enter_ok, -req_m - fq * fill_p * taker_fee, 0.0))
            t_count = t_count + jnp.sum(enter_ok.astype(jnp.int32))

            in_pos = jnp.where(enter_ok, jnp.ones_like(in_pos, dtype=jnp.bool_), in_pos)
            pos_side = jnp.where(enter_ok, ts.astype(jnp.int8), pos_side)
            entry_p = jnp.where(enter_ok, fill_p, entry_p)
            entry_idx = jnp.where(enter_ok, jnp.full_like(entry_idx, i, dtype=jnp.int32), entry_idx)
            amount = jnp.where(enter_ok, fq, amount)

            return (bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count, equity_curve, is_liq)

        carry = lax.cond(((i % rb) == 0) & ~is_liq, _entry_exit, lambda c: c, carry)
        bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count, _, _ = carry  # noqa: PLW2901

        # Funding fee
        fund_inc = jnp.where(jnp.isfinite(fr_i), amount * cl_i * fr_i * pos_side.astype(jnp.float32), 0.0)
        fund_inc = jnp.where(in_pos, fund_inc, 0.0)

        # Exit checks
        used_mtm = jnp.sum(jnp.where(in_pos, amount * cl_i / 1.0, 0.0))
        unreal_mtm = jnp.sum(jnp.where(in_pos, (cl_i - entry_p) * amount * pos_side.astype(jnp.float32), 0.0))
        current_equity = bal + used_mtm + unreal_mtm

        # Simple ATR stop (use_simple_atr=1)
        sl_hit_open_long = (pos_side == 1) & (op_i <= entry_p - atr_prev * atr_mult) & in_pos
        sl_hit_bar_long = (pos_side == 1) & (lo_i <= entry_p - atr_prev * atr_mult) & in_pos
        sl_hit_open_short = (pos_side == -1) & (op_i >= entry_p + atr_prev * atr_mult) & in_pos
        sl_hit_bar_short = (pos_side == -1) & (hi_i >= entry_p + atr_prev * atr_mult) & in_pos
        sl_hit = sl_hit_open_long | sl_hit_bar_long | sl_hit_open_short | sl_hit_bar_short

        # Max hold exit
        hold_exit = in_pos & (max_hold_bars > 0) & ((i - entry_idx.astype(jnp.int32)) >= max_hold_bars)

        # Kill signal (none)
        kill_exit = jnp.zeros(n_syms, dtype=jnp.bool_)

        exit_trigger = sl_hit | hold_exit | kill_exit

        # Exit price
        sl_exit_px = jnp.where(
            pos_side == 1,
            jnp.where(sl_hit_open_long, op_i * (1.0 - slippage_rate), (entry_p - atr_prev * atr_mult) * (1.0 - slippage_rate)),
            jnp.where(sl_hit_open_short, op_i * (1.0 + slippage_rate), (entry_p + atr_prev * atr_mult) * (1.0 + slippage_rate)),
        )
        hold_exit_px = op_i * (1.0 - slippage_rate * pos_side.astype(jnp.float32))
        exit_px = jnp.where(sl_hit | hold_exit, jnp.where(sl_hit, sl_exit_px, hold_exit_px), 0.0)

        # Process exits (vectorized)
        pnl_x = (exit_px - entry_p) * amount * pos_side.astype(jnp.float32)
        fee_x = amount * exit_px * taker_fee
        ret_x = (amount * entry_p) / 1.0 + (pnl_x - fee_x - fund_inc)
        bal = bal + jnp.sum(jnp.where(exit_trigger, ret_x, 0.0))
        t_count = t_count + jnp.sum(exit_trigger.astype(jnp.int32))

        in_pos = jnp.where(exit_trigger, jnp.zeros_like(in_pos), in_pos)
        pos_side = jnp.where(exit_trigger, jnp.zeros_like(pos_side), pos_side)
        entry_p = jnp.where(exit_trigger, jnp.zeros_like(entry_p), entry_p)
        amount = jnp.where(exit_trigger, jnp.zeros_like(amount), amount)

        # Liquidation guard
        def _liq_all(carry: tuple) -> tuple:
            _, _, _, _, _, _, t_count, _, _ = carry
            exit_px_liq = jnp.where(
                pos_side.astype(jnp.float32) != 0,
                cl_i * (1.0 - slippage_rate * pos_side.astype(jnp.float32)),
                cl_i,
            )
            pnl_liq = (exit_px_liq - entry_p) * amount * pos_side.astype(jnp.float32)
            fee_liq = amount * exit_px_liq * taker_fee
            ret_liq = amount * entry_p / 1.0 + (pnl_liq - fee_liq - fund_inc)
            new_bal = jnp.sum(jnp.where(in_pos, ret_liq, 0.0))
            tc = t_count + jnp.sum(in_pos.astype(jnp.int32))
            return (new_bal,
                    jnp.zeros_like(in_pos), jnp.zeros_like(pos_side), jnp.zeros_like(entry_p),
                    entry_idx, jnp.zeros_like(amount), tc,
                    equity_curve, jnp.bool_(True))

        carry = lax.cond((current_equity <= 0.0) & ~is_liq, _liq_all, lambda c: c, carry)
        bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count, _, is_liq = carry  # noqa: PLW2901

        eq_for_curve = jnp.where(is_liq, 0.0, jnp.maximum(current_equity, 0.0))
        equity_curve = equity_curve.at[i].set(eq_for_curve)

        return (bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count, equity_curve, is_liq), None

    init = (bal, in_pos, pos_side, entry_p, entry_idx, amount, t_count, equity_curve, is_liq)
    (bal, in_pos, pos_side, entry_p, _, amount, t_count, equity_curve, _), _ = lax.scan(_bar_step, init, jnp.arange(1, n_bars))

    # Final closeout (no slippage, matching numba behavior)
    last_cl = close_2d[n_bars - 1]
    pnl_last = (last_cl - entry_p) * amount * pos_side.astype(jnp.float32)
    fee_last = amount * last_cl * taker_fee
    ret_last = (amount * entry_p) / 1.0 + (pnl_last - fee_last)

    def _closeout(bal: Array) -> Array:
        return bal + jnp.sum(jnp.where(in_pos, ret_last, 0.0))

    final_balance = lax.cond(jnp.any(in_pos), _closeout, lambda b: b, bal)
    return equity_curve, jnp.array([final_balance], dtype=jnp.float32), jnp.array([t_count], dtype=jnp.int32).reshape(1)


# ---------------------------------------------------------------------------
# Public batch API
# ---------------------------------------------------------------------------


def simulate_batch_target_weights_jax(
    close_2d: Array,
    high_2d: Array,
    low_2d: Array,
    open_2d: Array,
    funding_rate: Array,
    target_weights_batch: Array,
    initial_balance: float,
    maker_fee: float,
    taker_fee: float,
    slippage_rate: float,
    rebalance_bars: int,
    max_hold_bars: int,
    atr_2d: Array,
    atr_mult_batch: Array,
    trail_mult_batch: Array,
    max_vram_gb: float = 8.5,
) -> tuple[Array, Array, Array]:
    try:
        devices = jax.devices()
        if not devices:
            raise RuntimeError("No JAX devices available")
    except Exception as exc:
        raise JaxBatchUnavailableError(f"JAX GPU unavailable: {exc}") from exc

    def _to_jax(x: Array) -> Array:
        if not isinstance(x, Array):
            return jnp.array(x, dtype=jnp.float32)
        return x.astype(jnp.float32) if x.dtype != jnp.float32 else x

    close_2d = _to_jax(close_2d)
    high_2d = _to_jax(high_2d)
    low_2d = _to_jax(low_2d)
    open_2d = _to_jax(open_2d)
    funding_rate = _to_jax(funding_rate)
    target_weights_batch = _to_jax(target_weights_batch)
    atr_2d = _to_jax(atr_2d)
    if not isinstance(atr_mult_batch, Array):
        atr_mult_batch = jnp.array(atr_mult_batch, dtype=jnp.float32)
    if not isinstance(trail_mult_batch, Array):
        trail_mult_batch = jnp.array(trail_mult_batch, dtype=jnp.float32)

    _ = max_vram_gb
    _ = maker_fee  # reserved for future limit-order support

    vmap_func = jax.vmap(
        _single_backtest,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
    )

    try:
        equity_curves, final_balances, trade_counts = vmap_func(
            target_weights_batch, atr_mult_batch, trail_mult_batch,
            close_2d, high_2d, low_2d, open_2d, funding_rate, atr_2d,
            initial_balance, maker_fee, taker_fee, slippage_rate,
            rebalance_bars, max_hold_bars,
        )
    except Exception as exc:
        raise JaxBatchUnavailableError(f"JAX batch execution failed (OOM?): {exc}") from exc

    return equity_curves, final_balances[:, 0], trade_counts[:, 0]
