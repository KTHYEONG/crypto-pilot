# ruff: noqa
def test_compute_funding_backfill_accrues_held_epochs() -> None:
    from decimal import Decimal

    import pandas as pd

    from src.live.funding_backfill import compute_funding_backfill
    from src.live.ledger import PositionSnapshot

    # Given: A 는 09-02 01:30~09-03 01:30 롱 2, 이후 B 숏 1
    t0 = pd.Timestamp("2026-09-02 01:30Z")
    t1 = pd.Timestamp("2026-09-03 01:30Z")
    end = pd.Timestamp("2026-09-03 20:00Z")
    history = (
        PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("2")}),
        PositionSnapshot(effective_from=t1, positions={"BBBUSDT": Decimal("-1")}),
    )
    a_epochs = pd.DatetimeIndex(["2026-09-02 08:00", "2026-09-02 16:00", "2026-09-03 00:00", "2026-09-03 08:00"], tz="UTC")
    b_epochs = pd.DatetimeIndex(["2026-09-03 08:00", "2026-09-03 16:00"], tz="UTC")
    funding = {
        "AAAUSDT": pd.Series([0.001, 0.001, 0.002, 0.005], index=a_epochs),
        "BBBUSDT": pd.Series([0.001, -0.002], index=b_epochs),
    }
    marks = {
        "AAAUSDT": pd.Series([100.0, 100.0, 100.0, 100.0], index=a_epochs),
        "BBBUSDT": pd.Series([50.0, 50.0], index=b_epochs),
    }

    # When
    result = compute_funding_backfill(history, funding, marks, end=end)

    # Then: A = -(0.001*2*100)*2 - (0.002*2*100) = -0.8 (09-03 08:00 은 청산 후라 제외), B = +0.05 - 0.1 = -0.05
    assert result.start == t0
    assert result.end == end
    assert result.by_symbol == {"AAAUSDT": Decimal("-0.8"), "BBBUSDT": Decimal("-0.05")}
    assert result.cash_delta == Decimal("-0.85")
    assert result.epochs == 5


def test_compute_funding_backfill_window_excludes_start_and_includes_end() -> None:
    from decimal import Decimal

    import pandas as pd

    from src.live.funding_backfill import compute_funding_backfill
    from src.live.ledger import PositionSnapshot

    # Given: effective_from 과 정확히 같은 epoch 은 제외, end 와 같은 epoch 은 포함, end 이후는 제외
    t0 = pd.Timestamp("2026-09-02 08:00Z")
    end = pd.Timestamp("2026-09-02 16:00Z")
    history = (PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("1")}),)
    epochs = pd.DatetimeIndex(["2026-09-02 00:00", "2026-09-02 08:00", "2026-09-02 16:00", "2026-09-03 00:00"], tz="UTC")
    funding = {"AAAUSDT": pd.Series([0.1, 0.2, 0.001, 0.3], index=epochs)}
    marks = {"AAAUSDT": pd.Series([10.0, 10.0, 10.0, 10.0], index=epochs)}

    # When
    result = compute_funding_backfill(history, funding, marks, end=end)

    # Then
    assert result.cash_delta == Decimal("-0.01")
    assert result.epochs == 1


def test_compute_funding_backfill_rejects_empty_or_non_positive_window() -> None:
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import compute_funding_backfill
    from src.live.ledger import PositionSnapshot

    t0 = pd.Timestamp("2026-09-02 01:30Z")
    history = (PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("1")}),)

    with pytest.raises(DataIntegrityError, match="requires position history"):
        compute_funding_backfill((), {}, {}, end=t0)
    with pytest.raises(DataIntegrityError, match="nothing to backfill"):
        compute_funding_backfill(history, {}, {}, end=t0)


def test_compute_funding_backfill_fails_closed_on_funding_coverage() -> None:
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import compute_funding_backfill
    from src.live.ledger import PositionSnapshot

    t0 = pd.Timestamp("2026-09-02 01:30Z")
    end = pd.Timestamp("2026-09-03 20:00Z")
    history = (PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("2"), "CCCUSDT": Decimal("1")}),)
    # Given: A 는 09-02 08:00 -> 09-03 16:00 사이 8h30m 초과 공백, C 는 펀딩 시리즈 자체가 없음
    gap_epochs = pd.DatetimeIndex(["2026-09-02 08:00", "2026-09-03 16:00"], tz="UTC")
    marks = {"AAAUSDT": pd.Series([100.0, 100.0], index=gap_epochs)}
    funding = {"AAAUSDT": pd.Series([0.001, 0.001], index=gap_epochs)}

    with pytest.raises(DataIntegrityError, match="funding coverage") as excinfo:
        compute_funding_backfill(history, funding, marks, end=end)
    assert "AAAUSDT" in str(excinfo.value)
    assert "CCCUSDT" in str(excinfo.value)

    # Given: 연속 커버리지지만 창 내 비유한 펀딩비
    full_epochs = pd.DatetimeIndex(["2026-09-02 08:00", "2026-09-02 16:00", "2026-09-03 00:00", "2026-09-03 08:00", "2026-09-03 16:00"], tz="UTC")
    solo = (PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("2")}),)
    nan_funding = {"AAAUSDT": pd.Series([0.001, float("nan"), 0.001, 0.001, 0.001], index=full_epochs)}
    nan_marks = {"AAAUSDT": pd.Series([100.0] * 5, index=full_epochs)}
    with pytest.raises(DataIntegrityError, match="non-finite funding rate"):
        compute_funding_backfill(solo, nan_funding, nan_marks, end=end)


def test_compute_funding_backfill_fails_closed_on_missing_mark() -> None:
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import compute_funding_backfill
    from src.live.ledger import PositionSnapshot

    t0 = pd.Timestamp("2026-09-02 01:30Z")
    end = pd.Timestamp("2026-09-02 20:00Z")
    history = (PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("2")}),)
    epochs = pd.DatetimeIndex(["2026-09-02 08:00:00.007", "2026-09-02 16:00"], tz="UTC")
    funding = {"AAAUSDT": pd.Series([0.001, 0.001], index=epochs)}
    # Given: 16:00 봉 mark 누락 (08:00 epoch 은 ms 지터가 있어도 floor('h') 로 매칭된다)
    marks = {"AAAUSDT": pd.Series([100.0], index=pd.DatetimeIndex(["2026-09-02 08:00"], tz="UTC"))}

    with pytest.raises(DataIntegrityError, match="mark missing"):
        compute_funding_backfill(history, funding, marks, end=end)


def test_reconstruct_position_history_cumulates_exact_decimals(tmp_path) -> None:
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import reconstruct_position_history

    d1 = pd.Timestamp("2026-09-02 00:00Z")
    d2 = pd.Timestamp("2026-09-03 00:00Z")
    fills = pd.DataFrame(
        {
            "decision_time": pd.to_datetime([d1, d1, d2, d2], utc=True),
            "symbol": ["AAAUSDT", "AAAUSDT", "AAAUSDT", "BBBUSDT"],
            "quantity_delta": [0.1, 0.2, -0.3, 5.0],
        }
    )
    effective = {d1: pd.Timestamp("2026-09-02 01:26Z"), d2: pd.Timestamp("2026-09-03 01:27Z")}

    # When
    history = reconstruct_position_history(fills, effective)

    # Then: float 누적 드리프트 없이 0.3 -> 0 으로 정확히 상쇄
    assert [snap.effective_from for snap in history] == [effective[d1], effective[d2]]
    assert history[0].positions == {"AAAUSDT": Decimal("0.3")}
    assert history[1].positions == {"BBBUSDT": Decimal("5")}

    with pytest.raises(DataIntegrityError, match="effective time missing"):
        reconstruct_position_history(fills, {d1: effective[d1]})
    with pytest.raises(DataIntegrityError, match="strictly increasing"):
        reconstruct_position_history(fills, {d1: effective[d2], d2: effective[d1]})
    with pytest.raises(DataIntegrityError, match="no fills"):
        reconstruct_position_history(fills.iloc[0:0], effective)
    with pytest.raises(DataIntegrityError, match="missing columns"):
        reconstruct_position_history(fills.drop(columns=["quantity_delta"]), effective)


def test_fill_effective_times_uses_first_intent_outcome_of_run(tmp_path) -> None:
    import json

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import fill_effective_times

    d1 = pd.Timestamp("2026-09-02 00:00Z")
    rows = [
        {"event": "symbol_dropped", "run_id": "20260902", "ts": "2026-09-02T01:26:00+00:00"},
        {"event": "intent_outcome", "run_id": "20260901", "ts": "2026-09-02T01:20:00+00:00"},
        {"event": "intent_outcome", "run_id": "20260902", "ts": "2026-09-02T01:26:10+00:00"},
        {"event": "intent_outcome", "run_id": "20260902", "ts": "2026-09-02T01:26:05+00:00"},
    ]
    (tmp_path / "2026-09-02.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # When
    result = fill_effective_times(tmp_path, [d1])

    # Then
    assert result == {d1: pd.Timestamp("2026-09-02 01:26:05Z")}

    with pytest.raises(DataIntegrityError, match="audit"):
        fill_effective_times(tmp_path, [pd.Timestamp("2026-09-03 00:00Z")])
    (tmp_path / "2026-09-04.jsonl").write_text('{"event": "cycle_complete", "run_id": "20260904", "ts": "2026-09-04T01:00:00+00:00"}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="audit"):
        fill_effective_times(tmp_path, [pd.Timestamp("2026-09-04 00:00Z")])
    (tmp_path / "2026-09-05.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="audit"):
        fill_effective_times(tmp_path, [pd.Timestamp("2026-09-05 00:00Z")])


def test_resolve_backfill_end_derivation_order() -> None:
    from decimal import Decimal

    import pandas as pd

    from src.live.funding_backfill import resolve_backfill_end
    from src.live.ledger import POSITION_HISTORY_MAX, LedgerState, PositionSnapshot

    now = pd.Timestamp("2026-09-16 10:00Z")
    bootstrap = pd.Timestamp("2026-09-15 01:05Z")
    snap = PositionSnapshot(effective_from=bootstrap, positions={"AAAUSDT": Decimal("1")})
    base = LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"))

    # 1) 엔진이 남긴 accrual 시작 마커가 최우선
    marked = LedgerState(positions=base.positions, cash_usdt=base.cash_usdt, funding_accrual_started_at=bootstrap, position_history=(snap,) * POSITION_HISTORY_MAX)
    assert resolve_backfill_end(marked, now, None) == (bootstrap, False)
    # 2) 원장 이력이 비었으면 now 까지 소급하고 accrual 시작점을 시드
    assert resolve_backfill_end(base, now, None) == (now, True)
    # 3) 마커 없음 + trim 되지 않은 이력 -> 첫 스냅샷
    untrimmed = LedgerState(positions=base.positions, cash_usdt=base.cash_usdt, position_history=(snap,))
    assert resolve_backfill_end(untrimmed, now, None) == (bootstrap, False)
    # 4) 유도 불가(trim 가능성) -> 운영자 지정값 사용
    trimmed = LedgerState(positions=base.positions, cash_usdt=base.cash_usdt, position_history=(snap,) * POSITION_HISTORY_MAX)
    assert resolve_backfill_end(trimmed, now, bootstrap) == (bootstrap, False)
    # 5) 유도값과 같은 운영자 지정값은 허용
    assert resolve_backfill_end(untrimmed, now, bootstrap) == (bootstrap, False)


def test_resolve_backfill_end_fails_closed() -> None:
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import resolve_backfill_end
    from src.live.ledger import POSITION_HISTORY_MAX, LedgerState, PositionSnapshot

    now = pd.Timestamp("2026-09-16 10:00Z")
    bootstrap = pd.Timestamp("2026-09-15 01:05Z")
    snap = PositionSnapshot(effective_from=bootstrap, positions={"AAAUSDT": Decimal("1")})
    positions = {"AAAUSDT": Decimal("1")}

    with pytest.raises(DataIntegrityError, match="requires cash_usdt"):
        resolve_backfill_end(LedgerState(positions=positions), now, None)
    with pytest.raises(DataIntegrityError, match="already applied"):
        resolve_backfill_end(LedgerState(positions=positions, cash_usdt=Decimal("1"), funding_backfilled_through=bootstrap), now, None)
    with pytest.raises(DataIntegrityError, match="ambiguous accrual start"):
        resolve_backfill_end(LedgerState(positions=positions, cash_usdt=Decimal("1"), funding_accrued_through=bootstrap), now, None)
    with pytest.raises(DataIntegrityError, match="ambiguous accrual start"):
        resolve_backfill_end(LedgerState(positions=positions, cash_usdt=Decimal("1"), funding_watermarks={"AAAUSDT": bootstrap}), now, None)
    trimmed = LedgerState(positions=positions, cash_usdt=Decimal("1"), position_history=(snap,) * POSITION_HISTORY_MAX)
    with pytest.raises(DataIntegrityError, match="--accrual-start"):
        resolve_backfill_end(trimmed, now, None)
    untrimmed = LedgerState(positions=positions, cash_usdt=Decimal("1"), position_history=(snap,))
    with pytest.raises(DataIntegrityError, match="does not match"):
        resolve_backfill_end(untrimmed, now, bootstrap + pd.Timedelta(minutes=1))
    with pytest.raises(DataIntegrityError, match="must not be after now"):
        resolve_backfill_end(trimmed, now, now + pd.Timedelta(seconds=1))
    with pytest.raises(DataIntegrityError, match="tz-aware"):
        resolve_backfill_end(trimmed, now, pd.Timestamp("2026-09-15 01:05"))


def test_apply_funding_backfill_updates_cash_and_markers() -> None:
    from decimal import Decimal

    import pandas as pd

    from src.live.funding_backfill import BackfillPlan, apply_funding_backfill
    from src.live.ledger import LedgerState

    start = pd.Timestamp("2026-09-02 01:26Z")
    end = pd.Timestamp("2026-09-15 01:05Z")
    state = LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"))
    seeding = BackfillPlan(start=start, end=end, cash_delta=Decimal("1.5"), by_symbol={"AAAUSDT": Decimal("1.5")}, epochs=3, seeds_accrual_start=True)

    seeded = apply_funding_backfill(state, seeding)
    assert seeded.cash_usdt == Decimal("1001.5")
    assert seeded.funding_backfilled_through == end
    assert seeded.funding_accrual_started_at == end
    assert seeded.funding_accrued_through == end
    assert seeded.positions == state.positions

    engine_started = pd.Timestamp("2026-09-15 01:05Z")
    running = LedgerState(positions=state.positions, cash_usdt=Decimal("1000"), funding_accrued_through=pd.Timestamp("2026-09-16 01:05Z"), funding_accrual_started_at=engine_started)
    plain = BackfillPlan(start=start, end=engine_started, cash_delta=Decimal("-2"), by_symbol={"AAAUSDT": Decimal("-2")}, epochs=3, seeds_accrual_start=False)
    applied = apply_funding_backfill(running, plain)
    assert applied.cash_usdt == Decimal("998")
    assert applied.funding_accrued_through == pd.Timestamp("2026-09-16 01:05Z")
    assert applied.funding_accrual_started_at == engine_started
    assert applied.funding_backfilled_through == engine_started


def test_run_paper_funding_backfill_dry_run_then_apply_is_idempotent(tmp_path) -> None:

    from decimal import Decimal

    import pandas as pd

    from src.live.fills import FillEvent, append_fills
    from src.live.ledger import LedgerState, save_ledger
    from src.live.settings import LiveSettings

    decision = pd.Timestamp("2026-09-02 00:00Z")
    ledger_path = tmp_path / "ledger.json"
    fills_dir = tmp_path / "fills"
    audit_dir = tmp_path / "shadow_cycle"
    audit_dir.mkdir()
    save_ledger(ledger_path, LedgerState(positions={"AAAUSDT": Decimal("2")}, equity_high_water_mark=Decimal("1000"), cash_usdt=Decimal("1000")))
    append_fills(
        [
            FillEvent(
                decision_time=decision, timestamp=decision, symbol="AAAUSDT", quantity_delta=Decimal("2"),
                fill_price=Decimal("100"), fee_bps=5.0, reason="immediate_taker", pre_trade_equity=Decimal("1000"),
                liquidity="taker", mode="paper", run_id="20260902", leg_index=0, client_order_id="20260902",
            )
        ],
        fills_dir,
    )
    (audit_dir / "2026-09-02.jsonl").write_text('{"event": "intent_outcome", "run_id": "20260902", "ts": "2026-09-02T01:30:00+00:00"}\n', encoding="utf-8")
    epochs = pd.DatetimeIndex(["2026-09-02 08:00", "2026-09-02 16:00", "2026-09-03 00:00"], tz="UTC")
    funding_loader = lambda symbols: {"AAAUSDT": pd.Series([0.001, 0.001, 0.001], index=epochs)}
    mark_loader = lambda symbols: {"AAAUSDT": pd.Series([100.0, 100.0, 100.0], index=epochs)}
    settings = LiveSettings(mode="paper", ledger_path=str(ledger_path), fills_dir=str(fills_dir))
    now = pd.Timestamp("2026-09-03 02:00Z")

    import json

    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import run_paper_funding_backfill
    from src.live.ledger import load_ledger

    kwargs = dict(
        now=now, funding_loader=funding_loader, mark_loader=mark_loader, shadow_audit_dir=audit_dir,
        backfill_audit_path=tmp_path / "backfill.jsonl", heartbeat_path=tmp_path / "missing_heartbeat.json",
    )

    # When: dry-run
    dry = run_paper_funding_backfill(settings, apply=False, **kwargs)

    # Then: 원장 불변, 계획은 -(0.001*2*100)*3 = -0.6
    assert dry.cash_delta == Decimal("-0.6")
    assert dry.start == pd.Timestamp("2026-09-02 01:30Z")
    assert dry.end == now
    assert dry.seeds_accrual_start is True
    assert load_ledger(ledger_path).cash_usdt == Decimal("1000")
    assert load_ledger(ledger_path).funding_backfilled_through is None

    # When: apply
    applied = run_paper_funding_backfill(settings, apply=True, **kwargs)

    # Then
    reloaded = load_ledger(ledger_path)
    assert applied.cash_delta == Decimal("-0.6")
    assert reloaded.cash_usdt == Decimal("999.4")
    assert reloaded.funding_backfilled_through == now
    assert reloaded.funding_accrual_started_at == now
    assert reloaded.funding_accrued_through == now
    records = [json.loads(line) for line in (tmp_path / "backfill.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["applied"] for r in records] == [False, True]
    assert records[1]["event"] == "paper_funding_backfill"
    assert Decimal(records[1]["cash_delta"]) == Decimal("-0.6")
    assert {k: Decimal(v) for k, v in records[1]["by_symbol"].items()} == {"AAAUSDT": Decimal("-0.6")}

    # When/Then: 재실행은 fail-closed
    with pytest.raises(DataIntegrityError, match="already applied"):
        run_paper_funding_backfill(settings, apply=True, **kwargs)


def test_run_paper_funding_backfill_guards(tmp_path) -> None:

    from decimal import Decimal

    import pandas as pd

    from src.live.fills import FillEvent, append_fills
    from src.live.ledger import LedgerState, save_ledger
    from src.live.settings import LiveSettings

    decision = pd.Timestamp("2026-09-02 00:00Z")
    ledger_path = tmp_path / "ledger.json"
    fills_dir = tmp_path / "fills"
    audit_dir = tmp_path / "shadow_cycle"
    audit_dir.mkdir()
    save_ledger(ledger_path, LedgerState(positions={"AAAUSDT": Decimal("2")}, equity_high_water_mark=Decimal("1000"), cash_usdt=Decimal("1000")))
    append_fills(
        [
            FillEvent(
                decision_time=decision, timestamp=decision, symbol="AAAUSDT", quantity_delta=Decimal("2"),
                fill_price=Decimal("100"), fee_bps=5.0, reason="immediate_taker", pre_trade_equity=Decimal("1000"),
                liquidity="taker", mode="paper", run_id="20260902", leg_index=0, client_order_id="20260902",
            )
        ],
        fills_dir,
    )
    (audit_dir / "2026-09-02.jsonl").write_text('{"event": "intent_outcome", "run_id": "20260902", "ts": "2026-09-02T01:30:00+00:00"}\n', encoding="utf-8")
    epochs = pd.DatetimeIndex(["2026-09-02 08:00", "2026-09-02 16:00", "2026-09-03 00:00"], tz="UTC")
    funding_loader = lambda symbols: {"AAAUSDT": pd.Series([0.001, 0.001, 0.001], index=epochs)}
    mark_loader = lambda symbols: {"AAAUSDT": pd.Series([100.0, 100.0, 100.0], index=epochs)}
    settings = LiveSettings(mode="paper", ledger_path=str(ledger_path), fills_dir=str(fills_dir))
    now = pd.Timestamp("2026-09-03 02:00Z")

    import json

    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import run_paper_funding_backfill
    from src.live.ledger import load_ledger

    heartbeat = tmp_path / "heartbeat.json"
    kwargs = dict(
        now=now, funding_loader=funding_loader, mark_loader=mark_loader, shadow_audit_dir=audit_dir,
        backfill_audit_path=tmp_path / "backfill.jsonl", heartbeat_path=heartbeat,
    )

    # Given: 데몬이 execute 단계 진행 중(신선한 heartbeat)
    heartbeat.write_text(json.dumps({"stage": "execute", "status": "RUNNING", "ts": (now - pd.Timedelta(minutes=5)).isoformat()}), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="daemon busy"):
        run_paper_funding_backfill(settings, apply=True, **kwargs)
    assert load_ledger(ledger_path).cash_usdt == Decimal("1000")
    # dry-run 은 읽기 전용이라 busy 여도 허용
    assert run_paper_funding_backfill(settings, apply=False, **kwargs).cash_delta == Decimal("-0.6")
    # stale busy heartbeat(> BACKFILL_BUSY_STALE_S)는 크래시 잔재로 보고 허용
    heartbeat.write_text(json.dumps({"stage": "execute", "status": "RUNNING", "ts": (now - pd.Timedelta(hours=2)).isoformat()}), encoding="utf-8")
    assert run_paper_funding_backfill(settings, apply=True, **kwargs).cash_delta == Decimal("-0.6")

    # Given: 억제되지 않는 모드
    with pytest.raises(DataIntegrityError, match="mutation-suppressed"):
        run_paper_funding_backfill(LiveSettings(mode="live_testnet", ledger_path=str(ledger_path), fills_dir=str(fills_dir)), apply=False, **kwargs)

    # Given: 체결 누적과 원장 포지션 불일치
    mismatch_path = tmp_path / "mismatch.json"
    save_ledger(mismatch_path, LedgerState(positions={"AAAUSDT": Decimal("3")}, cash_usdt=Decimal("1000")))
    with pytest.raises(DataIntegrityError, match="do not reconcile"):
        run_paper_funding_backfill(LiveSettings(mode="paper", ledger_path=str(mismatch_path), fills_dir=str(fills_dir)), apply=False, **kwargs)

    # Given: 해당 모드 체결 없음
    with pytest.raises(DataIntegrityError, match="no fills"):
        run_paper_funding_backfill(LiveSettings(mode="shadow", ledger_path=str(mismatch_path), fills_dir=str(fills_dir)), apply=False, **kwargs)




def test_funding_backfill_minor_fail_closed_guards(tmp_path) -> None:
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import (
        BackfillPlan,
        _assert_daemon_idle,
        apply_funding_backfill,
        reconstruct_position_history,
    )
    from src.live.ledger import LedgerState

    now = pd.Timestamp("2026-09-15 02:00Z")
    d1 = pd.Timestamp("2026-09-02 00:00Z")

    # Given: 비유한 체결 수량
    fills = pd.DataFrame({"decision_time": pd.to_datetime([d1], utc=True), "symbol": ["AAAUSDT"], "quantity_delta": [float("inf")]})
    with pytest.raises(DataIntegrityError, match="non-finite quantity_delta"):
        reconstruct_position_history(fills, {d1: pd.Timestamp("2026-09-02 01:26Z")})

    # Given: 현금 없는 원장에 적용
    plan = BackfillPlan(start=d1, end=now, cash_delta=Decimal("1"), by_symbol={"AAAUSDT": Decimal("1")}, epochs=1, seeds_accrual_start=False)
    with pytest.raises(DataIntegrityError, match="requires cash_usdt"):
        apply_funding_backfill(LedgerState(positions={"AAAUSDT": Decimal("1")}), plan)

    # Given: 깨진 JSON / dict 가 아닌 heartbeat
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="heartbeat unreadable"):
        _assert_daemon_idle(heartbeat, now)
    heartbeat.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="heartbeat unreadable"):
        _assert_daemon_idle(heartbeat, now)

    # Given: idle heartbeat 는 통과
    heartbeat.write_text('{"stage": "idle", "ts": "2026-09-15T01:59:00+00:00"}', encoding="utf-8")
    assert _assert_daemon_idle(heartbeat, now) is None
