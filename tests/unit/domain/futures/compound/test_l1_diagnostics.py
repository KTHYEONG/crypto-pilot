from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder


class TestL1AdmissionRecorder:
    def test_disabled_by_default(self) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            rec = L1AdmissionRecorder()
            assert not rec.enabled
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old

    def test_enabled_when_debug_is_1(self) -> None:
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder()
            assert rec.enabled
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_sleeve_noop_when_disabled(self) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            rec = L1AdmissionRecorder()
            rec.record_sleeve(signal_id="s", fold=0, cluster=0, beta=0.0, se_hac=0.1, se_ols_ratio=1.0, prob=0.5, n_obs=100, n_blocks=1, admitted=True)
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old

    def test_record_gate_noop_when_disabled(self) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            rec = L1AdmissionRecorder()
            rec.record_gate(admitted_sleeves=1, distinct_series=1, oos_bars=10, ann_growth=0.0, ann_lcb90=0.0, pw_block=5.0, turnover=0.0, cost_drag=0.0, admitted=False)
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old

    def test_record_gate_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "l1_admission.jsonl"
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder(path=path)
            assert rec.enabled
            rec.record_gate(admitted_sleeves=2, distinct_series=1, oos_bars=50, ann_growth=0.05, ann_lcb90=0.01, pw_block=5.0, turnover=0.1, cost_drag=0.0002, admitted=True)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["tag"] == "EVAL"
            assert parsed["admitted"] is True
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_sleeve_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "l1_admission.jsonl"
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder(path=path)
            rec.record_sleeve(signal_id="trend:fast", fold=1, cluster=2, beta=0.5, se_hac=0.2, se_ols_ratio=2.5, prob=0.95, n_obs=500, n_blocks=12, admitted=True)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["tag"] == "ALGO"
            assert parsed["signal_id"] == "trend:fast"
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_unwritable_directory_does_not_raise(self) -> None:
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder(path=Path("/nonexistent_dir/out.jsonl"))
            rec.record_gate(admitted_sleeves=1, distinct_series=1, oos_bars=10, ann_growth=0.0, ann_lcb90=0.0, pw_block=5.0, turnover=0.0, cost_drag=0.0, admitted=False)
            assert not rec.enabled
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)
