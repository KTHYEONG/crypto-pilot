from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pytest_mock import MockerFixture

from src.domain.futures.strategy.tiered_workflow.cross_tf_diagnostics import STAGE_ORDER


def _make_stage_entry(tf: str, seed: int = 0) -> dict[str, object]:
    return {"count": 10, "digest": f"digest_{seed}_{tf}"}


_COMMON_TFS = ("2h", "4h", "6h", "8h", "12h", "1d")


def _write_label_json(
    out_dir: Path,
    label: str,
    *,
    tfs: tuple[str, ...] = _COMMON_TFS,
    missing_stage: str | None = None,
) -> None:
    payload: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        if stage == missing_stage:
            continue
        payload[stage] = {tf: _make_stage_entry(tf) for tf in tfs}
    payload["runner_result"] = {"exit_code": 0, "reason": "l1_mode_done"}
    (out_dir / f"{label}.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class TestRunSupervised:
    """Scenario 4 (Integration): supervisor runs all 4 labels, calls diagnose_snapshots."""

    def test_run_supervised_returns_1_when_incomplete_trace(self, mocker: MockerFixture, tmp_path: Path) -> None:
        out_dir = tmp_path / "logs/futures/diagnostics/l1_cross_tf"
        out_dir.mkdir(parents=True)
        for label in ("control", "control_repeat", "treatment", "fusion_ablation"):
            missing = "outer_folds" if label == "treatment" else None
            _write_label_json(out_dir, label, missing_stage=missing)

        call_order: list[str] = []

        def fake_popen(args: list[str], **kwargs: Any) -> Any:
            label = args[-1]
            call_order.append(label)
            proc = mocker.MagicMock()
            proc.pid = 12345
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            return proc

        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_diagnosis.subprocess.Popen", side_effect=fake_popen)
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_diagnosis.psutil.Process")
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_diagnosis._OUT_DIR",
            out_dir,
        )

        from src.domain.futures.strategy.run_l1_cross_tf_diagnosis import run_supervised

        exit_code = run_supervised()

        assert call_order == ["control", "control_repeat", "treatment", "fusion_ablation"]
        assert exit_code == 1
        assert (out_dir / "diagnosis.json").exists()
        diagnosis = json.loads((out_dir / "diagnosis.json").read_text(encoding="utf-8"))
        assert diagnosis["classification"] == "incomplete_trace"
        assert diagnosis["complete"] is False

    def test_run_supervised_returns_0_when_all_complete(self, mocker: MockerFixture, tmp_path: Path) -> None:
        out_dir = tmp_path / "logs/futures/diagnostics/l1_cross_tf"
        out_dir.mkdir(parents=True)
        for label in ("control", "control_repeat", "treatment", "fusion_ablation"):
            _write_label_json(out_dir, label)

        def fake_popen(args: list[str], **kwargs: Any) -> Any:
            proc = mocker.MagicMock()
            proc.pid = 12345
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            return proc

        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_diagnosis.subprocess.Popen", side_effect=fake_popen)
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_diagnosis.psutil.Process")
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_diagnosis._OUT_DIR",
            out_dir,
        )

        from src.domain.futures.strategy.run_l1_cross_tf_diagnosis import run_supervised

        exit_code = run_supervised()
        assert exit_code == 0

    def test_run_supervised_returns_1_when_child_exit_code_nonzero(self, mocker: MockerFixture, tmp_path: Path) -> None:
        out_dir = tmp_path / "logs/futures/diagnostics/l1_cross_tf"
        out_dir.mkdir(parents=True)
        for label in ("control", "control_repeat", "treatment", "fusion_ablation"):
            _write_label_json(out_dir, label)
            if label == "treatment":
                payload = json.loads((out_dir / f"{label}.json").read_text(encoding="utf-8"))
                payload["runner_result"] = {"exit_code": 1, "reason": "layer1_blocked"}
                (out_dir / f"{label}.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

        def fake_popen(args: list[str], **kwargs: Any) -> Any:
            proc = mocker.MagicMock()
            proc.pid = 12345
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            return proc

        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_diagnosis.subprocess.Popen", side_effect=fake_popen)
        mocker.patch("src.domain.futures.strategy.run_l1_cross_tf_diagnosis.psutil.Process")
        mocker.patch(
            "src.domain.futures.strategy.run_l1_cross_tf_diagnosis._OUT_DIR",
            out_dir,
        )

        from src.domain.futures.strategy.run_l1_cross_tf_diagnosis import run_supervised

        exit_code = run_supervised()
        assert exit_code == 1
