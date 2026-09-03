from __future__ import annotations

import pytest

from src.quant.technical_experts.contracts import TechnicalCandidate


def _candidate(**overrides: object) -> TechnicalCandidate:
    base: dict[str, object] = {
        "candidate_id": "ema_alignment_long",
        "return_source": "technical_ema_alignment_long",
        "family": "ema_alignment",
        "side": "LONG",
        "config": {"fast": 20, "mid": 50, "slow": 200},
        "min_history_bars": 201,
    }
    base.update(overrides)
    return TechnicalCandidate(
        candidate_id=str(base["candidate_id"]),
        return_source=str(base["return_source"]),
        family=str(base["family"]),
        side=str(base["side"]),  # type: ignore[arg-type]
        config=base["config"],  # type: ignore[arg-type]
        min_history_bars=int(base["min_history_bars"]),
    )


class TestTechnicalCandidate:
    def test_candidate_rejects_invalid_identity(self) -> None:
        with pytest.raises(ValueError, match="candidate_id"):
            _candidate(candidate_id="")
        with pytest.raises(ValueError, match="return_source"):
            _candidate(return_source="")
        with pytest.raises(ValueError, match="family"):
            _candidate(family="")

    def test_candidate_rejects_invalid_side_and_history(self) -> None:
        with pytest.raises(ValueError, match="side"):
            _candidate(side="FLAT")
        with pytest.raises(ValueError, match="min_history_bars"):
            _candidate(min_history_bars=0)

    def test_candidate_rejects_empty_or_non_numeric_config(self) -> None:
        with pytest.raises(ValueError, match="config"):
            _candidate(config={})
        with pytest.raises(ValueError, match="numeric"):
            _candidate(config={"fast": "20"})

    def test_candidate_rejects_mismatched_return_source(self) -> None:
        # A source that names a different family than the candidate is rejected.
        with pytest.raises(ValueError, match="return_source"):
            _candidate(return_source="technical_macd_histogram_regime_long")
        # A source whose side disagrees with the candidate's side is rejected.
        with pytest.raises(ValueError, match="return_source"):
            _candidate(return_source="technical_ema_alignment_short")

    def test_valid_candidate_preserves_exact_identity(self) -> None:
        candidate = _candidate()
        assert candidate.side == "LONG"
        assert candidate.return_source == "technical_ema_alignment_long"
        assert candidate.config == {"fast": 20, "mid": 50, "slow": 200}
        assert candidate.min_history_bars == 201


def test_candidate_signature_is_frozen() -> None:
    from inspect import signature

    params = signature(TechnicalCandidate).parameters
    assert list(params) == [
        "candidate_id", "return_source", "family", "side", "config", "min_history_bars",
    ]
