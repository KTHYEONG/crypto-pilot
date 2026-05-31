"""Regression tests for AlphaArtifactHash.structural_hash cross-split IS/HO/OOS consistency."""
from __future__ import annotations

from src.domain.futures.forecast.contracts import AlphaArtifactHash


def _make_hash(
    alpha_cfg: str = "a1",
    feature_cfg: str = "f1",
    label_cfg: str = "l1",
    train_window: str = "tw_is",
    fold_spec: str = "fold_is",
    model_family: str = "lgbm",
    selected_horizon: int = 6,
) -> AlphaArtifactHash:
    return AlphaArtifactHash(
        alpha_config_hash=alpha_cfg,
        feature_config_hash=feature_cfg,
        label_config_hash=label_cfg,
        train_window_hash=train_window,
        fold_spec_hash=fold_spec,
        model_family=model_family,
        selected_horizon=selected_horizon,
    )


class TestStructuralHashCrossplit:
    def test_structural_hash_stable_across_is_ho_oos(self) -> None:
        # IS, HO, OOS have different data windows → different train_window_hash and fold_spec_hash
        h_is = _make_hash(train_window="tw_is", fold_spec="fold_0_of_3")
        h_ho = _make_hash(train_window="tw_ho", fold_spec="fold_1_of_3")
        h_oos = _make_hash(train_window="tw_oos", fold_spec="fold_2_of_3")

        # structural_hash must match across splits (config fields identical)
        assert h_is.structural_hash() == h_ho.structural_hash()
        assert h_is.structural_hash() == h_oos.structural_hash()

    def test_combined_hash_differs_across_splits(self) -> None:
        # combined() includes window hashes → must differ across IS/HO/OOS
        h_is = _make_hash(train_window="tw_is", fold_spec="fold_0")
        h_oos = _make_hash(train_window="tw_oos", fold_spec="fold_2")

        assert h_is.combined() != h_oos.combined()

    def test_structural_hash_detects_config_change(self) -> None:
        # Different alpha_config_hash → structural_hash must differ
        h1 = _make_hash(alpha_cfg="config_v1")
        h2 = _make_hash(alpha_cfg="config_v2")

        assert h1.structural_hash() != h2.structural_hash()

    def test_structural_hash_detects_horizon_change(self) -> None:
        h1 = _make_hash(selected_horizon=6)
        h2 = _make_hash(selected_horizon=12)

        assert h1.structural_hash() != h2.structural_hash()

    def test_structural_hash_detects_model_family_change(self) -> None:
        h1 = _make_hash(model_family="lgbm")
        h2 = _make_hash(model_family="xgboost")

        assert h1.structural_hash() != h2.structural_hash()

    def test_structural_hash_length_16(self) -> None:
        h = _make_hash()
        assert len(h.structural_hash()) == 16

    def test_structural_hash_deterministic(self) -> None:
        h = _make_hash()
        assert h.structural_hash() == h.structural_hash()

    def test_combined_hash_still_tracks_windows(self) -> None:
        # combined() remains useful for full provenance (full 7-field fingerprint)
        h_is = _make_hash(train_window="tw_is")
        h_ho = _make_hash(train_window="tw_ho")

        assert h_is.combined() != h_ho.combined()
        assert len(h_is.combined()) == 16
