from __future__ import annotations

from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS, MIRROR_DEFAULT_DATASETS


def test_default_datasets_exclude_only_aggtrades() -> None:
    assert tuple(name for name in MIRROR_DATASET_SPECS if name != "aggTrades") == MIRROR_DEFAULT_DATASETS
    assert "aggTrades" in MIRROR_DATASET_SPECS


def test_bucket_roots_follow_cadence_and_name() -> None:
    for name, spec in MIRROR_DATASET_SPECS.items():
        assert spec.name == name
        assert spec.bucket_root == f"data/futures/um/{spec.cadence}/{name}/"
        assert spec.bucket_root.endswith("/")


def test_kline_families_use_one_minute_infix() -> None:
    for name in ("klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines"):
        assert MIRROR_DATASET_SPECS[name].key_infix == "1m/"
    assert MIRROR_DATASET_SPECS["metrics"].key_infix == ""


def test_only_aggtrades_is_residual_and_text_columns_exist() -> None:
    residual = [name for name, spec in MIRROR_DATASET_SPECS.items() if spec.id_residual]
    assert residual == ["aggTrades"]
    for spec in MIRROR_DATASET_SPECS.values():
        assert spec.text_columns <= set(spec.columns)
