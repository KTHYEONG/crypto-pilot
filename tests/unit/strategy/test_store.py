from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.strategy import store
from src.strategy.store import EXAMPLE_STORE_ROOT, load_strategy_values, ml_tilt_values, strategy_store_root

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "strategy_store"
_PARAMS = Path("crypto-pilot") / "params.toml"


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    shutil.copytree(FIXTURE_ROOT, root)
    return root


def _rewrite(root: Path, transform) -> None:
    path = root / _PARAMS
    path.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")


def _drop_key(key: str):
    def transform(text: str) -> str:
        if key == "members":
            return re.sub(r'\[\[flow_mom\.members\]\]\nname = "[^"]*"\nsign = -?\d\n\n?', "", text)
        if key == "sets":
            return re.sub(r"\[committee\.sets\]\n(?:[^\n]+\n)*", "", text)
        return re.sub(rf"^{key} = [^\n]*\n", "", text, count=1, flags=re.M)
    return transform


def _replace(old: str, new: str):
    def transform(text: str) -> str:
        assert old in text, old
        return text.replace(old, new, 1)
    return transform


def test_loads_fixture_values_in_member_order() -> None:
    values = load_strategy_values(FIXTURE_ROOT)
    assert values.example is False
    assert values.members[0] == ("mom_168h", 1)
    assert [name for name, _ in values.members] == [
        "mom_168h", "taker_imb_24h", "amihud",
    ]
    assert values.committee_default == "flow_momentum"
    assert values.committee_members == values.committee_sets["flow_momentum"]
    assert isinstance(values.breadth, int)
    assert isinstance(values.name_clip, float)


@pytest.mark.parametrize(
    "key_path",
    [
        ("meta", "schema"), ("meta", "version"), ("meta", "created_at"), ("meta", "source"),
        ("meta", "validated_by"), ("flow_mom", "breadth"), ("flow_mom", "control_breadth"),
        ("flow_mom", "min_rank_symbols"), ("flow_mom", "name_clip"),
        ("flow_mom", "growth_exposure_multiplier"), ("flow_mom", "members"),
        ("committee", "default"), ("committee", "sets"),
    ],
)
def test_missing_key_fails_closed_naming_the_key(tmp_path: Path, key_path: tuple[str, str]) -> None:
    root = _copy_fixture(tmp_path)
    _rewrite(root, _drop_key(key_path[1]))
    with pytest.raises(DataIntegrityError, match=key_path[1]):
        load_strategy_values(root)


def test_type_violations_fail_closed(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    _rewrite(root, _replace("breadth = 24", "breadth = true"))
    with pytest.raises(DataIntegrityError, match="breadth"):
        load_strategy_values(root)
    root = _copy_fixture(tmp_path / "b")
    _rewrite(root, _replace("sign = 1", "sign = 0"))
    with pytest.raises(DataIntegrityError, match="sign"):
        load_strategy_values(root)
    root = _copy_fixture(tmp_path / "c")
    _rewrite(root, _replace('name = "taker_imb_24h"', 'name = "mom_168h"'))
    with pytest.raises(DataIntegrityError, match="duplicate"):
        load_strategy_values(root)
    root = _copy_fixture(tmp_path / "d")
    _rewrite(root, _replace('default = "flow_momentum"', 'default = "no_such_set"'))
    with pytest.raises(DataIntegrityError, match=r"committee\.default"):
        load_strategy_values(root)

def test_error_message_never_contains_a_value(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    _rewrite(root, _replace("name_clip = 0.07", "name_clip = -0.5"))
    with pytest.raises(DataIntegrityError) as exc:
        load_strategy_values(root)
    assert "-0.5" not in str(exc.value)


def test_relative_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", "relative/store")
    with pytest.raises(DataIntegrityError, match="absolute"):
        strategy_store_root()


def test_unset_env_uses_example_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRYPTO_STRATEGY_STORE", raising=False)
    assert strategy_store_root() == EXAMPLE_STORE_ROOT
    values = load_strategy_values(strategy_store_root())
    assert values.example is True
    assert values.root == EXAMPLE_STORE_ROOT


def test_env_store_never_falls_back_to_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(empty))
    with pytest.raises(DataIntegrityError):
        load_strategy_values(strategy_store_root())


def test_store_revision_reads_file_or_none(tmp_path: Path) -> None:
    assert store.store_revision(tmp_path) is None
    (tmp_path / "STORE_REVISION").write_text("  abc123 \n", encoding="utf-8")
    assert store.store_revision(tmp_path) == "abc123"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('version = "fixture"', "version = 1", "meta.version"),
        ("name_clip = 0.07", 'name_clip = "wide"', "flow_mom.name_clip"),
        ("name_clip = 0.07", "name_clip = inf", "flow_mom.name_clip"),
        ('schema = "crypto.strategy_values.v1"', 'schema = "other.v9"', "meta.schema"),
        ("example = false", 'example = "no"', "meta.example"),
    ],
)
def test_scalar_violations_fail_closed(tmp_path: Path, old: str, new: str, message: str) -> None:
    root = _copy_fixture(tmp_path)
    _rewrite(root, _replace(old, new))
    with pytest.raises(DataIntegrityError, match=message.replace(".", r"\.")):
        load_strategy_values(root)


def test_structural_violations_fail_closed(tmp_path: Path) -> None:
    def _with_tail(text: str, tail: str) -> str:
        head = text[: text.index("[[flow_mom.members]]")]
        return head + tail

    committee = '[committee]\ndefault = "flow_momentum"\n'
    members = '[[flow_mom.members]]\nname = "mom_168h"\nsign = 1\n\n'
    cases = [
        ("members = []\n\n" + committee + '[committee.sets]\nflow_momentum = ["mom_168h"]\n', "flow_mom.members"),
        ("members = [1]\n\n" + committee + '[committee.sets]\nflow_momentum = ["mom_168h"]\n', r"flow_mom\.members\[0\]"),
        (members + committee + "sets = 3\n", "committee.sets"),
        (members + committee + '[committee.sets]\nflow_momentum = ["mom_168h", 5]\n', "committee.sets.flow_momentum"),
        (members + committee + '[committee.sets]\nflow_momentum = ["mom_168h", "mom_168h"]\n', "duplicate"),
    ]
    for index, (tail, message) in enumerate(cases):
        root = _copy_fixture(tmp_path / str(index))
        _rewrite(root, lambda text, tail=tail: _with_tail(text, tail))
        with pytest.raises(DataIntegrityError, match=message):
            load_strategy_values(root)


def test_non_table_sections_fail_closed(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / _PARAMS).write_text('meta = 1\nflow_mom = 2\ncommittee = 3\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="malformed"):
        load_strategy_values(root)


_ML_TILT_SECTION = """
[ml_tilt]
tilt = 0.10
horizon_days = 1
label_clip = 1.0
vol_floor_quantile = 0.01
smooth_halflife_days = 1.0
retrain_days = 30
purge_days = 2
schedule_anchor = "2000-01-01T00:00:00Z"
feature_breadth = 48
num_leaves = 3
min_child_samples = 10
learning_rate = 0.1
n_trees = 10
subsample = 0.5
colsample_bytree = 0.5
reg_lambda = 1.0
seed = 1
min_train_rows = 100
"""


def test_ml_values_absent_returns_none(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    assert ml_tilt_values(root) is None


def test_ml_values_reject_short_purge(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / _PARAMS).write_text(
        (root / _PARAMS).read_text(encoding="utf-8") + _ML_TILT_SECTION.replace("purge_days = 2", "purge_days = 1"),
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError, match=r"ml_tilt\.purge_days"):
        ml_tilt_values(root)


def test_example_store_has_ml_tilt() -> None:
    values = ml_tilt_values(EXAMPLE_STORE_ROOT)
    assert values is not None
    assert values.schedule_anchor.tzinfo is not None


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ("tilt = 0.10", "tilt = 1.0", "tilt"),
        ("tilt = 0.10", "tilt = nan", "tilt"),
        ("vol_floor_quantile = 0.01", "vol_floor_quantile = 0.5", "vol_floor_quantile"),
        ("subsample = 0.5", "subsample = 1.5", "subsample"),
        ("colsample_bytree = 0.5", "colsample_bytree = 0.0", "colsample_bytree"),
        ("colsample_bytree = 0.5", "colsample_bytree = 1.5", "colsample_bytree"),
        ("feature_breadth = 48", "feature_breadth = 1", "feature_breadth"),
        ('schedule_anchor = "2000-01-01T00:00:00Z"', 'schedule_anchor = "not-a-date"', "schedule_anchor"),
        ('schedule_anchor = "2000-01-01T00:00:00Z"', 'schedule_anchor = "2000-01-01T00:00:00"', "schedule_anchor"),
        ('schedule_anchor = "2000-01-01T00:00:00Z"', 'schedule_anchor = "2000-01-01T00:00:00+02:00"', "schedule_anchor"),
    ],
)
def test_ml_values_out_of_range_fail_closed(tmp_path: Path, old: str, new: str, key: str) -> None:
    root = _copy_fixture(tmp_path)
    (root / _PARAMS).write_text(
        (root / _PARAMS).read_text(encoding="utf-8") + _ML_TILT_SECTION.replace(old, new),
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError, match=rf"ml_tilt\.{key}"):
        ml_tilt_values(root)


def test_ml_values_unparseable_params_fail_closed(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / _PARAMS).write_text("[ml_tilt\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="unreadable"):
        ml_tilt_values(root)


def test_ml_section_must_be_a_table(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / _PARAMS).write_text("ml_tilt = 5\n" + (root / _PARAMS).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="malformed"):
        ml_tilt_values(root)
