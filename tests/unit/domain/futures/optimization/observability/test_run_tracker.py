"""Tests for run_tracker champion-store and study reset helpers.

Covers:
- C: get_or_create_study(resume=False)가 기존 study를 삭제하고 새로 만드는지
- D: champion store(load_champion_params/update_champion_store) 갱신 로직
"""

from __future__ import annotations

import optuna
import pytest
from optuna.samplers import TPESampler

from src.domain.futures.optimization.observability.run_tracker import (
    champion_store_study_name,
    get_or_create_study,
    load_champion_params,
    update_champion_store,
)

_SPACE = {
    "kelly_fraction": {"type": "float", "low": 0.15, "high": 0.55},
    "K_RANK": {"type": "int", "low": 1, "high": 5, "step": 1},
}


@pytest.fixture
def storage() -> optuna.storages.BaseStorage:
    return optuna.storages.InMemoryStorage()


# ---------------------------------------------------------------------------
# get_or_create_study(resume=False)
# ---------------------------------------------------------------------------


def test_get_or_create_study_resume_false_deletes_existing_trials(
    storage: optuna.storages.BaseStorage,
) -> None:
    """resume=False면 기존 study의 trial이 모두 사라진 새 study가 반환된다."""
    # Arrange
    name = "l2_study_reset_target"
    old_study = optuna.create_study(study_name=name, storage=storage, direction="maximize")
    old_study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=3)
    assert len(old_study.trials) == 3

    # Act
    new_study = get_or_create_study(
        study_name=name, storage=storage, sampler=TPESampler(seed=0), resume=False
    )

    # Assert
    assert len(new_study.trials) == 0


def test_get_or_create_study_resume_true_keeps_existing_trials(
    storage: optuna.storages.BaseStorage,
) -> None:
    """resume=True면 기존 trial이 보존된다."""
    # Arrange
    name = "l2_study_resume_target"
    old_study = optuna.create_study(study_name=name, storage=storage, direction="maximize")
    old_study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=2)

    # Act
    resumed = get_or_create_study(
        study_name=name, storage=storage, sampler=TPESampler(seed=0), resume=True
    )

    # Assert
    assert len(resumed.trials) == 2


# ---------------------------------------------------------------------------
# champion_store_study_name
# ---------------------------------------------------------------------------


def test_champion_store_study_name_is_stable_per_tag() -> None:
    """동일 tag는 동일한 레저 study 이름을 생성한다."""
    assert champion_store_study_name("4h") == champion_store_study_name("4h")
    assert champion_store_study_name("4h") != champion_store_study_name("1h")


# ---------------------------------------------------------------------------
# load_champion_params
# ---------------------------------------------------------------------------


def test_load_champion_params_returns_none_when_no_ledger_exists(
    storage: optuna.storages.BaseStorage,
) -> None:
    """레저 study가 아예 없으면 None을 반환한다."""
    result = load_champion_params(tag="4h", storage=storage)

    assert result is None


def test_load_champion_params_returns_none_when_ledger_is_empty(
    storage: optuna.storages.BaseStorage,
) -> None:
    """레저 study는 있으나 trial이 없으면 None을 반환한다."""
    optuna.create_study(
        study_name=champion_store_study_name("4h"), storage=storage, direction="maximize"
    )

    result = load_champion_params(tag="4h", storage=storage)

    assert result is None


# ---------------------------------------------------------------------------
# update_champion_store
# ---------------------------------------------------------------------------


def test_update_champion_store_creates_first_champion(
    storage: optuna.storages.BaseStorage,
) -> None:
    """레저가 비어있을 때 첫 챔피언은 항상 갱신된다."""
    # Act
    updated = update_champion_store(
        tag="4h",
        storage=storage,
        params={"kelly_fraction": 0.3, "K_RANK": 3},
        value=0.10,
        space=_SPACE,
    )

    # Assert
    assert updated is True
    loaded = load_champion_params(tag="4h", storage=storage)
    assert loaded == {"kelly_fraction": 0.3, "K_RANK": 3}


def test_update_champion_store_skips_when_not_better(
    storage: optuna.storages.BaseStorage,
) -> None:
    """기존 챔피언보다 낮은 value는 레저를 갱신하지 않는다."""
    # Arrange
    update_champion_store(
        tag="4h", storage=storage, params={"kelly_fraction": 0.3, "K_RANK": 3}, value=0.20, space=_SPACE
    )

    # Act
    updated = update_champion_store(
        tag="4h", storage=storage, params={"kelly_fraction": 0.4, "K_RANK": 4}, value=0.05, space=_SPACE
    )

    # Assert
    assert updated is False
    loaded = load_champion_params(tag="4h", storage=storage)
    assert loaded == {"kelly_fraction": 0.3, "K_RANK": 3}


def test_update_champion_store_updates_when_strictly_better(
    storage: optuna.storages.BaseStorage,
) -> None:
    """기존 챔피언보다 높은 value는 레저를 갱신한다."""
    # Arrange
    update_champion_store(
        tag="4h", storage=storage, params={"kelly_fraction": 0.3, "K_RANK": 3}, value=0.10, space=_SPACE
    )

    # Act
    updated = update_champion_store(
        tag="4h", storage=storage, params={"kelly_fraction": 0.45, "K_RANK": 2}, value=0.25, space=_SPACE
    )

    # Assert
    assert updated is True
    loaded = load_champion_params(tag="4h", storage=storage)
    assert loaded == {"kelly_fraction": 0.45, "K_RANK": 2}


def test_update_champion_store_ignores_unknown_keys_not_in_space(
    storage: optuna.storages.BaseStorage,
) -> None:
    """space에 없는 키(예: 제거된 dead param)는 레저에 기록되지 않는다."""
    # Act
    update_champion_store(
        tag="4h",
        storage=storage,
        params={"kelly_fraction": 0.3, "K_RANK": 3, "DEAD_PARAM": 99},
        value=0.10,
        space=_SPACE,
    )

    # Assert
    loaded = load_champion_params(tag="4h", storage=storage)
    assert loaded == {"kelly_fraction": 0.3, "K_RANK": 3}


def test_update_champion_store_is_isolated_per_tag(
    storage: optuna.storages.BaseStorage,
) -> None:
    """tag가 다르면 독립된 레저로 분리된다 (4h 갱신이 1h에 영향 없음)."""
    # Act
    update_champion_store(
        tag="4h", storage=storage, params={"kelly_fraction": 0.3, "K_RANK": 3}, value=0.10, space=_SPACE
    )

    # Assert
    assert load_champion_params(tag="1h", storage=storage) is None
