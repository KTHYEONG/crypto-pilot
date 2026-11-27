from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS, MIRROR_DEFAULT_DATASETS

_DOWNLOAD_WORKERS_RANGE = range(1, 9)


@dataclass(frozen=True, slots=True)
class MirrorConfig:
    root: Path
    datasets: tuple[str, ...] = MIRROR_DEFAULT_DATASETS
    drive_budget_gib: int = 250
    drive_free_floor_gib: int = 1024
    local_free_floor_gib: int = 30
    outbox_cap_gib: int = 2
    close_grace_days: int = 5
    listing_max_age_hours: int = 12
    download_workers: int = 4
    verbatim_max_fraction: float = 0.01
    verbatim_min_sources: int = 20
    stalled_daily_days: int = 7
    stalled_monthly_days: int = 45

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        unknown = [name for name in self.datasets if name not in MIRROR_DATASET_SPECS]
        if unknown:
            raise ValueError(f"unknown mirror datasets: {unknown}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("duplicate mirror datasets")
        ordered = tuple(name for name in MIRROR_DATASET_SPECS if name in self.datasets)
        object.__setattr__(self, "datasets", ordered)
        positive = {
            "drive_budget_gib": self.drive_budget_gib,
            "drive_free_floor_gib": self.drive_free_floor_gib,
            "local_free_floor_gib": self.local_free_floor_gib,
            "outbox_cap_gib": self.outbox_cap_gib,
            "listing_max_age_hours": self.listing_max_age_hours,
            "verbatim_min_sources": self.verbatim_min_sources,
            "stalled_daily_days": self.stalled_daily_days,
            "stalled_monthly_days": self.stalled_monthly_days,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if self.close_grace_days < 0:
            raise ValueError(f"close_grace_days must be >= 0, got {self.close_grace_days}")
        if self.download_workers not in _DOWNLOAD_WORKERS_RANGE:
            raise ValueError(f"download_workers must be within 1..8, got {self.download_workers}")
        if not 0.0 <= self.verbatim_max_fraction <= 1.0:
            raise ValueError(f"verbatim_max_fraction must be within [0, 1], got {self.verbatim_max_fraction}")

    @property
    def outbox_objects_dir(self) -> Path:
        return self.root / "outbox" / "objects"

    @property
    def outbox_index_dir(self) -> Path:
        return self.root / "outbox" / "index"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def status_dir(self) -> Path:
        return self.root / "_status"
