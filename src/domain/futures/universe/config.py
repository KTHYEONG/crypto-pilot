"""Universe configuration and deterministic hashing utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _canonicalize(value: Any) -> Any:
    """Convert nested values into deterministic JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _sha256_json(payload: dict[str, Any]) -> str:
    canonical_payload = _canonicalize(payload)
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage2Config:
    """Data quality gates.

    Notes:
        min_is_bars_4h = 9개월 x 30일 x 6 bars/day x 80% coverage = 1,296 bars.
        Stage 5의 listing_age_days >= 90이 단기 상장 종목을 걸러주므로,
        Stage 2는 롤링 지표(ADV-30d, vol-30d, coverage-60d) 계산에 충분한 기간만
        요구한다. 백테스트 IS 윈도우 충분성은 optimizer fold에서 별도 검증.

    """

    min_is_coverage: float = 0.80
    min_is_bars_4h: int = 1_296
    min_coverage_60d: float = 0.95
    max_zero_volume_bars_60d: int = 1
    max_gap_bars: int = 200
    max_gap_count: int = 1
    max_frozen_bars_60d: int = 4


@dataclass(frozen=True, slots=True)
class Stage3Config:
    """Liquidity and execution feasibility gates."""

    min_adv_usdt_median: float = 25_000_000.0
    # 실측 분포 기반: p99 ≈ 1.084e-9, 임계값 = p99 × 1.5 ≈ 1.627e-9
    max_amihud_30d: float = 1.63e-9
    max_clip_to_adv: float = 0.005
    screening_tier: str = "mid"
    screening_clip_usdt_by_tier: dict[str, float] = field(
        default_factory=lambda: {
            "seed": 1_000.0,
            "small": 5_000.0,
            "mid": 10_000.0,
            "large": 25_000.0,
            "xlarge": 50_000.0,
        }
    )
    capacity_clip_usdt_list: tuple[float, ...] = (50_000.0, 100_000.0)


@dataclass(frozen=True, slots=True)
class Stage4Config:
    """Execution-cost model gates."""

    max_execution_cost_bps: float = 50.0
    default_taker_fee_bps: float = 5.0
    default_half_spread_bps: float = 1.0
    spread_source_switch_date: str = "2020-01-01"
    pre2020_half_spread_bps: float = 2.5
    post2020_half_spread_bps: float = 1.0
    default_impact_coef_bps: float = 18.0


@dataclass(frozen=True, slots=True)
class Stage5Config:
    """Risk-event and anomaly gates.

    Notes:
        vol_30d는 4h 바 기준 연율화 변동성 (std * sqrt(6*365)).
        median ~0.75, p90 ~1.81 수준의 스케일.
        min=0.05(5% 연율, 사실상 거래 없는 코인 제거),
        max=4.0(400% 연율, 극단적 meme 제거).

        funding_sign_flip_min_abs: 부호 반전이 이상치로 인정받으려면
        양쪽 모두 절대값이 이 임계값 이상이어야 한다.
        +0.001% → -0.001% 수준의 중립 진동은 이상치에서 제외.

    """

    min_listing_age_days: int = 90
    min_vol_30d: float = 0.05   # 5% annualized — 거래 없는 죽은 코인 제거
    max_vol_30d: float = 4.0    # 400% annualized — 극단적 meme/junk 제거
    max_abs_funding_z: float = 2.5
    enable_funding_sign_flip: bool = True
    funding_sign_flip_min_abs: float = 0.001  # |funding| > threshold 양쪽 모두여야 flip 이상치
    funding_sign_flip_columns: tuple[str, ...] = (
        "funding_sign_flip_1d",
        "funding_sign_reversal_1d",
        "funding_sign_change_1d",
    )
    funding_prev_rate_column: str = "funding_rate_8h_prev"


@dataclass(frozen=True, slots=True)
class Stage6Config:
    """Membership selection controls."""

    k_in: int = 20
    k_out: int = 35
    min_dwell_days: int = 90
    anchor_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    basket_ref: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    basket_weights: tuple[float, ...] = (0.45, 0.25, 0.08)
    corr_cluster_threshold: float = 0.70  # 상관계수 >= threshold → 동일 클러스터로 연결


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """Top-level universe configuration with deterministic hash."""

    schema_version: int = 1
    timeframe: str = "4h"
    ledger_confidence: str = "reconstructed"
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    stage5: Stage5Config = field(default_factory=Stage5Config)
    stage6: Stage6Config = field(default_factory=Stage6Config)

    def to_payload(self) -> dict[str, Any]:
        """Return canonical payload for persistence and hashing."""
        return asdict(self)

    def config_hash(self) -> str:
        """Compute deterministic SHA256 hash for reproducibility."""
        return _sha256_json(self.to_payload())


def hash_config(config: UniverseConfig) -> str:
    """Return deterministic config hash."""
    return config.config_hash()
