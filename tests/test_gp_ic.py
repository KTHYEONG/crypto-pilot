import logging
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트 설정
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.opt_config import OPT_FUTURES_CONFIG, get_quarterly_window
from src.domain.futures.ml_pipeline.ml_pipeline_runner import run_ml_pipeline_for_universe

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_gp_ic_fast")


def run_numerical_benchmark():
    """1. 수치적 벡터화 성능 벤치마크 (기존 test_fast_ic 로직)"""
    _logger.info("\n" + "=" * 60)
    _logger.info(" [PART 1] Numerical IC Calculation Benchmark")
    _logger.info("=" * 60)

    n_days, n_syms = 500, 10
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=n_days, freq="h"), [f"SYM_{i}" for i in range(n_syms)]],
        names=["datetime", "symbol"],
    )
    df = pd.DataFrame(
        {"val1": np.random.randn(len(idx)), "target": np.random.randn(len(idx))}, index=idx
    )

    # A. 기존 Loop 방식
    t0 = time.time()
    ic_loop = []
    for _dt, g in df.groupby(level="datetime"):
        if len(g) >= 3:
            ic_loop.append(g["val1"].corr(g["target"], method="spearman"))
    t_loop = time.time() - t0

    # B. 개선된 Vectorized 방식
    t1 = time.time()
    u_c = df["val1"].unstack()
    u_t = df["target"].unstack()
    ic_vec = u_c.rank(axis=1).corrwith(u_t.rank(axis=1), axis=1).dropna().tolist()
    t_vec = time.time() - t1

    speedup = t_loop / t_vec if t_vec > 0 else 0
    _logger.info(f" Loop-based IC: {t_loop:.4f}s")
    _logger.info(f" Vectorized IC: {t_vec:.4f}s")
    _logger.info(f" Result Match:  {math.isclose(np.mean(ic_loop), np.mean(ic_vec), rel_tol=1e-7)}")
    _logger.info(f" SPEEDUP:      {speedup:.1f}x")


def run_gp_pipeline_validation():
    """2. 실제 GP 파이프라인 통합 검증 (기존 test_gp_ic_fast 로직)"""
    _logger.info("\n" + "=" * 60)
    _logger.info(" [PART 2] GP ML Pipeline Integration Validation")
    _logger.info("=" * 60)

    res = get_quarterly_window()
    fetch_start, start, is_end, end = res
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
    tf = "1h"

    cfg = dict(OPT_FUTURES_CONFIG)
    cfg["FUTURES_ML_GP_GENERATIONS"] = 3
    cfg["FUTURES_ML_GP_POPULATION"] = 300
    cfg["FUTURES_ML_IC_FDR_Q"] = 0.3
    cfg["FUTURES_ML_IC_FILTER_USE_HAC"] = False

    # 캐시 무시를 위해 캐시 키에 영향을 줄 수 있는 설정 변경 (또는 직접 삭제 로직 추가 가능)
    # 여기서는 순수 계산 검증이므로 기존 캐시가 있다면 활용함

    t0 = time.time()
    ml_out = run_ml_pipeline_for_universe(
        test_symbols,
        tf,
        fetch_start,
        end,
        cfg,
        workers=4,
        n_jobs=4,
        is_end_date=is_end,
        is_start_date=start,
    )
    total_time = time.time() - t0

    best_fitness = ml_out.alpha_panel.attrs.get("best_fitness", 0.0)
    filter_meta = ml_out.alpha_panel.attrs.get("alpha_component_filter", {})

    _logger.info(f" Pipeline Execution Time: {total_time:.2f}s")
    _logger.info(f" Best Fitness (IS):       {best_fitness:.6f}")
    _logger.info(f" Surviving Alphas:       {filter_meta.get('n_surviving', 0)} / 15.0")
    _logger.info("=" * 60)


if __name__ == "__main__":
    run_numerical_benchmark()
    run_gp_pipeline_validation()
