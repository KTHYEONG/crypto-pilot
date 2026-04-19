import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.hmm_state_inferrer import HMMStateInferrer
from src.domain.futures.ml_pipeline.feature_engineering import SYSTEMIC_HMM_FEATURE_COLUMNS, HMM_SEMANTIC_PROB_COLUMNS
from config.settings import FUTURES_CACHE_DIR

warnings.filterwarnings("ignore")

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("test_hmm_only")

def test_hmm_isolated():
    _logger.info("=" * 85)
    _logger.info(" [TEST] HMM State Inferrer (Isolated)")
    _logger.info("=" * 85)
    
    # 0. Cleanup Cache
    if FUTURES_CACHE_DIR.exists():
        for f in FUTURES_CACHE_DIR.glob("*SyntheticTest*"):
            f.unlink()
    
    # 1. Create Synthetic Data
    n_samples = 1500
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="h", tz="UTC")
    
    feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
    _logger.info(f"Target Feature Columns ({len(feat_cols)}): {feat_cols}")
    
    data = np.random.randn(n_samples, len(feat_cols)) * 0.5 # Low base noise
    
    # Inject regime structure (Shifted after 500 for prediction window)
    # 500-750: Bull Trend (High return, Low vol)
    data[500:750, 0] += 3.0  # btc_trend_vol_adj_24h
    data[500:750, 1] -= 1.5  # realized_vol_regime
    
    # 750-1000: Crisis (Low return, High vol, High VoV)
    data[750:1000, 0] -= 4.0 
    data[750:1000, 1] += 4.0 
    data[750:1000, 6] += 3.0 
    
    # 1000-1250: Bear Trend (Low return, Medium vol)
    data[1000:1250, 0] -= 2.0
    data[1000:1250, 1] += 1.0
    
    # 1250-1500: Chop/Sideways (Zero return, Low vol)
    data[1250:1500, 0] *= 0.0
    data[1250:1500, 1] -= 1.5

    features_df = pd.DataFrame(data, index=dates, columns=feat_cols)
    returns_ser = features_df["btc_trend_vol_adj_24h"]
    
    # 2. Initialize HMM
    hmm_inferrer = HMMStateInferrer(n_states=4)
    
    # 3. Fit and Predict
    _logger.info("Fitting HMM on synthetic data...")
    is_end_idx = 1300 # IS/OOS split point
    
    probs = hmm_inferrer.fit_predict_systemic(
        features_df=features_df,
        returns_ser=returns_ser,
        is_end_idx=is_end_idx,
        symbol="SyntheticTest",
        tf="1h"
    )
    
    # 4. Verification
    _logger.info("\n" + "-" * 85)
    _logger.info(" [VERIFICATION] HMM Output")
    _logger.info("-" * 85)
    
    semantic_cols = list(HMM_SEMANTIC_PROB_COLUMNS)
    _logger.info(f"Probs shape: {probs.shape}")
    
    _logger.info(f"Probs slice (500-510):\n{probs.iloc[500:510]}")
    
    prob_vals = probs[semantic_cols]
    non_zero = probs[(prob_vals > 0).any(axis=1)]
    
    if not non_zero.empty:
        _logger.info(f"First non-zero row index: {non_zero.index[0]}")
    else:
        _logger.warning("All probability rows are zero!")
    
    # Check if semantic columns exist
    has_semantic = all(c in probs.columns for c in semantic_cols)
    _logger.info(f"Has Semantic Labels: {has_semantic}")
    
    if has_semantic:
        # Check average probabilities in the structural regions
        bull_p = probs.iloc[500:750][semantic_cols[0]].mean()
        bear_p = probs.iloc[1000:1250][semantic_cols[1]].mean()
        chop_p = probs.iloc[1250:1500][semantic_cols[2]].mean()
        crisis_p = probs.iloc[750:1000][semantic_cols[3]].mean()
        
        _logger.info(f" Average '{semantic_cols[0]}' p (500-750):   {bull_p:.4f}")
        _logger.info(f" Average '{semantic_cols[3]}' p (750-1000):    {crisis_p:.4f}")
        _logger.info(f" Average '{semantic_cols[1]}' p (1000-1250): {bear_p:.4f}")
        _logger.info(f" Average '{semantic_cols[2]}' p (1250-1500): {chop_p:.4f}")
        
        if bull_p > 0.35:
            _logger.info(" [SUCCESS] Bull trend captured.")
        if crisis_p > 0.35:
            _logger.info(" [SUCCESS] Crisis regime captured.")
            
    _logger.info("=" * 85)
    _logger.info(" [RESULT] HMM isolated test completed.")
    _logger.info("=" * 85)

if __name__ == "__main__":
    test_hmm_isolated()
