import os
import sys

def prepend_imports(filepath, imports):
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    out_lines = []
    inserted = False
    for line in lines:
        out_lines.append(line)
        if line.startswith('from __future__ import annotations') and not inserted:
            out_lines.append(imports)
            inserted = True
            
    if not inserted:
        out_lines.insert(0, imports)
        
    with open(filepath, 'w') as f:
        f.write('\n'.join(out_lines))

prepend_imports('src/domain/futures/opt_futures_utils/objective.py', """
from .signal_cache import _dataset_fingerprint_from_df, _build_signal_cache_key, get_or_compute_signals, _SignalCacheKey, _cache_lock, _arrays_cache, _ARRAYS_CACHE_MAXSIZE
from .data_utils import _dataframe_to_symbol_arrays, _build_aligned_2d_from_prebuilt, _segment_with_context
from .oos_evaluator import evaluate_symbol_fold
""")

prepend_imports('src/domain/futures/opt_futures_utils/oos_evaluator.py', """
from .signal_cache import _dataset_fingerprint_from_df, _build_signal_cache_key, get_or_compute_signals, _SignalCacheKey, _cache_lock, _arrays_cache, _ARRAYS_CACHE_MAXSIZE
from .data_utils import _segment_with_context, align_data_for_2d_engine, _dataframe_to_symbol_arrays, _build_aligned_2d_from_prebuilt
from .objective import calc_tail_ratio_from_equity, _log_tw_from_ret_pct
""")
