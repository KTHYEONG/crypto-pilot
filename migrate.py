import os
import shutil
from pathlib import Path
import re

def main():
    root = Path("src")
    
    # 1. Define new directories
    dirs = [
        "src/core/exchange",
        "src/core/indicators",
        "src/core/optimization",
        "src/core/utils",
        "src/strategy_base",
        "src/domain/spot",
        "src/domain/futures",
        "src/execution"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        init_file = Path(d) / "__init__.py"
        init_file.touch()
        
    # Mapping old file path to new file path
    move_map = {}
    
    # Core Exchange
    move_map["src/spot_strategy/upbit_client.py"] = "src/core/exchange/upbit_client.py"
    move_map["src/futures_strategy/binance_client.py"] = "src/core/exchange/binance_client.py"
    
    # Core Indicators
    move_map["src/common/indicators_advanced.py"] = "src/core/indicators/indicators_advanced.py"
    move_map["src/futures_strategy/indicators_advanced_futures.py"] = "src/core/indicators/indicators_advanced_futures.py"
    move_map["src/spot_strategy/signals/numpy_ops.py"] = "src/core/indicators/numpy_ops_spot.py"
    move_map["src/futures_strategy/signals/numpy_ops.py"] = "src/core/indicators/numpy_ops_futures.py"
    
    # Core Utils
    move_map["src/common/secure_config.py"] = "src/core/utils/secure_config.py"
    move_map["src/common/components.py"] = "src/core/utils/components.py"
    move_map["src/common/utils.py"] = "src/core/utils/utils.py"
    move_map["src/common/cloud_optimizer.py"] = "src/core/utils/cloud_optimizer.py"
    
    # Core Optimization
    move_map["src/optimization/opt_utils.py"] = "src/core/optimization/opt_utils.py"
    
    # Execution
    move_map["src/spot_strategy/opt_spot.py"] = "src/execution/opt_main_spot.py"
    move_map["src/futures_strategy/opt_futures.py"] = "src/execution/opt_main_futures.py"
    move_map["src/spot_strategy/spot_bot.py"] = "src/execution/trader_spot.py"
    move_map["src/futures_strategy/real_trader_futures.py"] = "src/execution/trader_futures.py"
    
    # Strategy Base
    move_map["src/strategy/base/__init__.py"] = "src/strategy_base/__init__.py"
    move_map["src/strategy/base/core.py"] = "src/strategy_base/core.py"
    move_map["src/strategy/base/ultimate.py"] = "src/strategy_base/ultimate.py"
    
    # Now domain files. We will just recursively move everything else in spot_strategy and futures_strategy
    # except the ones already mapped.
    
    for path in Path("src/spot_strategy").rglob("*.py"):
        old_str = str(path)
        if old_str in move_map: continue
        # domain/spot
        rel = path.relative_to("src/spot_strategy")
        new_str = f"src/domain/spot/{rel}"
        move_map[old_str] = new_str
        
    for path in Path("src/futures_strategy").rglob("*.py"):
        old_str = str(path)
        if old_str in move_map: continue
        # domain/futures
        rel = path.relative_to("src/futures_strategy")
        new_str = f"src/domain/futures/{rel}"
        move_map[old_str] = new_str
        
    # Execute moves
    for old, new in move_map.items():
        old_p = Path(old)
        new_p = Path(new)
        if old_p.exists():
            os.makedirs(new_p.parent, exist_ok=True)
            shutil.move(old, new)
            
    # Clean up empty old directories
    for d in ["src/spot_strategy", "src/futures_strategy", "src/common", "src/strategy/base", "src/strategy", "src/optimization"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            
    # Import Rewrite Rules
    # order matters! More specific first.
    import_replacements = [
        ("src.spot_strategy.upbit_client", "src.core.exchange.upbit_client"),
        ("src.futures_strategy.binance_client", "src.core.exchange.binance_client"),
        
        ("src.common.indicators_advanced", "src.core.indicators.indicators_advanced"),
        ("src.futures_strategy.indicators_advanced_futures", "src.core.indicators.indicators_advanced_futures"),
        ("src.spot_strategy.signals.numpy_ops", "src.core.indicators.numpy_ops_spot"),
        ("src.futures_strategy.signals.numpy_ops", "src.core.indicators.numpy_ops_futures"),
        
        ("src.common.secure_config", "src.core.utils.secure_config"),
        ("src.common.components", "src.core.utils.components"),
        ("src.common.utils", "src.core.utils.utils"),
        ("src.common.cloud_optimizer", "src.core.utils.cloud_optimizer"),
        
        ("src.optimization.opt_utils", "src.core.optimization.opt_utils"),
        
        ("src.strategy.base", "src.strategy_base"),
        
        ("src.spot_strategy.opt_spot", "src.execution.opt_main_spot"),
        ("src.spot_strategy.spot_bot", "src.execution.trader_spot"),
        
        ("src.futures_strategy.opt_futures", "src.execution.opt_main_futures"),
        ("src.futures_strategy.real_trader_futures", "src.execution.trader_futures"),
        
        ("src.spot_strategy", "src.domain.spot"),
        ("src.futures_strategy", "src.domain.futures"),
        ("src.common", "src.core.utils"), # Catch-all for any missed common imports
    ]
    
    # Process all py files in project (including tests)
    all_py_files = list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py"))
    
    for f in all_py_files:
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
            
        new_content = content
        for old_imp, new_imp in import_replacements:
            new_content = new_content.replace(old_imp, new_imp)
            
        if content != new_content:
            with open(f, "w", encoding="utf-8") as file:
                file.write(new_content)
                
    print("Migration complete.")

if __name__ == "__main__":
    main()
