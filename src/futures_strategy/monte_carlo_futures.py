
import numpy as np


class FuturesMonteCarloSimulator:
    """
    Block Bootstrap Monte Carlo Simulator for Futures Trading Strategies.
    
    Key Improvements over Standard Bootstrap:
    - Preserves time-series dependencies (autocorrelation)
    - Maintains volatility clustering patterns
    - Captures realistic winning/losing streaks
    - Provides more conservative (realistic) risk estimates
    
    References:
    - Politis & Romano (1994): "The Stationary Bootstrap"
    - Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
    """
    
    def __init__(self, trades):
        """
        :param trades: List of percentage returns from trades
        """
        self.trades = np.asarray(trades, dtype=np.float64)

    def _calculate_optimal_block_size(self, n_trades):
        """
        Heuristic for expected block size.
        Uses sqrt(N) with safety bounds to balance dependency capture and diversity.
        """
        if n_trades <= 10:
            return 3

        block_size = int(round(np.sqrt(n_trades)))
        block_size = max(3, block_size)
        block_size = min(block_size, 20)

        # Ensure enough blocks remain in one path (at least ~8 blocks)
        block_size = min(block_size, max(3, n_trades // 8))
        return int(block_size)

    def _moving_block_bootstrap_sample(self, trades_arr, n_trades, block_size, rng):
        """
        Moving Block Bootstrap: fixed-length contiguous blocks.
        """
        extended_trades = np.concatenate([trades_arr, trades_arr[:block_size]])
        n_blocks_needed = int(np.ceil(n_trades / block_size))
        start_indices = rng.integers(0, len(trades_arr), size=n_blocks_needed)

        sample = np.empty(n_blocks_needed * block_size, dtype=np.float64)
        cursor = 0
        for start_idx in start_indices:
            sample[cursor:cursor + block_size] = extended_trades[start_idx:start_idx + block_size]
            cursor += block_size

        return sample[:n_trades]

    def _stationary_bootstrap_sample(self, trades_arr, n_trades, avg_block_size, rng):
        """
        Non-circular Stationary Bootstrap:
        random-length contiguous blocks, but no wrap-around seam from tail->head.
        """
        p = 1.0 / max(avg_block_size, 1)
        sample = np.empty(n_trades, dtype=np.float64)
        idx = rng.integers(0, n_trades)

        for i in range(n_trades):
            if i == 0 or rng.random() < p:
                idx = rng.integers(0, n_trades)
            else:
                # Avoid circular seam; restart when reaching the end.
                if idx >= n_trades - 1:
                    idx = rng.integers(0, n_trades)
                else:
                    idx += 1
            sample[i] = trades_arr[idx]

        return sample

    def run(
        self,
        n_simulations=10000,
        initial_balance=1_000_000.0,
        use_block_bootstrap=True,
        method=None,
        block_size=None,
        random_state=None,
    ):
        """
        Run Monte Carlo Simulation with configurable bootstrap methods.

        :param n_simulations: Number of simulation paths (default: 10,000)
        :param initial_balance: Starting capital
        :param use_block_bootstrap: Backward compatibility flag
        :param method: 'stationary_block' (default), 'moving_block', or 'iid'
        :param block_size: Optional block size override for block methods
        :param random_state: Optional int seed for reproducibility
        :return: Dictionary with statistical results
        """
        if method is None:
            method = "stationary_block" if use_block_bootstrap else "iid"

        valid_methods = {"stationary_block", "moving_block", "iid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose one of {sorted(valid_methods)}.")

        trades_arr = self.trades[np.isfinite(self.trades)]
        n_trades = len(trades_arr)
        if n_trades < 5:
            return {
                "prob_profit": 0.0,
                "mean_return_pct": 0.0,
                "median_return_pct": 0.0,
                "worst_case_mdd": 0.0,
                "lower_bound_95": 0.0,
                "upper_bound_95": 0.0,
                "block_size_used": 0,
                "method_used": method,
                "n_trades_used": int(n_trades),
                "simulations": int(n_simulations),
            }

        if method == "iid":
            block_size_used = 1
        else:
            if block_size is None:
                block_size_used = self._calculate_optimal_block_size(n_trades)
            else:
                block_size_used = int(block_size)
            # Hard cap keeps enough effective blocks for diversity.
            block_size_used = max(3, min(block_size_used, max(3, n_trades // 8)))

        rng = np.random.default_rng(random_state)

        simulation_final_balances = []
        simulation_mdds = []

        for _ in range(n_simulations):
            if method == "stationary_block":
                sampled_rets = self._stationary_bootstrap_sample(
                    trades_arr, n_trades, block_size_used, rng
                )
            elif method == "moving_block":
                sampled_rets = self._moving_block_bootstrap_sample(
                    trades_arr, n_trades, block_size_used, rng
                )
            else:
                sampled_rets = rng.choice(trades_arr, size=n_trades, replace=True)

            cumulative_ret_pct = np.cumsum(sampled_rets, dtype=np.float64)
            equity_curve = initial_balance * (1 + cumulative_ret_pct / 100.0)
            equity_curve = np.insert(equity_curve, 0, initial_balance)
            simulation_final_balances.append(float(equity_curve[-1]))

            running_max = np.maximum.accumulate(equity_curve)
            with np.errstate(divide="ignore", invalid="ignore"):
                drawdown = (equity_curve - running_max) / running_max * 100
                mdd = float(np.min(drawdown))
                if not np.isfinite(mdd):
                    mdd = 0.0
            simulation_mdds.append(mdd)

        simulation_final_balances = np.array(simulation_final_balances)
        simulation_mdds = np.array(simulation_mdds)

        sim_returns_pct = (simulation_final_balances - initial_balance) / initial_balance * 100

        prob_profit = np.mean(sim_returns_pct > 0) * 100
        mean_return = np.mean(sim_returns_pct)
        median_return = np.median(sim_returns_pct)

        lower_bound = np.percentile(sim_returns_pct, 2.5)
        upper_bound = np.percentile(sim_returns_pct, 97.5)

        worst_case_mdd = np.percentile(simulation_mdds, 5)

        return {
            "prob_profit": float(prob_profit),
            "mean_return_pct": float(mean_return),
            "median_return_pct": float(median_return),
            "worst_case_mdd": float(worst_case_mdd),
            "lower_bound_95": float(lower_bound),
            "upper_bound_95": float(upper_bound),
            "block_size_used": int(block_size_used),
            "method_used": method,
            "n_trades_used": int(n_trades),
            "simulations": int(n_simulations),
        }
