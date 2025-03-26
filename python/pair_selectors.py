# pair_selectors.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import coint

def process_pair(i, j, keys, data, significance):
    S1 = data[keys[i]]
    S2 = data[keys[j]]
    score, pvalue, _ = coint(S1, S2)
    if pvalue < significance:
        model = sm.OLS(S1, sm.add_constant(S2)).fit()
        hedge_ratio = model.params.iloc[1]
        return (keys[i], keys[j], pvalue, hedge_ratio)
    return None

def find_cointegrated_pairs_parallel(data, significance=0.05, n_jobs=-1):
    keys = data.columns
    n = len(keys)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(i, j, keys, data, significance)
        for i in range(n) for j in range(i+1, n)
    )
    pairs = [result for result in results if result is not None]
    return pairs

def get_candidate_pairs(window_data, significance=0.05, n_jobs=-1):
    """
    Compute and return candidate cointegrated pairs for the given window data.
    """
    return find_cointegrated_pairs_parallel(window_data, significance=significance, n_jobs=n_jobs)

def compute_half_life(series):
    """
    Compute the half-life of mean reversion for a given series.
    Uses the method: Δseries = α + β * lag(series) + error, then half-life = -ln(2)/β.
    Returns infinity if beta >= 0.
    """
    series_lag = series.shift(1)
    delta_series = series.diff()
    df = pd.concat([series, series_lag, delta_series], axis=1).dropna()
    df.columns = ['series', 'lag_series', 'delta_series']
    
    model = sm.OLS(df['delta_series'], sm.add_constant(df['lag_series'])).fit()
    beta = model.params['lag_series']
    if beta >= 0:
        return np.inf
    half_life = -np.log(2) / beta
    return half_life

def choose_best_pair(candidate_pairs, window_data, cointegration_window,
                     method='spread', combine_std=False, trader_func=None, sim_period=None):
    """
    Given candidate pairs (pre-computed) and the window data, select the best candidate.
    
    If trader_func and sim_period are provided, the function simulates trading for each candidate
    over the last sim_period days of window_data and returns the candidate with the highest pnl.
    
    Otherwise, for each candidate, a metric series is computed using:
      - 'spread' method: metric = S1 - hedge_ratio * S2
      - 'ratio' method: metric = S1 / S2
    The half-life is computed from the metric series and, if combine_std is True,
    a composite score = half_life / std is used. The candidate with the lowest composite score 
    (or lowest half-life if combine_std is False) is chosen. In case of no valid score,
    the candidate with the lowest p-value is selected.
    
    Parameters:
      candidate_pairs: List of candidate pairs (each as (stock1, stock2, pvalue, hedge_ratio)).
      window_data: DataFrame with price data for the current window.
      cointegration_window: The length of the window (used as a threshold for reasonable scores).
      method: 'spread' (default) or 'ratio' to choose the metric.
      combine_std: If True, use composite score = half_life / std.
      trader_func: Optional function that simulates trading for a candidate pair. It should
                   accept (stock1, stock2, hedge_ratio, sim_data) and return a pnl value.
      sim_period: Number of days to simulate trading if trader_func is provided.
    
    Returns:
      The best candidate pair tuple (stock1, stock2, pvalue, hedge_ratio) or None.
    """
    # If a trader function is provided, simulate each candidate's trading performance.
    if trader_func is not None and sim_period is not None:
    # Include the required lookback period (e.g., 60 days) along with the simulation period.
        lookback_days = 60  # Adjust or pass this value as needed.
        total_period = sim_period + lookback_days
        sim_data = window_data.iloc[-total_period:] if len(window_data) >= total_period else window_data.copy()
        candidate_results = []
        for candidate in candidate_pairs:
            stock1, stock2, pval, hedge_ratio = candidate
            pnl = trader_func(stock1, stock2, hedge_ratio, sim_data, lookback_days=lookback_days)
            candidate_results.append((candidate, pnl))
        best_candidate = max(candidate_results, key=lambda x: x[1])[0]
        return best_candidate
    else:
        # Otherwise, use the statistical metric approach.
        candidates = []
        for candidate in candidate_pairs:
            stock1, stock2, pval, hedge_ratio = candidate
            if method == 'spread':
                metric_series = window_data[stock1] - hedge_ratio * window_data[stock2]
            elif method == 'ratio':
                metric_series = window_data[stock1] / window_data[stock2]
            else:
                raise ValueError("Invalid method. Choose 'spread' or 'ratio'.")
            
            hl = compute_half_life(metric_series)
            if combine_std:
                std_val = metric_series.std()
                composite_score = hl / std_val if std_val != 0 else np.inf
            else:
                composite_score = hl
            
            candidates.append((candidate, composite_score))
        
        valid_candidates = [(cand, score) for (cand, score) in candidates if np.isfinite(score) and score < cointegration_window]
        
        if valid_candidates:
            best_candidate = min(valid_candidates, key=lambda x: (x[1], x[0][2]))[0]
        else:
            best_candidate = min(candidate_pairs, key=lambda x: x[2])
        return best_candidate

from traders import simulate_pair_trading_with_sizing

if __name__ == "__main__":
    # Example test for pair selection:
    from data_reader import load_data
    file_path = "/workspaces/codespaces-jupyter/data/data.csv"
    data = load_data(file_path)
    idx = len(data) - 1
    start_idx = max(0, idx - 252)
    window_data = data.iloc[start_idx:idx]
    candidates = get_candidate_pairs(window_data, significance=0.05)
    
    # Using the 'spread' method with composite scoring.
    best_pair_spread = choose_best_pair(candidates, window_data, cointegration_window=252,
                                        method='spread', combine_std=True)
    print("Selected pair using spread (statistical):", best_pair_spread)
    
    # Alternatively, using the 'ratio' method.
    best_pair_ratio = choose_best_pair(candidates, window_data, cointegration_window=252,
                                       method='ratio', combine_std=True)
    print("Selected pair using ratio (statistical):", best_pair_ratio)
    
    
    best_pair_trader = choose_best_pair(candidates, window_data, cointegration_window=252,
                                        method='spread', combine_std=True,
                                        trader_func=simulate_pair_trading_with_sizing, sim_period=30)
    print("Selected pair using trader simulation:", best_pair_trader)
