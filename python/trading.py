# Adım 0: Gerekli importları yap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from joblib import Parallel, delayed
from statsmodels.tsa.stattools import coint

# Adım 1: Veriyi yükle
file_path = "/workspaces/codespaces-jupyter/data/data.csv"
data = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)

print(f"Dataset shape: {data.shape}")
print(f"First date on data: {data.iloc[0].name}")
print(f"Last date on data: {data.iloc[-1].name}")
print("\n" + "=" * 50 + "\n")

# Adım 2: Veriyi böl
train_end_date = pd.Timestamp('2024-03-12')
test_end_date = pd.Timestamp('2025-03-12')
lookback_days = 365 # 1 yıl

train_data = data.loc[:train_end_date]
test_start_date = train_end_date - pd.Timedelta(days=lookback_days)
test_data = data.loc[test_start_date:test_end_date]

print("Training Data Shape:", train_data.shape)
print("Test Data Shape (including lookback):", test_data.shape)
print("\n" + "=" * 50 + "\n")

# Adım 3.1: Eşbütünleşik çiftleri bul

''' 
    Eşbütünleşik çift testi 
'''
def process_pair(i, j, keys, data, significance):
    S1 = data[keys[i]]
    S2 = data[keys[j]]
    score, pvalue, _ = coint(S1, S2)
    if pvalue < significance:
        model = sm.OLS(S1, sm.add_constant(S2)).fit()
        hedge_ratio = model.params.iloc[1]
        return (keys[i], keys[j], pvalue, hedge_ratio)
    return None

'''
    Paralelleştirilmiş eşbütünleşik çift bulma metodu
'''
def find_cointegrated_pairs_parallel(data, significance=0.05, n_jobs=-1):
    keys = data.columns
    n = len(keys)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(i, j, keys, data, significance)
        for i in range(n) for j in range(i+1, n)
    )
    
    pairs = [result for result in results if result is not None]
    return pairs

# Adım 3.2: Çiftler arasından en iyisini seç
def compute_half_life(spread):
    """
    Compute the half-life of mean reversion for a given spread.
    Uses the method described in statistical arbitrage literature:
      Δspread = α + β * lag(spread) + error, then half-life = -ln(2)/β.
    If β is non-negative (indicating no mean reversion), returns infinity.
    """
    spread_lag = spread.shift(1)
    delta_spread = spread.diff()
    df = pd.concat([spread, spread_lag, delta_spread], axis=1).dropna()
    df.columns = ['spread', 'lag_spread', 'delta_spread']
    
    model = sm.OLS(df['delta_spread'], sm.add_constant(df['lag_spread'])).fit()
    beta = model.params['lag_spread']
    if beta >= 0:
        return np.inf
    half_life = -np.log(2) / beta
    return half_life

def select_pair(idx, test_data, cointegration_window, cointegration_significance=0.05):
    """
    Improved pair selection:
    - Use the data in the window ending at index idx.
    - Run the cointegration tests over this window.
    - For each candidate pair, compute the half-life of mean reversion.
    - Select the candidate pair with the lowest half-life (i.e., fastest mean reversion).
      If no candidate has a finite (acceptable) half-life, fallback to the one with lowest p-value.
    """
    start_idx = max(0, idx - cointegration_window)
    window_data = test_data.iloc[start_idx:idx]
    candidate_pairs = find_cointegrated_pairs_parallel(window_data, significance=cointegration_significance)
    
    if candidate_pairs:
        candidates = []
        for candidate in candidate_pairs:
            stock1, stock2, pval, hedge_ratio = candidate
            spread = window_data[stock1] - hedge_ratio * window_data[stock2]
            hl = compute_half_life(spread)
            candidates.append((candidate, hl))
        # Filter out candidates with non-mean reverting behavior (hl = infinity) or extremely high half-life.
        valid_candidates = [(cand, hl) for (cand, hl) in candidates if hl != np.inf and hl < cointegration_window]
        
        if valid_candidates:
            # Choose candidate with lowest half-life; in case of ties, choose one with lower p-value.
            best_candidate = min(valid_candidates, key=lambda x: (x[1], x[0][2]))[0]
        else:
            # Fallback: choose candidate with the lowest p-value.
            best_candidate = min(candidate_pairs, key=lambda x: x[2])
        return best_candidate
    else:
        return None


# Adım 3.3: Seçilen iki stok için "pairs trading"i simüle et
def simulate_pair_trading_with_sizing(stock1, stock2, hedge_ratio, sim_data, lookback_days=60,
                                      entry_threshold=1, exit_threshold=0, position_fraction=0.1,
                                      initial_capital=10000):
    """
    Simulate pairs trading for a given segment using position sizing.
    
    Parameters:
      stock1, stock2: Stock symbols.
      hedge_ratio: Fixed hedge ratio.
      sim_data: DataFrame for the simulation segment.
      lookback_days: Rolling window length for computing statistics.
      entry_threshold, exit_threshold: Z-score thresholds.
      position_fraction: Fraction of current capital to commit on each trade.
      initial_capital: Capital available at the start of the segment.
      
    Returns:
      signals: List of trade signals with details.
      segment_pnl: Cumulative profit/loss (difference between ending and starting capital) for the segment.
      trade_pnls: List of individual trade returns.
      final_capital: Updated capital at the end of the segment.
    """
    # Compute the spread between the two stocks.
    spread = sim_data[stock1] - hedge_ratio * sim_data[stock2]
    roll_mean = spread.rolling(window=lookback_days).mean()
    roll_std = spread.rolling(window=lookback_days).std().replace(0, 1e-8)
    
    signals = []
    position = 0   # 0: no position, 1: long spread, -1: short spread
    entry_price = 0.0
    position_size = 0.0
    capital = initial_capital
    trade_pnls = []
    
    # Iterate over the simulation segment (starting after the initial lookback window)
    for t in range(lookback_days, len(spread)):
        current_zscore = (spread.iloc[t] - roll_mean.iloc[t]) / roll_std.iloc[t]
        current_date = spread.index[t]
        
        if position == 0:
            # If no position, check entry conditions
            if current_zscore > entry_threshold:
                # Enter short spread (short stock1, long stock2)
                position = -1
                entry_price = spread.iloc[t]
                # Avoid division by zero; use absolute entry price as basis.
                if abs(entry_price) < 1e-6:
                    continue
                position_size = (capital * position_fraction) / abs(entry_price)
                signals.append((current_date, 'Enter Short', current_zscore, position_size, entry_price))
            elif current_zscore < -entry_threshold:
                # Enter long spread (long stock1, short stock2)
                position = 1
                entry_price = spread.iloc[t]
                if abs(entry_price) < 1e-6:
                    continue
                position_size = (capital * position_fraction) / abs(entry_price)
                signals.append((current_date, 'Enter Long', current_zscore, position_size, entry_price))
        else:
            # Check exit condition
            if (position == -1 and current_zscore < exit_threshold) or (position == 1 and current_zscore > exit_threshold):
                exit_price = spread.iloc[t]
                # Calculate trade return (P/L in currency)
                trade_return = (exit_price - entry_price) * position * position_size
                capital += trade_return
                trade_pnls.append(trade_return)
                signals.append((current_date, 'Exit', current_zscore, position_size, exit_price, trade_return))
                # Close position
                position = 0
                entry_price = 0.0
                position_size = 0.0
                
    segment_pnl = capital - initial_capital
    return signals, segment_pnl, trade_pnls, capital

# Adım 4:
def simulate_dynamic_pair_trading_with_sizing(data, 
                                              initial_lookback=252, 
                                              rebalance_interval=15, 
                                              loss_threshold=-np.inf,
                                              sim_lookback_days=60, 
                                              entry_threshold=1, 
                                              exit_threshold=0, 
                                              cointegration_significance=0.05, 
                                              cointegration_window=252,
                                              position_fraction=0.1,
                                              starting_capital=10000):
    """
    Simulate dynamic pair trading over multiple segments using position sizing.
    
    For each segment, the simulation function uses the current capital.
    After each segment, the capital is updated.
    The pair can be re-selected at the end of each segment.
    """
    overall_signals = []
    overall_trade_pnls = []
    
    num_days = len(data)
    current_index = initial_lookback
    capital = starting_capital

    # Initial pair selection using data window ending at current_index.
    current_pair = select_pair(current_index, data, cointegration_window, cointegration_significance)
    if current_pair is None:
        print("Simülasyonu başlatmak için gerekli çift bulunamadı.")
        return None
    
    current_stock1, current_stock2, current_pvalue, current_hedge_ratio = current_pair
    print(f"Başlangıç değeri seçildi: {current_stock1} & {current_stock2}")

    # Dynamic simulation loop (each segment rebalances after a fixed interval)
    while current_index < num_days:
        segment_end_index = min(current_index + rebalance_interval, num_days)
        # Use the last sim_lookback_days for simulation; ensure there are enough data points.
        segment_data = data.iloc[current_index - sim_lookback_days: segment_end_index]
        
        # Run simulation for the segment with the current capital.
        signals, seg_pnl, trade_pnls, updated_capital = simulate_pair_trading_with_sizing(
            current_stock1, current_stock2, current_hedge_ratio,
            segment_data, lookback_days=sim_lookback_days,
            entry_threshold=entry_threshold, exit_threshold=exit_threshold,
            position_fraction=position_fraction,
            initial_capital=capital
        )
        overall_signals.extend(signals)
        overall_trade_pnls.extend(trade_pnls)
        print(f"Segment {current_index} to {segment_end_index}: pnl = {seg_pnl:.2f}, Capital updated to: {updated_capital:.2f}")
        
        # Update capital for the next segment.
        capital = updated_capital
        
        # Rebalance: re-select pair at the end of the segment.
        new_pair = select_pair(current_index, data, cointegration_window, cointegration_significance)
        if new_pair is not None:
            current_stock1, current_stock2, current_pvalue, current_hedge_ratio = new_pair
            print(f"Rebalanced pair to: {current_stock1} & {current_stock2} at index {segment_end_index}")
        else:
            print(f"No new candidate pair found at index {segment_end_index}. Continuing with current pair.")
        
        current_index = segment_end_index

    percent_return = ((capital / starting_capital) - 1) * 100
    results = {
        'signals': overall_signals,
        'final_capital': capital,
        'percent_return': percent_return,
        'trade_pnls': overall_trade_pnls
    }
    return results

# Run the dynamic simulation with position sizing on in-sample (training) data.
dynamic_results = simulate_dynamic_pair_trading_with_sizing(
    data = train_data[-504:], 
    initial_lookback=252, 
    rebalance_interval=15, 
    loss_threshold=-np.inf, 
    sim_lookback_days=30, 
    entry_threshold=1, 
    exit_threshold=0, 
    cointegration_significance=0.05, 
    cointegration_window=60,
    position_fraction=0.1,
    starting_capital=10000
)

if dynamic_results is not None:
    print("\nDynamic Simulation Final Capital:", dynamic_results['final_capital'])
    print("Percentage Return: {:.2f}%".format(dynamic_results['percent_return']))
    print("Number of trades:", len(dynamic_results['trade_pnls']))
    for signal in dynamic_results['signals']:
        print(signal)

