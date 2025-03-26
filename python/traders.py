# traders.py
import numpy as np
import pandas as pd

def simulate_pair_trading_with_sizing(stock1, stock2, hedge_ratio, sim_data, 
                                      method="spread", 
                                      lookback_days=60,
                                      entry_threshold=1, exit_threshold=0.5, 
                                      position_fraction=0.1,
                                      initial_capital=10000):
    """
    Simulate pairs trading for a given segment using position sizing.
    
    Parameters:
      stock1, stock2: column names in sim_data
      hedge_ratio: used for spread calculation (or estimated externally)
      sim_data: a DataFrame with datetime index and price columns for each stock
      method: one of "spread", "ratio", or "bollinger_bands"
      lookback_days: rolling window to compute mean and std (or bands)
      entry_threshold: for 'spread' and 'ratio', the zscore threshold for entry; 
                       for 'bollinger_bands', the multiplier for the bands
      exit_threshold: for 'spread' and 'ratio', exit when abs(zscore) < this value;
                      for 'bollinger_bands', exit when indicator returns within bands
      position_fraction: fraction of capital allocated per trade
      initial_capital: starting capital
      
    Returns:
      signals: list of tuples with trade actions and parameters
      segment_pnl: profit/loss for the segment
      trade_pnls: list of individual trade PnL values
      capital: updated capital after simulation
    """
    # Choose the indicator based on method.
    if method in ["spread", "bollinger_bands"]:
        # Spread indicator
        indicator = sim_data[stock1] - hedge_ratio * sim_data[stock2]
    elif method == "ratio":
        # Ratio indicator
        indicator = sim_data[stock1] / sim_data[stock2]
    else:
        raise ValueError("Invalid method. Choose from 'spread', 'ratio', 'bollinger_bands'.")
    
    # Calculate rolling statistics.
    roll_mean = indicator.rolling(window=lookback_days).mean()
    roll_std = indicator.rolling(window=lookback_days).std().replace(0, 1e-8)
    
    signals = []
    position = 0   # 0: no position, 1: long, -1: short
    entry_price = 0.0
    position_size = 0.0
    capital = initial_capital
    trade_pnls = []
    
    for t in range(lookback_days, len(indicator)):
        current_date = indicator.index[t]
        
        if method in ["spread", "ratio"]:
            # Compute the z-score for signal generation.
            current_zscore = (indicator.iloc[t] - roll_mean.iloc[t]) / roll_std.iloc[t]
            print("Current Z score: ", current_zscore)
            print("Exit threshold: ", exit_threshold)
            # If a position is open, exit when the absolute zscore falls below exit_threshold.
            if position != 0:
                if abs(current_zscore) < abs(exit_threshold):
                    exit_price = indicator.iloc[t]
                    trade_return = (exit_price - entry_price) * position * position_size
                    capital += trade_return
                    trade_pnls.append(trade_return)
                    signals.append((current_date, 'Exit', current_zscore, position_size, exit_price, trade_return))
                    position = 0
                    entry_price = 0.0
                    position_size = 0.0
                    continue  # Skip entry processing this step.
            
            # Only enter a trade if no position is open.
            if position == 0:
                if current_zscore > entry_threshold:
                    potential_entry_price = indicator.iloc[t]
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = -1  # For a high zscore, short the spread/ratio.
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    signals.append((current_date, 'Enter Short', current_zscore, position_size, entry_price))
                elif current_zscore < -entry_threshold:
                    potential_entry_price = indicator.iloc[t]
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = 1  # For a low zscore, long the spread/ratio.
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    signals.append((current_date, 'Enter Long', current_zscore, position_size, entry_price))
        
        elif method == "bollinger_bands":
            # For the bollinger_bands method, use a Bollinger Bands–like approach.
            upper_band = roll_mean + entry_threshold * roll_std
            lower_band = roll_mean - entry_threshold * roll_std
            current_value = indicator.iloc[t]
            
            # If a position is open, exit when the indicator returns within the bands.
            if position != 0:
                if lower_band.iloc[t] < current_value < upper_band.iloc[t]:
                    exit_price = current_value
                    trade_return = (exit_price - entry_price) * position * position_size
                    capital += trade_return
                    trade_pnls.append(trade_return)
                    signals.append((current_date, 'Exit', None, position_size, exit_price, trade_return))
                    position = 0
                    entry_price = 0.0
                    position_size = 0.0
                    continue
            
            # Only enter a new position if none is open.
            if position == 0:
                # Enter short if current value is above the upper band.
                if current_value > upper_band.iloc[t]:
                    potential_entry_price = current_value
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = -1
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    signals.append((current_date, 'Enter Short', None, position_size, entry_price))
                # Enter long if current value is below the lower band.
                elif current_value < lower_band.iloc[t]:
                    potential_entry_price = current_value
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = 1
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    signals.append((current_date, 'Enter Long', None, position_size, entry_price))
    
    # At the end of the loop, force an exit if a position is still open.
    if position != 0:
        exit_price = indicator.iloc[-1]
        trade_return = (exit_price - entry_price) * position * position_size
        capital += trade_return
        trade_pnls.append(trade_return)
        signals.append((indicator.index[-1], 'Forced Exit', None, position_size, exit_price, trade_return))
        position = 0
        entry_price = 0.0
        position_size = 0.0

    segment_pnl = capital - initial_capital
    return signals, segment_pnl, trade_pnls, capital


import numpy as np
import pandas as pd

def simulate_pair_trading_with_sizing_with_loss_threshold(stock1, stock2, hedge_ratio, sim_data, 
                                      method="spread", 
                                      lookback_days=60,
                                      entry_threshold=1, exit_threshold=0.5, 
                                      position_fraction=0.1,
                                      loss_threshold=0.1,
                                      initial_capital=10000):
    """
    Simulate pairs trading for a given segment using position sizing, with an added loss threshold.
    
    If a trade's percentage loss exceeds loss_threshold, the position is closed immediately and the 
    function returns the number of days that the trade was active.
    
    Parameters:
      stock1, stock2: column names in sim_data
      hedge_ratio: used for spread calculation (or estimated externally)
      sim_data: a DataFrame with datetime index and price columns for each stock
      method: one of "spread", "ratio", or "bollinger_bands"
      lookback_days: rolling window to compute mean and std (or bands)
      entry_threshold: for 'spread' and 'ratio', the zscore threshold for entry; 
                       for 'bollinger_bands', the multiplier for the bands
      exit_threshold: for 'spread' and 'ratio', exit when abs(zscore) < this value;
                      for 'bollinger_bands', exit when indicator returns within bands
      position_fraction: fraction of capital allocated per trade
      loss_threshold: if the trade loses more than this fraction (e.g. 0.1 for 10%), exit immediately
      initial_capital: starting capital
      
    Returns:
      signals: list of tuples with trade actions and parameters
      segment_pnl: profit/loss for the segment
      trade_pnls: list of individual trade PnL values
      capital: updated capital after simulation
      trading_days: if a loss-threshold exit occurred, the number of days the trade was active;
                      otherwise 0.
    """
    # Calculate indicator based on method.
    if method in ["spread", "bollinger_bands"]:
        indicator = sim_data[stock1] - hedge_ratio * sim_data[stock2]
    elif method == "ratio":
        indicator = sim_data[stock1] / sim_data[stock2]
    else:
        raise ValueError("Invalid method. Choose from 'spread', 'ratio', 'bollinger_bands'.")
    
    # Rolling statistics.
    roll_mean = indicator.rolling(window=lookback_days).mean()
    roll_std = indicator.rolling(window=lookback_days).std().replace(0, 1e-8)
    
    signals = []
    position = 0   # 0: no position, 1: long, -1: short
    entry_price = 0.0
    position_size = 0.0
    capital = initial_capital
    trade_pnls = []
    trading_days = -1

    # Loop over each day after the lookback period.
    for t in range(lookback_days, len(indicator)):
        current_date = indicator.index[t]
        current_indicator = indicator.iloc[t]
        
        # If a position is open, check if the loss threshold is exceeded.
        if position != 0 and trade_entry_index is not None:
            if position == 1:
                # For a long position, loss when the current value is below the entry price.
                loss_pct = (entry_price - current_indicator) / entry_price
            else:
                # For a short position, loss when the current value is above the entry price.
                loss_pct = (current_indicator - entry_price) / entry_price
            if loss_pct >= loss_threshold:
                exit_price = current_indicator
                trade_return = (exit_price - entry_price) * position * position_size
                capital += trade_return
                trade_pnls.append(trade_return)
                # For spread/ratio methods, we can compute the z-score for logging.
                if method in ["spread", "ratio"]:
                    current_zscore = (current_indicator - roll_mean.iloc[t]) / roll_std.iloc[t]
                    signals.append((current_date, 'Exit Loss Threshold', current_zscore, position_size, exit_price, trade_return))
                else:
                    signals.append((current_date, 'Exit Loss Threshold', None, position_size, exit_price, trade_return))
                trading_days = t - lookback_days
                segment_pnl = capital - initial_capital
                return signals, segment_pnl, trade_pnls, capital, trading_days

        if method in ["spread", "ratio"]:
            # Compute the z-score.
            current_zscore = (current_indicator - roll_mean.iloc[t]) / roll_std.iloc[t]
            
            # If a position is open, exit if the absolute zscore is below exit_threshold.
            if position != 0:
                if abs(current_zscore) < abs(exit_threshold):
                    exit_price = current_indicator
                    trade_return = (exit_price - entry_price) * position * position_size
                    capital += trade_return
                    trade_pnls.append(trade_return)
                    signals.append((current_date, 'Exit', current_zscore, position_size, exit_price, trade_return))
                    position = 0
                    entry_price = 0.0
                    position_size = 0.0
                    trade_entry_index = None
                    continue
            
            # If no position is open, check for entry signals.
            if position == 0:
                if current_zscore > entry_threshold:
                    potential_entry_price = current_indicator
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = -1   # short the spread/ratio for a high zscore.
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    trade_entry_index = t
                    signals.append((current_date, 'Enter Short', current_zscore, position_size, entry_price))
                elif current_zscore < -entry_threshold:
                    potential_entry_price = current_indicator
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = 1    # long the spread/ratio for a low zscore.
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    trade_entry_index = t
                    signals.append((current_date, 'Enter Long', current_zscore, position_size, entry_price))
                    
        elif method == "bollinger_bands":
            # Compute Bollinger-like bands.
            upper_band = roll_mean + entry_threshold * roll_std
            lower_band = roll_mean - entry_threshold * roll_std
            current_value = current_indicator
            
            # If a position is open, exit if the indicator returns within the bands.
            if position != 0:
                if lower_band.iloc[t] < current_value < upper_band.iloc[t]:
                    exit_price = current_value
                    trade_return = (exit_price - entry_price) * position * position_size
                    capital += trade_return
                    trade_pnls.append(trade_return)
                    signals.append((current_date, 'Exit', None, position_size, exit_price, trade_return))
                    position = 0
                    entry_price = 0.0
                    position_size = 0.0
                    trade_entry_index = None
                    continue
            
            # If no position is open, check for entry signals.
            if position == 0:
                if current_value > upper_band.iloc[t]:
                    potential_entry_price = current_value
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = -1
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    trade_entry_index = t
                    signals.append((current_date, 'Enter Short', None, position_size, entry_price))
                elif current_value < lower_band.iloc[t]:
                    potential_entry_price = current_value
                    if abs(potential_entry_price) < 1e-6:
                        continue
                    position = 1
                    entry_price = potential_entry_price
                    position_size = (capital * position_fraction) / abs(entry_price)
                    trade_entry_index = t
                    signals.append((current_date, 'Enter Long', None, position_size, entry_price))
                    
    # End of loop: if a position is still open, force an exit.
    if position != 0 and trade_entry_index is not None:
        exit_price = indicator.iloc[-1]
        trade_return = (exit_price - entry_price) * position * position_size
        capital += trade_return
        trade_pnls.append(trade_return)
        signals.append((indicator.index[-1], 'Forced Exit', None, position_size, exit_price, trade_return))
        
    segment_pnl = capital - initial_capital
    return signals, segment_pnl, trade_pnls, capital, trading_days
