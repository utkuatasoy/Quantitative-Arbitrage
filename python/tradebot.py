# tradebot.py
import numpy as np
import pandas as pd
from itertools import product
from data_reader import load_data
from pair_selectors import get_candidate_pairs, choose_best_pair
from traders import simulate_pair_trading_with_sizing, simulate_pair_trading_with_sizing_with_loss_threshold

def calculate_total_return(portfolio_values):
    """
    Calculate the overall total return percentage from the portfolio equity curve.
    """
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    return total_return

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    """
    Calculate the annualized Sharpe Ratio given an array of daily returns.
    """
    period_rf_rate = risk_free_rate / periods
    excess_returns = returns - period_rf_rate
    std_excess = np.std(excess_returns)
    if std_excess == 0:
        return np.nan
    sharpe_ratio = np.mean(excess_returns) / std_excess
    annualized_sharpe = sharpe_ratio * np.sqrt(periods)
    return annualized_sharpe

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the Maximum Drawdown (MDD) as a percentage from the portfolio equity curve.
    """
    portfolio_values = np.array(portfolio_values)
    cum_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cum_max) / cum_max
    max_drawdown = drawdowns.min()  # Negative value indicates loss
    return max_drawdown * 100

def calculate_win_rate(trade_pnls):
    """
    Calculate the win rate (% of profitable trades) from a list of trade PnLs.
    """
    trade_pnls = np.array(trade_pnls)
    if trade_pnls.size == 0:
        return np.nan
    wins = np.sum(trade_pnls > 0)
    win_rate = (wins / trade_pnls.size) * 100
    return win_rate

def calculate_average_trade_duration_from_signals(signals):
    """
    Calculate average trade duration (in days) from trade signals.
    Expects each signal to be a dict with 'entry_date' and 'exit_date' in ISO format.
    """
    durations = []
    for s in signals:
        if 'entry_date' in s and 'exit_date' in s:
            try:
                entry_date = pd.to_datetime(s['entry_date'])
                exit_date = pd.to_datetime(s['exit_date'])
                durations.append((exit_date - entry_date).days)
            except Exception as e:
                print("Error parsing dates in signal:", s, e)
    if durations:
        return np.mean(durations)
    return np.nan

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
    Returns overall signals, final capital, percentage return, trade PnLs,
    and a set of performance metrics.
    """
    overall_signals = []
    overall_trade_pnls = []
    portfolio_history = []  # List of tuples: (day_index, capital)
    
    num_days = len(data)
    current_index = initial_lookback
    capital = starting_capital
    portfolio_history.append((current_index, capital))
    
    while current_index < num_days:
        start_idx = max(0, current_index - cointegration_window)
        window_data = data.iloc[start_idx:current_index]
        candidates = get_candidate_pairs(window_data, significance=cointegration_significance)
        current_pair = choose_best_pair(
            candidates,
            window_data,
            cointegration_window,
            method='ratio',
            combine_std=True,
        )

        if current_pair is None:
            print("No candidate pair found to start simulation.")
            return None
        else:
            current_stock1, current_stock2, current_pvalue, current_hedge_ratio = current_pair
            print(f"Pair selected: {current_stock1} & {current_stock2}")

        segment_end_index = min(current_index + rebalance_interval, num_days)
        segment_data = data.iloc[current_index - sim_lookback_days: segment_end_index]
        
        signals, seg_pnl, trade_pnls, updated_capital, trading_days = simulate_pair_trading_with_sizing_with_loss_threshold(
            current_stock1, current_stock2, current_hedge_ratio,
            segment_data, lookback_days=sim_lookback_days,
            entry_threshold=entry_threshold, exit_threshold=exit_threshold,
            position_fraction=position_fraction,
            initial_capital=capital, method='bollinger_bands', loss_threshold=loss_threshold,
        )
        
        # --- Fix for signals missing date info ---
        # Use trading_days (or rebalance_interval) as the fallback duration.
        fixed_signals = []
        fallback_duration = trading_days if trading_days != -1 else rebalance_interval
        if isinstance(segment_data.index, pd.DatetimeIndex):
            segment_end_date_actual = segment_data.index[-1]
            fallback_entry_date = segment_end_date_actual - pd.Timedelta(days=fallback_duration)
        elif 'date' in segment_data.columns:
            segment_end_date_actual = pd.to_datetime(segment_data['date'].iloc[-1])
            fallback_entry_date = segment_end_date_actual - pd.Timedelta(days=fallback_duration)
        else:
            fallback_entry_date = str(current_index)
            segment_end_date_actual = str(segment_end_index)
        
        for s in signals:
            if isinstance(s, dict):
                if 'entry_date' not in s or 'exit_date' not in s:
                    if 'entry_date' not in s:
                        s['entry_date'] = str(fallback_entry_date)
                    if 'exit_date' not in s:
                        s['exit_date'] = str(segment_end_date_actual)
                fixed_signals.append(s)
            elif isinstance(s, tuple):
                new_signal = {'entry_date': str(fallback_entry_date), 'exit_date': str(segment_end_date_actual)}
                for i, item in enumerate(s[2:], start=1):
                    new_signal[f'info_{i}'] = item
                fixed_signals.append(new_signal)
            else:
                pass
        signals = fixed_signals
        # ---------------------------------------------------------
        
        overall_signals.extend(signals)
        overall_trade_pnls.extend(trade_pnls)
        interval = rebalance_interval if trading_days == -1 else trading_days
        segment_end_index = min(current_index + interval, num_days)
        print(f"Segment {current_index} to {segment_end_index}: pnl = {seg_pnl:.2f}, Capital updated to: {updated_capital:.2f}")
        
        capital = updated_capital
        current_index = segment_end_index
        portfolio_history.append((current_index, capital))
        
    percent_return = ((capital / starting_capital) - 1) * 100

    # Calculate performance metrics from the portfolio history and trade outcomes.
    portfolio_values = [cap for (day, cap) in portfolio_history]
    total_return = calculate_total_return(portfolio_values)
    
    # Derive approximate daily returns from the portfolio history.
    daily_returns = []
    for i in range(1, len(portfolio_history)):
        day_diff = portfolio_history[i][0] - portfolio_history[i-1][0]
        if day_diff > 0:
            r = (portfolio_history[i][1] / portfolio_history[i-1][1])**(1/day_diff) - 1
            daily_returns.append(r)
    daily_returns = np.array(daily_returns)
    annual_sharpe = calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else np.nan
    max_drawdown = calculate_max_drawdown(portfolio_values)
    win_rate = calculate_win_rate(np.array(overall_trade_pnls))
    avg_trade_duration = calculate_average_trade_duration_from_signals(overall_signals)
    
    results = {
        'signals': overall_signals,
        'final_capital': capital,
        'percent_return': percent_return,
        'trade_pnls': overall_trade_pnls,
        'performance_metrics': {
            'Total Return (%)': total_return,
            'Annualized Sharpe Ratio': annual_sharpe,
            'Maximum Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Average Trade Duration (days)': avg_trade_duration
        }
    }
    return results

if __name__ == "__main__":
    # Load data and define in-sample period.
    file_path = "/Users/hakandogan/LocalDocuments/yap 471/proje/PairsTrading-main/data/data.csv"
    data = load_data(file_path)
    in_sample_data = data[-504:]
    
    # Define parameter grid for grid search.
    param_grid = {
        'loss_threshold': [0.6, 0.8],
        'entry_threshold': [1, 1.2],
        'exit_threshold': [0.75, 0.85],
        'cointegration_window': [120, 150],
        'position_fraction': [0.3, 0.5],
        'rebalance_interval': [15, 20],
        'sim_lookback_days': [30, 45]
    }
    
    results_list = []
    for loss_threshold, entry_threshold, exit_threshold, cointegration_window, position_fraction, rebalance_interval, sim_lookback_days in product(
            param_grid['loss_threshold'],
            param_grid['entry_threshold'],
            param_grid['exit_threshold'],
            param_grid['cointegration_window'],
            param_grid['position_fraction'],
            param_grid['rebalance_interval'],
            param_grid['sim_lookback_days']):
        print(f"\nTesting parameters: loss_threshold={loss_threshold}, entry_threshold={entry_threshold}, "
              f"exit_threshold={exit_threshold}, cointegration_window={cointegration_window}, "
              f"position_fraction={position_fraction}, rebalance_interval={rebalance_interval}, sim_lookback_days={sim_lookback_days}")
        dynamic_results = simulate_dynamic_pair_trading_with_sizing(
            data=in_sample_data, 
            initial_lookback=252, 
            rebalance_interval=rebalance_interval, 
            loss_threshold=loss_threshold,
            sim_lookback_days=sim_lookback_days,
            entry_threshold=entry_threshold, 
            exit_threshold=exit_threshold, 
            cointegration_significance=0.05, 
            cointegration_window=cointegration_window,
            position_fraction=position_fraction,
            starting_capital=10000
        )
        if dynamic_results is not None:
            final_capital = dynamic_results['final_capital']
            percent_return = dynamic_results['percent_return']
            annual_sharpe = dynamic_results['performance_metrics']['Annualized Sharpe Ratio']
            results_list.append({
                'loss_threshold': loss_threshold,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'cointegration_window': cointegration_window,
                'position_fraction': position_fraction,
                'rebalance_interval': rebalance_interval,
                'sim_lookback_days': sim_lookback_days,
                'final_capital': final_capital,
                'percent_return': percent_return,
                'annual_sharpe': annual_sharpe
            })
    
    # Compile grid search results in a DataFrame and display.
    results_df = pd.DataFrame(results_list)
    print("\nGrid Search Results:")
    print(results_df)
    results_df.to_csv("grid_search_results.csv", index=False)

