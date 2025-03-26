# Quantitative Arbitrage: Multi-Asset Cointegration & Statistical Trading

This repository implements a **pairs trading bot** that exploits mean-reverting relationships between cointegrated assets. While examples emphasize BIST100, the approach can be applied to any liquid market.

---

## Project Overview

1. **Core Idea**  
   - We **detect cointegration** between assets (Engle-Granger test).  
   - **Go long** one asset and **short** the other when their spread diverges.  
   - **Profit** when the spread mean-reverts.

2. **Workflow**  
   - **Data**: Download & clean historical prices (e.g., Yahoo Finance).  
   - **Cointegration**: Identify candidate pairs.  
   - **Signals**: Trigger trades using Z-Scores or Bollinger Bands.  
   - **Backtester**: Simulate trades, track P&L, and compute performance metrics.  
   - **GUI**: A PyQt interface (`gui.py`) to configure parameters, visualize charts, and run simulations in real time.

---

## File Structure

- **`data_preprocessing.py`**  
  - Fetches and cleans daily data (removes bad records, forward-fills missing values).
- **`cointegration_tests.py`**  
  - Performs Engle-Granger tests; returns pairs that are statistically cointegrated.
- **`pair_selection.py`**  
  - Ranks pairs by volatility, half-life, or your custom logic.
- **`trading_signals.py`**  
  - Generates trading signals (Spread/Ratio Z-Score, Bollinger-based rules).
- **`backtester.py`**  
  - Runs a historical simulation: enters/exits positions, logs trades, calculates returns.
- **`gui.py`**  
  - **PyQt** interface. You can tweak parameters and see results without command line.
- **`main.py`**  
  - The main orchestrator:
    - `--mode backtest`: Command-line backtest  
    - `--mode gui`: Launches the PyQt interface  
---

## References

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs trading: Performance of a relative-value arbitrage rule.* *Review of Financial Studies*, 19(3), 797–827.  
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.* John Wiley & Sons.  
3. Caldeira, J. F., & Moura, O. A. (2013). *Selection of pairs of stocks based on cointegration: A statistical arbitrage strategy.* Brazilian Review of Finance, 11(3), 369–402.  
4. Krauss, C., Do, X., & Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500.* European Journal of Operational Research, 259(2), 689–702.

---

**Disclaimer**: This code is for educational purposes. In a live trading environment, be sure to account for transaction costs, liquidity constraints, and other market frictions. Use at your own risk!
