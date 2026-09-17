# Cross-Sectional Equity Alpha Research Pipeline

A self-contained, end-to-end systematic trading strategy research project, built to mirror the
day-to-day workflow of a **Quantitative Researcher** role like the one described for Trexquant
(Gurugram): form a hypothesis about market behaviour, mine data for signal, build a statistical/ML
model in Python, backtest it rigorously, and structure the code the way a research-to-production
handoff would expect.

## Why this project maps to the JD

| JD requirement | Where it shows up here |
|---|---|
| Form hypotheses about market behaviour | `data/generate_data.py` docstring + `features/feature_engineering.py`: momentum, short-term mean-reversion, and volatility hypotheses |
| Mine large financial / alternative datasets | `features/feature_engineering.py`: cross-sectional z-scores/ranks, technical indicators, volume-based features |
| Build statistical & ML models in Python | `models/model.py`: `GradientBoostingRegressor` trained on engineered features |
| Rigorous backtesting | `models/model.py` purged walk-forward CV + `backtest/backtest_engine.py` dollar-neutral portfolio construction with transaction costs |
| Python / ML / stats / probability / linear algebra | numpy, pandas, scikit-learn, scipy used throughout; z-scores, rank correlation (Spearman IC), covariance-driven factor simulation |
| Understanding of financial markets | Long/short dollar-neutral construction, turnover-based transaction costs, Sharpe/Calmar/drawdown reporting — the standard vocabulary of a systematic equities book |

## Pipeline

```
generate_data.py  →  feature_engineering.py  →  model.py  →  backtest_engine.py  →  metrics.py
  (synthetic          (momentum, reversal,       (purged        (dollar-neutral       (Sharpe,
   multi-asset          vol, RSI, cross-           walk-forward   long/short,           drawdown,
   OHLCV data)          sectional z/rank)          GBM signal,     turnover-costed        Calmar,
                                                    rank-IC eval)   daily P&L)             IC, etc.)
```

Run the whole thing with:

```bash
pip install -r requirements.txt
python main.py
```

This writes `outputs/backtest_results.csv`, `outputs/fold_ic.csv`, `outputs/feature_importances.csv`,
and a 4-panel diagnostic chart to `outputs/research_report.png` (equity curve, drawdown, per-fold
out-of-sample IC, and feature importances).

## Design choices worth calling out in an interview

1. **Synthetic data with a known ground truth.** `generate_data.py` simulates a market factor,
   sector factors, and *explicitly embeds* a short-horizon mean-reversion effect and a
   medium-horizon momentum effect into the idiosyncratic returns. This makes the whole pipeline
   reproducible without a paid data vendor, while still giving the model something real to find —
   the feature-importance plot recovering `mom_21d`, `mom_10d`, and `mom_63d` as the top predictors
   is a sanity check that the pipeline is correctly extracting the embedded signal. In production
   this module would be swapped for a loader against a vendor feed (e.g. an adjusted OHLCV table
   in a data warehouse); nothing downstream would need to change since it only assumes a
   `[date, asset, close, volume]` long-format frame.

2. **No look-ahead bias.** All features at time *t* use only information available at or before
   *t*. Labels (`fwd_ret`) are explicitly forward-shifted and are the *last* thing computed.

3. **Purged walk-forward cross-validation**, not k-fold. Randomly shuffled k-fold CV leaks future
   information into training whenever labels are constructed from a forward-looking window (here,
   a 5-day forward return) — a classic pitfall in financial ML. Instead, each fold trains on an
   expanding history, purges the `horizon` days immediately before the test window (since their
   labels overlap the test period), and evaluates strictly out-of-sample walking forward through
   time — closer to how a live retraining schedule would behave.

4. **Rank-IC as the primary skill metric**, not R². Cross-sectional equity strategies care about
   whether the *ranking* of assets by predicted return is informative, not the magnitude of any
   single prediction, so Spearman rank correlation between signal and realized forward return is
   reported per fold.

5. **Dollar-neutral, rank-weighted portfolio construction.** Long the top quintile / short the
   bottom quintile by signal, weighted by how extreme the rank is, scaled so gross long = gross
   short. This isolates the stock-selection signal from market/sector direction, which is the
   point of a market-neutral systematic strategy.

6. **Transaction costs are explicit**, charged in bps on turnover at each rebalance, and both
   gross and net performance are reported side by side so cost drag is visible rather than hidden.

## Honest caveats

The reported Sharpe ratio (~5) is a property of the **synthetic data**, which contains a strong,
cleanly embedded signal with no regime change, no structural breaks, and no crowding effects —
real equity alpha signals of this style typically run at OOS Sharpe ratios well under 2 after
costs. The point of this project is to demonstrate a correct, leakage-free research pipeline
end-to-end (hypothesis → features → walk-forward ML → costed backtest → diagnostics), not to
claim a tradeable live strategy. Natural next steps toward realism:

- Swap in real market data and add point-in-time fundamentals/alt-data joins
- Add borrow cost / short-availability constraints and a market-impact cost model
- Sector/factor-neutralize the portfolio via optimization (e.g. mean-variance with exposure
  constraints) rather than simple rank-weighting
- Add regime-aware validation (e.g. performance conditioned on volatility regime)
- Hyperparameter search nested *inside* each walk-forward fold's training window only

## File structure

```
trexquant_alpha_research/
├── data/generate_data.py           # synthetic multi-asset market data generator
├── features/feature_engineering.py # technical + cross-sectional feature construction
├── models/model.py                 # purged walk-forward GBM training & rank-IC evaluation
├── backtest/backtest_engine.py     # dollar-neutral portfolio construction + costed backtest
├── utils/metrics.py                # Sharpe, drawdown, Calmar, hit rate, turnover
├── main.py                         # orchestrates the full pipeline and produces the report
├── requirements.txt
└── outputs/                        # generated data, predictions, metrics, and report.png
```
