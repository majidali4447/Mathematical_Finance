# Delta Hedging with Real AAPL Data

This project extends Section 10 of the report by repeating the same short-call delta hedging simulation on one realized historical AAPL price path.

The project does **not** use Monte Carlo simulation and does **not** create additional simulated price paths. It uses only one historical AAPL path for January 2024--December 2024.

## Files

```text
delta_hedging_real_data/
│
├── main.py
├── requirements.txt
├── README.md
├── section_10_6.md
├── data/
└── outputs/
    ├── stock_path.png
    ├── delta_evolution.png
    ├── hedged_vs_unhedged.png
    ├── pnl_comparison.png
    ├── results.csv
    ├── portfolio_values.csv
    ├── pnl_summary.csv
    ├── pnl_comparison.csv
    ├── hedging_efficiency.csv
    └── section_10_6_generated.md
```

## What the code does

1. Downloads daily AAPL prices from Yahoo Finance for January 2024--December 2024.
2. Sets the call strike as `K = S_0`.
3. Estimates annualized volatility from realized daily log returns.
4. Computes Black--Scholes call price, Delta, Gamma, Vega and Theta for every trading day.
5. Applies the same short-call hedging logic to:
   - Unhedged short call
   - Daily delta hedge
   - Weekly delta hedge
   - Monthly delta hedge
6. Calculates realized Mean P&L, Standard Deviation, Variance and Terminal P&L from the single realized portfolio P&L time series.
7. Calculates hedging efficiency using:

```text
Hedging Efficiency = 1 - Var(hedged P&L) / Var(unhedged P&L)
```

## Important formula note

Vega and Theta match the original Black--Scholes formulas used in the report:

- Vega is **not** divided by 100.
- Theta is **not** divided by 252.

Therefore, Theta is reported in annualized Black--Scholes scale. If daily Theta is required separately, it should be clearly labelled as daily Theta.

## Run in VS Code or terminal

```bash
pip install -r requirements.txt
python main.py
```

After running, all figures and CSV files are saved automatically in the `outputs/` folder. The file `section_10_6.md` is also updated with the actual values generated from the CSV results.

## Run in Google Colab

```python
!pip install yfinance scipy matplotlib pandas numpy
!python main.py
```

If the project is uploaded to Colab, make sure the working directory contains `main.py` before running the command.

## Data source

Historical AAPL prices are downloaded using Yahoo Finance through the `yfinance` Python package.
