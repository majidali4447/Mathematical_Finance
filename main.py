Real Market Data Delta Hedging Application

This script repeats Section 10 using one realized historical AAPL price path.
No Monte Carlo simulation and no additional simulated price paths are used.

The program downloads AAPL daily prices for 2024 from Yahoo Finance, computes
Black--Scholes call prices and Greeks for every trading day, and compares an
unhedged short call with daily, weekly and monthly delta hedging.

All outputs are saved automatically in the outputs/ folder.
"""

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

# ------------------------------------------------------------
# 1. Project folders
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. Model and data parameters
# ------------------------------------------------------------
TICKER = "AAPL"
START_DATE = "2024-01-01"
END_DATE = "2025-01-01"
RISK_FREE_RATE = 0.02
TRADING_DAYS = 252

# ------------------------------------------------------------
# 3. Black--Scholes functions
# ------------------------------------------------------------
def black_scholes_call(S, K, T, r, sigma):
    """Return the Black--Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_greeks(S, K, T, r, sigma):
    """Return Delta, Gamma, Vega and Theta for a European call option."""
    if T <= 0:
        delta = 1.0 if S > K else 0.0
        return delta, 0.0, 0.0, 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)

    return delta, gamma, vega, theta

# ------------------------------------------------------------
# 4. Download real AAPL data
# ------------------------------------------------------------
print("Downloading AAPL historical prices from Yahoo Finance...")

raw_data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)

if raw_data.empty:
    raise RuntimeError("No Yahoo Finance data were downloaded.")

prices_df = raw_data[["Close"]].dropna().copy()
prices_df.rename(columns={"Close": "Stock Price"}, inplace=True)
prices_df.to_csv(DATA_DIR / "aapl_2024_prices.csv")

prices = prices_df["Stock Price"].to_numpy(dtype=float)
dates = prices_df.index
N = len(prices)

S0 = float(prices[0])
K = S0

log_returns = np.log(prices[1:] / prices[:-1])
sigma = float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS))
r = RISK_FREE_RATE

# ------------------------------------------------------------
# 5. Compute daily Black--Scholes prices and Greeks
# ------------------------------------------------------------
greek_records = []

for i, (date, S) in enumerate(zip(dates, prices)):
    tau = max((N - 1 - i) / TRADING_DAYS, 0.0)

    call_price = black_scholes_call(S, K, tau, r, sigma)
    delta, gamma, vega, theta = black_scholes_greeks(S, K, tau, r, sigma)

    greek_records.append({
        "Date": date,
        "Stock Price": S,
        "Time to Maturity": tau,
        "Call Price": call_price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
    })

greeks_df = pd.DataFrame(greek_records)
greeks_df.to_csv(OUTPUT_DIR / "results.csv", index=False)

call_prices = greeks_df["Call Price"].to_numpy(dtype=float)
deltas = greeks_df["Delta"].to_numpy(dtype=float)
gammas = greeks_df["Gamma"].to_numpy(dtype=float)
vegas = greeks_df["Vega"].to_numpy(dtype=float)
thetas = greeks_df["Theta"].to_numpy(dtype=float)

# ------------------------------------------------------------
# 6. Short-call portfolio and delta hedging strategies
# ------------------------------------------------------------
def run_strategy(rebalance_interval=None):
    """
    Compute the realized portfolio P&L time series.

    rebalance_interval = None -> unhedged short call
    rebalance_interval = 1 -> daily delta hedge
    rebalance_interval = 5 -> weekly delta hedge
    rebalance_interval = 21 -> monthly delta hedge
    """
    cash = call_prices[0]
    shares = 0.0

    if rebalance_interval is not None:
        shares = deltas[0]
        cash -= shares * prices[0]

    pnl = np.zeros(N)
    stock_position = np.zeros(N)
    cash_position = np.zeros(N)

    pnl[0] = cash + shares * prices[0] - call_prices[0]
    stock_position[0] = shares
    cash_position[0] = cash

    for t in range(1, N):
        cash *= math.exp(r / TRADING_DAYS)

        if rebalance_interval is not None and t % rebalance_interval == 0:
            new_shares = deltas[t]
            cash -= (new_shares - shares) * prices[t]
            shares = new_shares

        pnl[t] = cash + shares * prices[t] - call_prices[t]
        stock_position[t] = shares
        cash_position[t] = cash

    return pnl, stock_position, cash_position


unhedged_pnl, _, _ = run_strategy(None)
daily_pnl, _, _ = run_strategy(1)
weekly_pnl, _, _ = run_strategy(5)
monthly_pnl, _, _ = run_strategy(21)

portfolio_df = pd.DataFrame({
    "Date": dates,
    "Unhedged Short Call": unhedged_pnl,
    "Daily Delta Hedge": daily_pnl,
    "Weekly Delta Hedge": weekly_pnl,
    "Monthly Delta Hedge": monthly_pnl,
})
portfolio_df.to_csv(OUTPUT_DIR / "portfolio_values.csv", index=False)

# ------------------------------------------------------------
# 7. Realized P&L statistics
# ------------------------------------------------------------
def realized_statistics(name, pnl_series):
    return {
        "Strategy": name,
        "Mean P&L": float(np.mean(pnl_series)),
        "Standard Deviation": float(np.std(pnl_series, ddof=1)),
        "Variance": float(np.var(pnl_series, ddof=1)),
        "Terminal P&L": float(pnl_series[-1]),
    }


summary_df = pd.DataFrame([
    realized_statistics("Unhedged Short Call", unhedged_pnl),
    realized_statistics("Daily Delta Hedge", daily_pnl),
    realized_statistics("Weekly Delta Hedge", weekly_pnl),
    realized_statistics("Monthly Delta Hedge", monthly_pnl),
])

summary_df.to_csv(OUTPUT_DIR / "pnl_summary.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "pnl_comparison.csv", index=False)

unhedged_variance = float(
    summary_df.loc[
        summary_df["Strategy"] == "Unhedged Short Call",
        "Variance"
    ].iloc[0]
)

efficiency_records = []

for strategy in ["Daily Delta Hedge", "Weekly Delta Hedge", "Monthly Delta Hedge"]:
    hedged_variance = float(
        summary_df.loc[
            summary_df["Strategy"] == strategy,
            "Variance"
        ].iloc[0]
    )

    efficiency = 1.0 - hedged_variance / unhedged_variance

    efficiency_records.append({
        "Strategy": strategy,
        "Hedging Efficiency": efficiency,
    })

efficiency_df = pd.DataFrame(efficiency_records)
efficiency_df.to_csv(OUTPUT_DIR / "hedging_efficiency.csv", index=False)

# ------------------------------------------------------------
# 8. Figures
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(dates, prices)
plt.title("Real AAPL Stock Price Path, 2024")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "stock_path.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(dates, deltas)
plt.title("Delta Evolution for AAPL Call Option")
plt.xlabel("Date")
plt.ylabel("Delta")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "delta_evolution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(dates, gammas)
plt.title("Gamma Evolution for AAPL Call Option")
plt.xlabel("Date")
plt.ylabel("Gamma")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gamma_evolution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(dates, vegas)
plt.title("Vega Evolution for AAPL Call Option")
plt.xlabel("Date")
plt.ylabel("Vega")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vega_evolution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(dates, thetas)
plt.title("Theta Evolution for AAPL Call Option")
plt.xlabel("Date")
plt.ylabel("Theta")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "theta_evolution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(dates, unhedged_pnl, label="Unhedged Short Call")
plt.plot(dates, daily_pnl, label="Daily Delta Hedge")
plt.plot(dates, weekly_pnl, label="Weekly Delta Hedge")
plt.plot(dates, monthly_pnl, label="Monthly Delta Hedge")
plt.title("Unhedged and Delta-Hedged Portfolio Values")
plt.xlabel("Date")
plt.ylabel("Portfolio P&L")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hedged_vs_unhedged.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(summary_df["Strategy"], summary_df["Terminal P&L"])
plt.title("Terminal P&L Comparison")
plt.ylabel("Terminal P&L")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pnl_comparison.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 9. Generate Section 10.6 text
# ------------------------------------------------------------
def money(x):
    return f"{x:,.4f}"


summary_lookup = summary_df.set_index("Strategy")
eff_lookup = efficiency_df.set_index("Strategy")

section_text = f"""## 10.6 Real Market Data Application

We now repeat the simulation using one realized historical stock price path instead of the illustrative path used above. The stock selected is Apple Inc. (AAPL), and daily adjusted closing prices are downloaded from Yahoo Finance for the period January 2024 to December 2024. The option is again treated as a European call written on the underlying stock. The strike price is set equal to the initial stock price, so that K = S_0. The annualized volatility is estimated from the realized daily log returns of AAPL over the sample period, giving sigma = {sigma:.4f}, while the risk-free rate is kept at r = {r:.4f}.

For each trading day, the Black--Scholes call price, Delta, Gamma, Vega and Theta are computed using the same formulas as before. Vega and Theta are reported in the same scale as the original report formulas. Therefore, Vega is not divided by 100 and Theta is not divided by 252. Figure 10.6 shows the realized AAPL stock price path, while Figure 10.7 shows the corresponding Delta evolution of the call option. Figure 10.8, Figure 10.9 and Figure 10.10 report the real Gamma, Vega and Theta evolution.

*Figure 10.6: Real AAPL stock price path, January 2024--December 2024.*

*Figure 10.7: Delta evolution for the AAPL call option based on the realized stock path.*

*Figure 10.8: Gamma evolution for the AAPL call option.*

*Figure 10.9: Vega evolution for the AAPL call option.*

*Figure 10.10: Theta evolution for the AAPL call option.*

The same short-call hedging logic is then applied to the real data. Four cases are compared: an unhedged short call, a daily delta hedge, a weekly delta hedge and a monthly delta hedge. In each hedged portfolio, the stock position is adjusted to the option Delta at the chosen rebalancing frequency. The portfolio values are calculated along the same realized path, so no additional simulated paths are introduced. Figure 10.11 compares the realized P&L time series of the four strategies, and Figure 10.12 reports the terminal P&L at maturity.

*Figure 10.11: Unhedged and delta-hedged portfolio values for the realized AAPL path.*

*Figure 10.12: Terminal P&L comparison for the unhedged, daily, weekly and monthly strategies.*

Table 10.1 reports the realized P&L statistics obtained from the portfolio value time series. The values are produced directly by the Python code and saved in outputs/pnl_summary.csv.

| **Strategy** | **Mean P&L** | **Standard Deviation** | **Variance** | **Terminal P&L** |
|---|---:|---:|---:|---:|
| Unhedged Short Call | {money(summary_lookup.loc['Unhedged Short Call', 'Mean P&L'])} | {money(summary_lookup.loc['Unhedged Short Call', 'Standard Deviation'])} | {money(summary_lookup.loc['Unhedged Short Call', 'Variance'])} | {money(summary_lookup.loc['Unhedged Short Call', 'Terminal P&L'])} |
| Daily Delta Hedge | {money(summary_lookup.loc['Daily Delta Hedge', 'Mean P&L'])} | {money(summary_lookup.loc['Daily Delta Hedge', 'Standard Deviation'])} | {money(summary_lookup.loc['Daily Delta Hedge', 'Variance'])} | {money(summary_lookup.loc['Daily Delta Hedge', 'Terminal P&L'])} |
| Weekly Delta Hedge | {money(summary_lookup.loc['Weekly Delta Hedge', 'Mean P&L'])} | {money(summary_lookup.loc['Weekly Delta Hedge', 'Standard Deviation'])} | {money(summary_lookup.loc['Weekly Delta Hedge', 'Variance'])} | {money(summary_lookup.loc['Weekly Delta Hedge', 'Terminal P&L'])} |
| Monthly Delta Hedge | {money(summary_lookup.loc['Monthly Delta Hedge', 'Mean P&L'])} | {money(summary_lookup.loc['Monthly Delta Hedge', 'Standard Deviation'])} | {money(summary_lookup.loc['Monthly Delta Hedge', 'Variance'])} | {money(summary_lookup.loc['Monthly Delta Hedge', 'Terminal P&L'])} |

*Table 10.1: Realized P&L statistics for the unhedged and delta-hedged AAPL short-call portfolios.*

The hedging efficiency is calculated using the same formula as before,

Hedging Efficiency = 1 - Var(hedged P&L) / Var(unhedged P&L).

Table 10.2 reports the resulting efficiencies.

| **Hedging Frequency** | **Hedging Efficiency** |
|---|---:|
| Daily Delta Hedge | {eff_lookup.loc['Daily Delta Hedge', 'Hedging Efficiency']:.2%} |
| Weekly Delta Hedge | {eff_lookup.loc['Weekly Delta Hedge', 'Hedging Efficiency']:.2%} |
| Monthly Delta Hedge | {eff_lookup.loc['Monthly Delta Hedge', 'Hedging Efficiency']:.2%} |

*Table 10.2: Hedging efficiency for the realized AAPL short-call portfolios.*

The real market data application confirms the result observed in the illustrative simulation. The unhedged short-call portfolio remains exposed to the movement of the underlying stock, while delta hedging reduces the variability of the portfolio P&L. Daily rebalancing gives the largest reduction in variance, followed by weekly and monthly rebalancing. The Gamma, Vega and Theta paths show that important sensitivities remain even after Delta is hedged. This is consistent with the earlier discussion that delta hedging removes first-order spot exposure but does not remove curvature risk, volatility risk or time decay.
"""

(OUTPUT_DIR / "section_10_6_generated.md").write_text(section_text, encoding="utf-8")
(PROJECT_DIR / "section_10_6.md").write_text(section_text, encoding="utf-8")

print("Completed successfully.")
print(f"AAPL prices saved to: {DATA_DIR / 'aapl_2024_prices.csv'}")
print(f"Figures and CSV files saved to: {OUTPUT_DIR}")
print(f"Updated section saved to: {PROJECT_DIR / 'section_10_6.md'}")

print("\nP&L Summary:")
print(summary_df.round(6))

print("\nHedging Efficiency:")
print(efficiency_df.round(6))
