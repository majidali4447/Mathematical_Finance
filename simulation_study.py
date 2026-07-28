"""
Reproducible Single-Path Delta-Hedging Simulation

This program:
1. Generates one stock-price path using geometric Brownian motion.
2. Uses a fixed random seed for exact reproducibility.
3. Calculates Black--Scholes call prices and Greeks on every trading day.
4. Compares an unhedged short call with daily, weekly, and monthly Delta hedges.
5. Saves the data, figures, and LaTeX results table automatically.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


# ------------------------------------------------------------
# 1. Project folders
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "simulation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Reproducible simulation parameters
# ------------------------------------------------------------
S0 = 100.0
K = 100.0
MU = 0.05
RISK_FREE_RATE = 0.05
SIGMA_IMPLIED = 0.20
SIGMA_REALIZED = 0.24
MATURITY = 1.0

TRADING_DAYS = 252
NUMBER_OF_STEPS = 252
RANDOM_SEED = 42

DT = MATURITY / NUMBER_OF_STEPS


# ------------------------------------------------------------
# 3. Black--Scholes functions
# ------------------------------------------------------------
def black_scholes_call(S, K, tau, r, sigma):
    """Return the Black--Scholes European call price."""
    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike must be positive.")

    if tau <= 0:
        return max(S - K, 0.0)

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")

    d1 = (
        math.log(S / K)
        + (r + 0.5 * sigma**2) * tau
    ) / (sigma * math.sqrt(tau))

    d2 = d1 - sigma * math.sqrt(tau)

    return (
        S * norm.cdf(d1)
        - K * math.exp(-r * tau) * norm.cdf(d2)
    )


def black_scholes_greeks(S, K, tau, r, sigma):
    """Return Delta, Gamma, Vega and Theta for a European call."""
    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike must be positive.")

    if tau <= 0:
        if S > K:
            delta = 1.0
        elif S < K:
            delta = 0.0
        else:
            delta = 0.5

        return delta, 0.0, 0.0, 0.0

    d1 = (
        math.log(S / K)
        + (r + 0.5 * sigma**2) * tau
    ) / (sigma * math.sqrt(tau))

    d2 = d1 - sigma * math.sqrt(tau)
    density = norm.pdf(d1)

    delta = norm.cdf(d1)
    gamma = density / (S * sigma * math.sqrt(tau))
    vega = S * density * math.sqrt(tau)

    theta = (
        -(S * density * sigma) / (2 * math.sqrt(tau))
        - r * K * math.exp(-r * tau) * norm.cdf(d2)
    )

    return delta, gamma, vega, theta


# ------------------------------------------------------------
# 4. Generate one reproducible GBM stock-price path
# ------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
normal_shocks = rng.standard_normal(NUMBER_OF_STEPS)

prices = np.empty(NUMBER_OF_STEPS + 1)
prices[0] = S0

for t in range(NUMBER_OF_STEPS):
    prices[t + 1] = prices[t] * math.exp(
        (MU - 0.5 * SIGMA_REALIZED**2) * DT
        + SIGMA_REALIZED * math.sqrt(DT) * normal_shocks[t]
    )

trading_days = np.arange(NUMBER_OF_STEPS + 1)

stock_path_df = pd.DataFrame({
    "Trading Day": trading_days,
    "Normal Shock": np.insert(normal_shocks, 0, np.nan),
    "Stock Price": prices,
})

stock_path_df.to_csv(
    OUTPUT_DIR / "simulated_stock_path.csv",
    index=False
)


# ------------------------------------------------------------
# 5. Calculate call prices and Greeks
# ------------------------------------------------------------
records = []

for i, S in enumerate(prices):
    tau = max(
        (NUMBER_OF_STEPS - i) / TRADING_DAYS,
        0.0
    )

    call_price = black_scholes_call(
        S=S,
        K=K,
        tau=tau,
        r=RISK_FREE_RATE,
        sigma=SIGMA_IMPLIED,
    )

    delta, gamma, vega, theta = black_scholes_greeks(
        S=S,
        K=K,
        tau=tau,
        r=RISK_FREE_RATE,
        sigma=SIGMA_IMPLIED,
    )

    records.append({
        "Trading Day": i,
        "Stock Price": S,
        "Time to Maturity": tau,
        "Call Price": call_price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
    })

greeks_df = pd.DataFrame(records)

greeks_df.to_csv(
    OUTPUT_DIR / "simulation_prices_and_greeks.csv",
    index=False
)

call_prices = greeks_df["Call Price"].to_numpy(dtype=float)
deltas = greeks_df["Delta"].to_numpy(dtype=float)


# ------------------------------------------------------------
# 6. Short-call portfolio strategies
# ------------------------------------------------------------
def run_strategy(rebalance_interval=None):
    """
    Calculate one realized portfolio P&L time series.

    rebalance_interval = None: unhedged short call
    rebalance_interval = 1: daily Delta hedge
    rebalance_interval = 5: weekly Delta hedge
    rebalance_interval = 21: monthly Delta hedge
    """
    number_of_observations = len(prices)

    # Sell the call and receive its initial premium.
    cash = float(call_prices[0])
    shares = 0.0

    number_of_hedge_trades = 0
    stock_turnover = 0.0

    # Establish the initial Delta hedge.
    if rebalance_interval is not None:
        shares = float(deltas[0])
        initial_trade_value = shares * prices[0]

        cash -= initial_trade_value
        stock_turnover += abs(initial_trade_value)
        number_of_hedge_trades += 1

    pnl = np.zeros(number_of_observations)
    stock_position = np.zeros(number_of_observations)
    cash_position = np.zeros(number_of_observations)

    pnl[0] = cash + shares * prices[0] - call_prices[0]
    stock_position[0] = shares
    cash_position[0] = cash

    for t in range(1, number_of_observations):
        # Cash account earns the continuously compounded risk-free rate.
        cash *= math.exp(RISK_FREE_RATE * DT)

        # Do not open a new hedge at maturity.
        if (
            rebalance_interval is not None
            and t < number_of_observations - 1
            and t % rebalance_interval == 0
        ):
            new_shares = float(deltas[t])
            change_in_shares = new_shares - shares
            trade_value = change_in_shares * prices[t]

            cash -= trade_value
            stock_turnover += abs(trade_value)
            shares = new_shares
            number_of_hedge_trades += 1

        pnl[t] = cash + shares * prices[t] - call_prices[t]
        stock_position[t] = shares
        cash_position[t] = cash

    return {
        "pnl": pnl,
        "shares": stock_position,
        "cash": cash_position,
        "number_of_hedge_trades": number_of_hedge_trades,
        "stock_turnover": stock_turnover,
    }


strategies = {
    "Unhedged Short Call": run_strategy(None),
    "Daily Delta Hedge": run_strategy(1),
    "Weekly Delta Hedge": run_strategy(5),
    "Monthly Delta Hedge": run_strategy(21),
}


# ------------------------------------------------------------
# 7. Save complete portfolio paths
# ------------------------------------------------------------
portfolio_df = pd.DataFrame({
    "Trading Day": trading_days,
    "Stock Price": prices,
    "Call Price": call_prices,
})

for strategy_name, result in strategies.items():
    portfolio_df[f"{strategy_name} P&L"] = result["pnl"]
    portfolio_df[f"{strategy_name} Shares"] = result["shares"]
    portfolio_df[f"{strategy_name} Cash"] = result["cash"]

portfolio_df.to_csv(
    OUTPUT_DIR / "simulation_portfolio_values.csv",
    index=False
)


# ------------------------------------------------------------
# 8. Descriptive pathwise statistics
# ------------------------------------------------------------
summary_records = []

for strategy_name, result in strategies.items():
    pnl_series = result["pnl"]

    summary_records.append({
        "Strategy": strategy_name,
        "Mean P&L Across Trading Days": float(np.mean(pnl_series)),
        "P&L Standard Deviation": float(np.std(pnl_series, ddof=1)),
        "P&L Variance": float(np.var(pnl_series, ddof=1)),
        "Terminal P&L": float(pnl_series[-1]),
        "Number of Hedge Trades": result["number_of_hedge_trades"],
        "Total Stock Turnover": result["stock_turnover"],
    })

summary_df = pd.DataFrame(summary_records)

unhedged_variance = float(
    summary_df.loc[
        summary_df["Strategy"] == "Unhedged Short Call",
        "P&L Variance"
    ].iloc[0]
)

summary_df["Pathwise Hedging Efficiency"] = np.nan

for index, row in summary_df.iterrows():
    if row["Strategy"] != "Unhedged Short Call":
        hedged_variance = float(row["P&L Variance"])

        summary_df.loc[
            index,
            "Pathwise Hedging Efficiency"
        ] = 1.0 - hedged_variance / unhedged_variance

summary_df.to_csv(
    OUTPUT_DIR / "simulation_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 9. Generate figures
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(trading_days, prices)
plt.title("Reproducible Simulated Stock Price Path")
plt.xlabel("Trading day")
plt.ylabel("Stock price")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "simulation_stock_path.png",
    dpi=300
)
plt.close()


plt.figure(figsize=(8, 5))
plt.plot(trading_days, deltas)
plt.title("Delta Evolution Along the Simulated Path")
plt.xlabel("Trading day")
plt.ylabel("Call Delta")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "simulation_delta_evolution.png",
    dpi=300
)
plt.close()


plt.figure(figsize=(8, 5))

for strategy_name, result in strategies.items():
    plt.plot(
        trading_days,
        result["pnl"],
        label=strategy_name
    )

plt.title("Unhedged and Delta-Hedged Portfolio P&L")
plt.xlabel("Trading day")
plt.ylabel("Portfolio P&L")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "simulation_pnl_comparison.png",
    dpi=300
)
plt.close()


# ------------------------------------------------------------
# 10. Generate a LaTeX results table
# ------------------------------------------------------------
def latex_escape(text):
    return str(text).replace("&", r"\&")


table_rows = []

for _, row in summary_df.iterrows():
    efficiency = row["Pathwise Hedging Efficiency"]

    if pd.isna(efficiency):
        efficiency_text = "--"
    else:
        efficiency_text = f"{100.0 * efficiency:.2f}\\%"

    table_rows.append(
        f"{latex_escape(row['Strategy'])} & "
        f"{row['Mean P&L Across Trading Days']:.4f} & "
        f"{row['P&L Standard Deviation']:.4f} & "
        f"{row['Terminal P&L']:.4f} & "
        f"{efficiency_text} \\\\"
    )

results_table = r"""\begin{table}[H]
\centering
\caption{Pathwise P\&L results for the reproducible single-path simulation.}
\label{tab:simulationresults}
\begin{tabular}{lrrrr}
\hline
Strategy & Mean P\&L & Std. dev. & Terminal P\&L & Efficiency \\
\hline
""" + "\n".join(table_rows) + r"""
\hline
\end{tabular}
\end{table}
"""

(OUTPUT_DIR / "simulation_results_table.tex").write_text(
    results_table,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 11. Save run information
# ------------------------------------------------------------
run_information = f"""REPRODUCIBLE SINGLE-PATH SIMULATION

Initial stock price: {S0}
Strike: {K}
Physical drift: {MU}
Risk-free rate: {RISK_FREE_RATE}
Pricing volatility: {SIGMA_IMPLIED}
Stock-path volatility: {SIGMA_REALIZED}
Maturity: {MATURITY}
Trading intervals: {NUMBER_OF_STEPS}
Stock-price observations: {NUMBER_OF_STEPS + 1}
Random seed: {RANDOM_SEED}

Initial call price: {call_prices[0]:.6f}
Initial Delta: {deltas[0]:.6f}
Terminal stock price: {prices[-1]:.6f}
Terminal option payoff: {call_prices[-1]:.6f}
"""

(OUTPUT_DIR / "simulation_run_information.txt").write_text(
    run_information,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 12. Display results in the terminal
# ------------------------------------------------------------
print("Simulation completed successfully.")
print(f"Output folder: {OUTPUT_DIR}")
print()
print(run_information)
print("PATHWISE P&L SUMMARY")
print(summary_df.round(6).to_string(index=False))
