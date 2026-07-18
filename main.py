"""
Real Market Data Delta Hedging Application

"""

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

    rebalance_interval = None  -> unhedged short call
    rebalance_interval = 1     -> daily delta hedge
    rebalance_interval = 5     -> weekly delta hedge
    rebalance_interval = 21    -> monthly delta hedge
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
# 8. PNG figures, optional backup
# ------------------------------------------------------------
def save_line_chart(y, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(dates, y)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


save_line_chart(prices, "Real AAPL Stock Price Path, 2024", "Stock Price", "stock_path.png")
save_line_chart(deltas, "Delta Evolution for AAPL Call Option", "Delta", "delta_evolution.png")
save_line_chart(gammas, "Gamma Evolution for AAPL Call Option", "Gamma", "gamma_evolution.png")
save_line_chart(vegas, "Vega Evolution for AAPL Call Option", "Vega", "vega_evolution.png")
save_line_chart(thetas, "Theta Evolution for AAPL Call Option", "Theta", "theta_evolution.png")

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
# 9. Generate LaTeX/TikZ figures and LaTeX tables
# ------------------------------------------------------------
def tikz_coordinates(y_values):
    return " ".join(f"({i},{float(y):.6f})" for i, y in enumerate(y_values))


def write_tikz_line_plot(filename, y_values, ylabel, caption, label, color="plotblue"):
    coords = tikz_coordinates(y_values)

    tex = f"""\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}
[width=0.82\\textwidth,height=6.2cm,
xlabel={{Trading day}},ylabel={{{ylabel}}},
grid=major,axis lines=left]
\\addplot[thick,{color}] coordinates {{{coords}}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    (OUTPUT_DIR / filename).write_text(tex, encoding="utf-8")


def write_tikz_multi_plot(filename, series, ylabel, caption, label):
    plots = []
    for name, values, color in series:
        coords = tikz_coordinates(values)
        plots.append(f"\\\\addplot[thick,{color}] coordinates {{{coords}}};\n\\\\addlegendentry{{{name}}}")
    plots_code = "\n".join(plots)

    tex = f"""\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}
[width=0.82\\textwidth,height=6.2cm,
xlabel={{Trading day}},ylabel={{{ylabel}}},
grid=major,axis lines=left,
legend style={{at={{(0.02,0.98)}},anchor=north west}}]
{plots_code}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    tex = tex.replace("\\\\addplot", "\\addplot").replace("\\\\addlegendentry", "\\addlegendentry")
    (OUTPUT_DIR / filename).write_text(tex, encoding="utf-8")


write_tikz_line_plot(
    "figure10_6.tex",
    prices,
    "Stock price",
    "Real AAPL stock price path, January 2024--December 2024. The realized path is used to apply the delta hedging analysis to market data.",
    "fig:realstockpath",
    "plotblue"
)

write_tikz_line_plot(
    "figure10_7.tex",
    deltas,
    "Call Delta",
    "Delta evolution for the AAPL call option based on the realized stock path. Delta changes as the stock price and time to maturity change.",
    "fig:realdelta",
    "plotblue"
)

write_tikz_line_plot(
    "figure10_8.tex",
    gammas,
    "Gamma",
    "Gamma evolution for the AAPL call option. The path shows the curvature exposure that remains after first-order Delta hedging.",
    "fig:realgamma",
    "plotblue"
)

write_tikz_line_plot(
    "figure10_9.tex",
    vegas,
    "Vega",
    "Vega evolution for the AAPL call option. The path shows the volatility sensitivity of the option along the realized stock path.",
    "fig:realvega",
    "plotblue"
)

write_tikz_line_plot(
    "figure10_10.tex",
    thetas,
    "Theta",
    "Theta evolution for the AAPL call option. The path shows the time-decay component of the option value along the realized stock path.",
    "fig:realtheta",
    "plotblue"
)

write_tikz_multi_plot(
    "figure10_11.tex",
    [
        ("Unhedged short call", unhedged_pnl, "plotred"),
        ("Daily delta hedge", daily_pnl, "plotblue"),
        ("Weekly delta hedge", weekly_pnl, "plotgreen"),
        ("Monthly delta hedge", monthly_pnl, "plotpurple"),
    ],
    "Portfolio P\\&L",
    "Unhedged and delta-hedged portfolio values for the realized AAPL path. Delta hedging reduces directional variation, but residual P\\&L remains.",
    "fig:realhedgedunhedged"
)

# LaTeX tables. These use simple table formatting so they compile in most reports.
# If your main report already defines booktabs/tabularx, these tables will match cleanly.
def money(x):
    return f"{float(x):,.4f}"


def escape_latex_text(s):
    return str(s).replace("&", "\\\\&")


summary_latex_rows = []
for _, row in summary_df.iterrows():
    summary_latex_rows.append(
        f"{escape_latex_text(row['Strategy'])} & {money(row['Mean P&L'])} & {money(row['Standard Deviation'])} & {money(row['Variance'])} & {money(row['Terminal P&L'])} \\\\" 
    )

summary_table_tex = """\\begin{table}[H]
\\centering
\\caption{Realized P\\&L statistics for the unhedged and delta-hedged AAPL short-call portfolios.}
\\label{tab:realpnlstatistics}
\\begin{tabular}{lrrrr}
\\hline
Strategy & Mean P\\&L & Std. dev. & Variance & Terminal P\\&L \\\\
\\hline
""" + "\n".join(summary_latex_rows) + """
\\hline
\\end{tabular}
\\end{table}
"""
(OUTPUT_DIR / "table10_1.tex").write_text(summary_table_tex, encoding="utf-8")


eff_latex_rows = []
for _, row in efficiency_df.iterrows():
    eff_latex_rows.append(
        f"{escape_latex_text(row['Strategy'])} & {float(row['Hedging Efficiency']):.2%} \\\\"
    )

eff_table_tex = """\\begin{table}[H]
\\centering
\\caption{Hedging efficiency for the realized AAPL short-call portfolios.}
\\label{tab:realhedgingefficiency}
\\begin{tabular}{lr}
\\hline
Hedging frequency & Hedging efficiency \\\\
\\hline
""" + "\n".join(eff_latex_rows) + """
\\hline
\\end{tabular}
\\end{table}
"""
(OUTPUT_DIR / "table10_2.tex").write_text(eff_table_tex, encoding="utf-8")

# ------------------------------------------------------------
# 10. Generate Section 10.6 text
# ------------------------------------------------------------
summary_lookup = summary_df.set_index("Strategy")
eff_lookup = efficiency_df.set_index("Strategy")

section_text = f"""\\subsection{{Real Market Data Application}}

We now repeat the simulation using one realized historical stock price path instead of the illustrative path used above. The stock selected is Apple Inc. (AAPL), and daily adjusted closing prices are downloaded from Yahoo Finance for the period January 2024 to December 2024. The option is again treated as a European call written on the underlying stock. The strike price is set equal to the initial stock price, so that $K = S_0$. The annualized volatility is estimated from the realized daily log returns of AAPL over the sample period, giving $\\sigma = {sigma:.4f}$, while the risk-free rate is kept at $r = {r:.4f}$.

For each trading day, the Black--Scholes call price, Delta, Gamma, Vega and Theta are computed using the same formulas as before. Vega and Theta are reported in the same scale as the original report formulas. Therefore, Vega is not divided by 100 and Theta is not divided by 252. Figure~\\ref{{fig:realstockpath}} shows the realized AAPL stock price path, while Figure~\\ref{{fig:realdelta}} shows the corresponding Delta evolution of the call option. Figure~\\ref{{fig:realgamma}}, Figure~\\ref{{fig:realvega}} and Figure~\\ref{{fig:realtheta}} report the real Gamma, Vega and Theta evolution.

\\input{{outputs/figure10_6.tex}}

\\input{{outputs/figure10_7.tex}}

\\input{{outputs/figure10_8.tex}}

\\input{{outputs/figure10_9.tex}}

\\input{{outputs/figure10_10.tex}}

The same short-call hedging logic is then applied to the real data. Four cases are compared: an unhedged short call, a daily delta hedge, a weekly delta hedge and a monthly delta hedge. In each hedged portfolio, the stock position is adjusted to the option Delta at the chosen rebalancing frequency. The portfolio values are calculated along the same realized path, so no additional simulated paths are introduced. Figure~\\ref{{fig:realhedgedunhedged}} compares the realized P\\&L time series of the four strategies.

\\input{{outputs/figure10_11.tex}}

Table~\\ref{{tab:realpnlstatistics}} reports the realized P\\&L statistics obtained from the portfolio value time series. The values are produced directly by the Python code and saved in the output files.

\\input{{outputs/table10_1.tex}}

The hedging efficiency is calculated using the same formula as before,

\\[
\\text{{Hedging Efficiency}} = 1 - \\frac{{\\operatorname{{Var}}(\\text{{hedged P\\&L}})}}{{\\operatorname{{Var}}(\\text{{unhedged P\\&L}})}}.
\\]

Table~\\ref{{tab:realhedgingefficiency}} reports the resulting efficiencies.

\\input{{outputs/table10_2.tex}}

The real market data application confirms the result observed in the illustrative simulation. The unhedged short-call portfolio remains exposed to the movement of the underlying stock, while delta hedging reduces the variability of the portfolio P\\&L. Daily rebalancing gives the largest reduction in variance, followed by weekly and monthly rebalancing. The Gamma, Vega and Theta paths show that important sensitivities remain even after Delta is hedged. This is consistent with the earlier discussion that delta hedging removes first-order spot exposure but does not remove curvature risk, volatility risk or time decay.
"""

(OUTPUT_DIR / "section_10_6_generated.tex").write_text(section_text, encoding="utf-8")
(PROJECT_DIR / "section_10_6.tex").write_text(section_text, encoding="utf-8")

print("Completed successfully.")
print(f"AAPL prices saved to: {DATA_DIR / 'aapl_2024_prices.csv'}")
print(f"Figures, tables and CSV files saved to: {OUTPUT_DIR}")
print(f"Updated LaTeX section saved to: {PROJECT_DIR / 'section_10_6.tex'}")

print("\nP&L Summary:")
print(summary_df.round(6))

print("\nHedging Efficiency:")
print(efficiency_df.round(6))
