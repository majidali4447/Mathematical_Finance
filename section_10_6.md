## 10.6 Real Market Data Application

In this subsection, the same simulation methodology is repeated using real market data instead of the illustrative stock path used in the previous subsections. The purpose is not to introduce a new model, but only to examine whether the same delta-hedging behaviour is observed when the underlying asset follows an actual historical price path. For this purpose, daily adjusted AAPL prices from January 2024 to December 2024 are downloaded from Yahoo Finance and used as the realized stock price path.

The option considered is again a European call option written on the stock. The strike price is set equal to the initial stock price, so that the option is approximately at-the-money at the beginning of the period. The risk-free rate is kept fixed at \(r=2\%\), while the volatility parameter \(\sigma\) is estimated from the historical daily log returns of AAPL and annualized using 252 trading days. For each trading day, the Black--Scholes call price and the corresponding Delta, Gamma, Vega and Theta are computed. The resulting stock path and Delta evolution are shown in Figures 10.6 and 10.7.

*Figure 10.6: Real AAPL stock price path, January 2024--December 2024.*

*Figure 10.7: Delta evolution for the AAPL European call option.*

The short-call position is then evaluated under four strategies. The first strategy is the unhedged short call, where no stock position is taken after the call is sold. The remaining three strategies apply delta hedging with daily, weekly and monthly rebalancing, respectively. In each case, the hedge is updated using the Black--Scholes Delta computed from the realized AAPL price on that trading day. Therefore, all strategies are evaluated on exactly the same realized historical path.

Figure 10.8 compares the portfolio P&L paths. The unhedged short-call position remains directly exposed to movements in the stock price. By contrast, the delta-hedged portfolios show smaller fluctuations, because the stock position offsets part of the option exposure. The daily hedge adjusts most frequently and therefore follows the theoretical delta-neutral position more closely. The weekly and monthly hedges reduce risk as well, but leave more residual risk between rebalancing dates.

*Figure 10.8: Unhedged and delta-hedged portfolio values using real AAPL data.*

Figure 10.9 reports the realized terminal P&L for the four strategies. Since this subsection uses only one historical path, the terminal result should be interpreted as the realized outcome for the 2024 AAPL path, not as an average over simulated paths.

*Figure 10.9: Terminal P&L comparison for the unhedged and delta-hedged strategies.*

Table 10.3 reports the mean P&L, standard deviation, variance and terminal P&L calculated from the realized portfolio P&L time series. When `main.py` is executed, the completed table is written automatically from `outputs/pnl_summary.csv` into `section_10_6.md` and `outputs/section_10_6_generated.md`. No numerical value is manually inserted.

The hedging efficiency is computed using the same formula as before:

\[
\text{Hedging Efficiency}
=
1-
\frac{\operatorname{Var}(\text{hedged P\&L})}
{\operatorname{Var}(\text{unhedged P\&L})}.
\]

The resulting efficiencies are reported in Table 10.4. When `main.py` is executed, the completed table is written automatically from `outputs/hedging_efficiency.csv` into `section_10_6.md` and `outputs/section_10_6_generated.md`. No numerical value is manually inserted.

Overall, the real-data application supports the same interpretation as the illustrative simulation. Delta hedging reduces the variability of the short-call portfolio relative to the unhedged position. The reduction is strongest when the hedge is rebalanced daily, while weekly and monthly hedging leave progressively larger residual risk. This confirms, on one realized historical AAPL path, the practical importance of dynamic rebalancing in managing the risk of a short option position.
