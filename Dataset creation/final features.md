# Final Dataset — Feature Glossary
_Produced by `Dataset creation main.ipynb` (feature engineering & final merge)._

Below is a one-line description of every column in the final dataset.

| Feature | Description |
|---|---|
| `Date` | Calendar date of the observation. |
| `Company_ID` | Unique identifier for the company/security. |
| `COMPANY` | Human-readable company name. |
| `Open_x` | Company’s opening price for the day. |
| `High_x` | Company’s intraday high price. |
| `Low_x` | Company’s intraday low price. |
| `Close_x` | Company’s closing price for the day. |
| `Adj Close_x` | Split/dividend-adjusted closing price. |
| `Volume_x` | Traded volume for the day. |
| `VWAP_x` | Volume-weighted average price for the day. |
| `MACD_x` | MACD value (trend/momentum; e.g., 12–26 EMA spread). |
| `MACD_Signal_x` | MACD signal line (e.g., 9-period EMA of MACD). |
| `MACD_Hist_x` | MACD histogram (MACD minus signal). |
| `SMA_20` | 20-period simple moving average of close. |
| `EMA_20` | 20-period exponential moving average of close. |
| `WMA_20` | 20-period weighted moving average of close. |
| `Upper_Band` | Upper Bollinger Band (volatility envelope). |
| `Middle_Band` | Middle Bollinger Band (typically SMA). |
| `Lower_Band` | Lower Bollinger Band (volatility envelope). |
| `ATR_14` | 14-period Average True Range (volatility). |
| `RSI_14` | 14-period Relative Strength Index (momentum). |
| `Stoch_K` | Stochastic oscillator %K (momentum). |
| `Stoch_D` | Stochastic oscillator %D (signal for %K). |
| `OBV` | On-Balance Volume (volume-price flow indicator). |
| `SAR` | Parabolic SAR (stop-and-reverse trend indicator). |
| `CCI_20` | 20-period Commodity Channel Index (deviation/momentum). |
| `ROC` | Rate of Change over the configured base window. |
| `Price_Change` | Day-over-day absolute price change. |
| `Price_Change_Percent` | Day-over-day percent price change. |
| `Volume_Change` | Day-over-day absolute volume change. |
| `Volume_Change_Percent` | Day-over-day percent volume change. |
| `High_Low_Spread` | Intraday spread (`High` − `Low`). |
| `Close_Open_Ratio` | Close divided by Open (relative move). |
| `High_Low_Ratio` | High divided by Low (intraday range ratio). |
| `Price_Range` | Daily range measure (e.g., high–low spread). |
| `Price_Volume_Product` | Product of price and volume (liquidity proxy). |
| `Lag_Close_x_1` | Previous day’s close price (lag-1). |
| `Lag_Volume_x_1` | Previous day’s volume (lag-1). |
| `Lag_Close_x_5` | Close price lagged by 5 trading days. |
| `Lag_Volume_x_5` | Volume lagged by 5 trading days. |
| `Lag_Close_x_10` | Close price lagged by 10 trading days. |
| `Lag_Volume_x_10` | Volume lagged by 10 trading days. |
| `Day_of_Week` | Day of week (0–6 or Mon–Sun). |
| `Month` | Calendar month number (1–12). |
| `Quarter` | Calendar quarter (1–4). |
| `Week_of_Year` | ISO week number. |
| `Rolling_Std_Close` | Rolling standard deviation of close (volatility). |
| `Rolling_Std_Volume` | Rolling standard deviation of volume. |
| `Rolling_Mean_Close` | Rolling mean of closing price. |
| `Rolling_Max_Close` | Rolling maximum of closing price. |
| `Rolling_Min_Close` | Rolling minimum of closing price. |
| `Rolling_Price_Range` | Rolling range of price (e.g., max − min). |
| `Z_Score_Close` | Z-score of close vs. rolling mean/std. |
| `Momentum` | Price momentum over a defined lookback. |
| `Price_Acceleration` | Change in momentum (second-order move). |
| `OBV_Change` | Day-over-day change in On-Balance Volume. |
| `Volume_Ratio` | Current volume vs. rolling average volume. |
| `Rolling_Return` | Return computed over a rolling window. |
| `Kurtosis` | Rolling kurtosis of returns (tail heaviness). |
| `Skewness` | Rolling skewness of returns (asymmetry). |
| `Sharpe_Ratio` | Rolling Sharpe ratio (risk-adjusted return). |
| `Fib_Level_0.236` | Fibonacci retracement level at 23.6% for window. |
| `Fib_Level_0.382` | Fibonacci retracement level at 38.2% for window. |
| `Fib_Level_0.618` | Fibonacci retracement level at 61.8% for window. |
| `Pivot_Point` | Classic pivot point from prior OHLC. |
| `ROC_5` | 5-day rate of change. |
| `ROC_10` | 10-day rate of change. |
| `sentiment_score` | Daily aggregated sentiment score from news. |
| `Lagged_Sentiment_1` | Sentiment score lagged by 1 day. |
| `Lagged_Sentiment_5` | Sentiment score lagged by 5 days. |
| `Lagged_Sentiment_10` | Sentiment score lagged by 10 days. |
| `Weighted_Sentiment_Metric` | Composite metric using learned linear weights. |
| `Market_Close` | Benchmark market index closing level (local). |
| `Market_Returns` | Daily returns of the benchmark market index. |
| `Global_Close` | Selected global index closing level. |
| `Global_Returns` | Daily returns of the selected global index. |
| `Rolling_Beta` | Rolling beta of stock vs. benchmark market. |
| `Rolling_Market_Correlation` | Rolling correlation with local market returns. |
| `Rolling_Global_Correlation` | Rolling correlation with global index returns. |
| `INR_USD_Close` | INR/USD exchange rate close (USD per INR). |
| `Crude_Oil_Close` | Crude oil benchmark closing price. |
| `Gold_Close` | Gold benchmark closing price. |
| `VIX` | Volatility index level (e.g., India VIX). |
| `Crude_Oil` | Crude oil benchmark level/series (companion series). |
| `Gold` | Gold benchmark level/series (companion series). |
| `USD_INR` | USD/INR exchange rate (INR per USD). |
| `SP500` | S&P 500 index level. |
| `Nifty50` | Nifty 50 index level. |
| `FTSE100` | FTSE 100 index level. |
| `DAX` | DAX index level. |
| `Nikkei225` | Nikkei 225 index level. |
| `Row_ID` | Unique row identifier. |
| `cluster` | Cluster label assigned by the clustering model. |
