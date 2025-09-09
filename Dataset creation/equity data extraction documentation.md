Equity Data Extraction Notebook Documentation:

Overview

This notebook is responsible for extracting ticker symbols and collecting historical stock price data for companies listed on the NSE (National Stock Exchange) and BSE (Bombay Stock Exchange). It serves as the data ingestion stage of the pipeline, preparing clean and reliable market data for further sentiment analysis, feature engineering, and modeling.

Steps Performed

Import Dependencies

Load required Python libraries for data handling and API calls.

Load BhavCopy Data (BSE Example)

Read raw BhavCopy CSV files (e.g., BhavCopy_BSE_CM_0_0_0_20240905.csv).

These files contain daily trading data such as open, high, low, close, volume, and ticker symbol.

Extract Unique Ticker Symbols

Extract the TckrSymb column from the BhavCopy file.

Save the cleaned list of unique tickers into bse_ticker_symbols.csv for reuse.

Load Predefined Ticker Lists (NSE & BSE)

Read NSE_tickers.csv and bse_tickers.csv.

Convert the SYMBOL column values into Python lists for batch processing.

Fetch Historical Stock Data

Use the custom function fetch_stock_data to download historical market data.

The function is executed separately for:

NSE tickers (suffix .NS)

BSE tickers (suffix .BO)

Data is fetched from external APIs (e.g., Yahoo Finance) and saved locally.

Handle and Log Failures

Any ticker symbols that fail during data retrieval (e.g., delisted or invalid tickers) are logged.

Separate CSV files are generated to track these failures:

failed_NSE_tickers.csv

failed_BSE_tickers.csv

Outputs Generated

Ticker Lists:

bse_ticker_symbols.csv

(Uses NSE_tickers.csv and bse_tickers.csv as inputs)

Historical Stock Data:

Retrieved for all valid NSE and BSE tickers.

Failure Logs:

failed_NSE_tickers.csv

failed_BSE_tickers.csv

Role in the Pipeline

This notebook ensures:

Clean ticker symbol extraction

Reliable historical stock data collection

Error tracking and logging

It is the first stage of the workflow, and its outputs are later combined with sentiment data and engineered features in subsequent notebooks.
