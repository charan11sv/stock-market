# News Sentiment Pipeline (Notebook: `sentiment_scores.ipynb`)

## Overview
This notebook builds a **company–day sentiment dataset** by fetching financial news, running **VADER sentiment analysis**, and aggregating scores per company and date. It supports news from curated **RSS feeds** (e.g., Reuters, Bloomberg, Economic Times, WSJ) and may also query **GDELT** for historical coverage. Intermediate results are checkpointed to avoid losing progress during long runs.

---

## Goals
- Collect relevant news for each listed company (NSE/BSE universe).
- Score articles using **VADER** (`compound`, `pos`, `neu`, `neg`).
- Detect **fresh events** (e.g., launches, earnings, announcements).
- Aggregate to **daily sentiment features** per company.
- Save **intermediate** and **final** datasets for downstream modeling.

---

## Inputs
- **`NSE_tickers.csv`** — list of tickers.
- **`namechange.csv`** — ticker→company name mapping (used to build richer search queries).
- (Optional during validation) **`merged_with_weighted_sentiment_final.parquet`** — used only to sanity-check columns in later steps.

> **Assumed columns**  
> - `NSE_tickers.csv`: must include a `SYMBOL` column  
> - `namechange.csv`: must include columns to map `SYMBOL` → `COMPANY` (e.g., `SYMBOL`, `COMPANY`)

---

## Outputs
- **`intermediate_sentiment_dataset.csv`** — checkpointed every N companies (N=2 in the current setup).
- **`company_sentiment_dataset.csv`** — final per-company, per-date sentiment dataset.

---

## Dependencies
- `pandas`
- `feedparser` (for RSS)
- `requests` (if using GDELT)
- `vaderSentiment` (`SentimentIntensityAnalyzer`) for sentiment scoring
- (Optional) `tqdm` for progress bars

---

## Processing Steps (What the notebook does)

1. **Load Companies**
   - Read `NSE_tickers.csv` and join with `namechange.csv` to attach human-readable `COMPANY` names for better news matching.
   - Build a **company list** for iteration (ticker + company name).

2. **Configure News Sources**
   - Define an **RSS feed list** (e.g., Reuters, Bloomberg, Economic Times, WSJ, etc.).
   - (Optionally) Prepare **GDELT** query templates to pull historical articles when RSS is insufficient.

3. **Fetch Articles (Per Company)**
   - For each company:
     - Construct search cues using the **company name** (and sometimes ticker).
     - Pull recent items from each **RSS feed** via `feedparser`.
     - Optionally call **GDELT** (via `requests`) for historical coverage matching the company name.
   - Basic normalization per article:
     - Extract: `published date`, `title`, `summary`, `link/source`.
     - **De-duplicate** by URL/title if necessary.
     - **Filter for relevance** using simple keyword/name checks.

4. **Fresh Event Detection**
   - Flag articles containing **recency/announcement** keywords like:  
     `"just now"`, `"announced"`, `"launch" / "launched"`, `"earnings"`, `"deal"`, `"acquisition"`, etc.
   - Store a Boolean flag, e.g., `is_fresh_event`.

5. **VADER Sentiment Analysis**
   - Initialize `SentimentIntensityAnalyzer`.
   - Build an **analyzable text** per article (e.g., `title + summary`).
   - Compute and store:
     - `compound` ([-1, 1])
     - `pos`, `neu`, `neg` (share of polarity)
   - Optionally clip/guard against empty/null text.

6. **Aggregate to Daily Company Sentiment**
   - Convert published timestamps to **date** (localize if needed).
   - Group by `Company_ID`/`COMPANY` and **Date**, aggregating:
     - `compound_mean` / `compound_median`
     - `pos_mean`, `neg_mean`, `neu_mean`
     - `num_articles` (count)
     - `num_fresh_events` (sum of `is_fresh_event`)
   - Keep both **level-0 (article)** and **level-1 (daily)** artifacts if desired; export the daily table.

7. **Batching & Checkpoints**
   - Process companies in **batches**; after every **2 companies**, append to or overwrite **`intermediate_sentiment_dataset.csv`**.
   - This ensures progress is saved even if fetching takes long or a source rate-limits.

8. **Export Final Dataset**
   - Concatenate all batch results.
   - Drop obvious duplicates / fix datatypes.
   - Save **`company_sentiment_dataset.csv`**.
   - If no data was found for a chunk, print a message (e.g., *"No sentiment data found for the companies processed."*).

---

## Aggregation Schema (Typical Columns)
Final `company_sentiment_dataset.csv` generally includes:
- `Date` — article date (YYYY-MM-DD)
- `Company_ID` — or `SYMBOL`
- `COMPANY` — human-readable name
- `num_articles`
- `compound_mean`, `compound_median`
- `pos_mean`, `neg_mean`, `neu_mean`
- `num_fresh_events`
- (Optional) `sources` / `top_keywords` / `example_headline` for debugging

> Exact columns can vary slightly depending on which sources return data.

---

## Configuration & Tuning Knobs
- **Batch size / checkpoint frequency:** default = 2 companies per checkpoint.
- **Feeds to query:** add/remove RSS endpoints in the feed list.
- **Fresh-event keywords:** extend the keyword list to reflect your use case.
- **Date window:** restrict by published time (e.g., last X days) if needed.
- **Company query format:** try including both `COMPANY` and `SYMBOL` to reduce false positives.

---

## Error Handling & Robustness
- **Per-company try/except** around fetch & parse; failures don’t halt the run.
- **Rate-limiting:** implicit via batching; consider sleeps/backoff if sources throttle.
- **Deduplication:** by URL/title to avoid double counting.
- **Empty batches:** explicitly reported; final CSV still written if earlier batches exist.

---

## Example (Conceptual) Snippets

**VADER scoring per article**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
text = f"{title or ''}. {summary or ''}".strip()
scores = analyzer.polarity_scores(text)
compound, pos, neu, neg = scores["compound"], scores["pos"], scores["neu"], scores["neg"]
