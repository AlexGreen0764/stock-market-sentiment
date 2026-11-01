# Stock market data preparation

This repository prepares the supplied stock price and Nasdaq article files for a visualization project about price trends, trading volume, and the association between news and subsequent price moves. It produces data tables and a quality report; it does not make trading recommendations or claim that articles cause price changes.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/stock_market_sentiment/` | Python data preparation package and command line entry point |
| `tests/unit/` | Focused parser, cleaning, and date alignment checks |
| `data/raw/full_history/` | Supplied per-symbol daily OHLCV CSV files (local input) |
| `data/raw/nasdaq_exteral_data.csv` | Supplied Nasdaq article export (local input; filename retained as received) |
| `data/processed/` | Two generated visualization tables |
| `reports/quality_report.json` | Generated coverage and data-quality audit |
| `config/default.json` | Date range, output directory, and news coverage thresholds |
| `Final Project_SP.pdf` | Supplied course project brief |

The two raw input collections are excluded from Git because the article export alone is about 23 GB. Preserve them locally. The origin and licensing of the historical price files and the article export still need to be confirmed by the data provider; article URLs point to individual Nasdaq pages but do not establish how the bulk file was collected.

## Run

Requires Python 3.10 or newer, PyArrow, PyTorch, and Transformers. From the repository root, install the package and its dependencies in editable mode once:

```powershell
python -m pip install -e .
```

Then run:

```powershell
python -m stock_market_sentiment
```

Every ticker with a CSV file in `data/raw/full_history/` is evaluated; there is no configured ticker list. Both news identifiers and price filename stems are trimmed and uppercased for matching and output, while the original file paths are preserved for reading. For example, `nvda.csv` matches `NVDA` news. Multiple files resolving to the same normalized ticker cause an explicit error. Historical ticker aliases and multiple symbols in one field are not automatically resolved. The default article publication window is **January 1, 2009 through December 31, 2020**, inclusive. A prior data-quality review found article publisher and stock ticker fields more complete in this period; the later years should not be presented as equally complete without a fresh audit.

The defaults come from `config/default.json`. CLI arguments override the file, for example:

```powershell
python -m stock_market_sentiment --start-date 2018-01-01 --end-date 2020-12-31
```

Each run writes exactly two Parquet tables. The pipeline scans the complete article file once, retaining matching in-window article metadata in memory but discarding article bodies. It then processes and writes one ticker's prices at a time, calculating returns and volume history before applying the requested date range. A full run can take time and requires memory for the matching article metadata. Use `python -m unittest discover -s tests` to run the focused tests with temporary fixtures.

## Score news sentiment

After preparation, run the separate sentiment stage. It reads the existing news Parquet file, so there is no need to rescan the raw article export:

```powershell
python -m stock_market_sentiment.sentiment
```

This uses [AnkitAI/FinSense-ModernBERT-Financial-News-Sentiment-Analysis](https://huggingface.co/AnkitAI/FinSense-ModernBERT-Financial-News-Sentiment-Analysis) locally. The first scoring run downloads its tokenizer and model weights from Hugging Face. It scores the `title` column, not article bodies, because the model is designed for financial sentences and headlines and the prepared data retains titles. Input is truncated to 128 tokens, matching the training length described in the model card.

The defaults target an i5-13600KF, RTX 5070 (12 GB), and 32 GB RAM: CUDA inference with BF16 weights, SDPA attention, a batch size of 128, six PyTorch CPU threads, and 8,192-record Parquet chunks. Batch size 128 was fastest in a small synthetic GPU benchmark (about 3,267 headline inferences/second versus 2,120 at batch size 32); this is not a full-dataset speed estimate. Install the CUDA build in your project environment before running:

```powershell
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu134
python -m stock_market_sentiment.sentiment
```

The scorer runs on CUDA only and uses fixed settings in `sentiment.py`; there are no command-line tuning parameters. It reads `data/processed/news_events.parquet` and writes `reports/sentiment_report.json`. The model and tokenizer both use the pinned commit `fa8b21b6b6549ca70263ee96b9ec52aba67cd2b4`, without an extra API call to resolve a revision. CUDA availability and the model's expected label mapping are checked before scoring.

The stage streams Parquet chunks and performs batched inference, preserving every existing row, column, and date type. Within each chunk, exact duplicate titles are scored once, tokenization runs once on all unique titles, and inputs are grouped by token length to minimize padding. On a 1,024-headline sample, length grouping reduced median inference time from 0.672 to 0.262 seconds; pinned transfers gave no improvement and were removed. Class probabilities are calculated in FP32 and copied to CPU once per chunk. Predictions are restored to their original row order. Memory use is bounded by a chunk rather than the full dataset. The tokenizer uses six Rayon threads unless `RAYON_NUM_THREADS` is explicitly set.

The stage writes a temporary Parquet file, then replaces `news_events.parquet` after all records score successfully. A failed inference leaves the original file intact. Rerunning replaces existing sentiment columns without duplicating them. Repeating preparation recreates the unscored news file, so run sentiment scoring after your final preparation run. Progress includes records per second and an estimated remaining time; estimates exclude model loading and can change with headline lengths.

| Added field | Meaning |
| --- | --- |
| `sentiment_score` | `P(positive) - P(negative)`, from -1 to +1 |
| `sentiment_label` | Most probable class: negative, neutral, or positive |
| `sentiment_confidence` | Probability of the most probable class |
| `sentiment_negative`, `sentiment_neutral`, `sentiment_positive` | All three class probabilities |

Scores and probabilities are numeric float32 columns for Power BI. Confidence is a model probability, not a calibrated accuracy estimate or a stock-return prediction. A headline's overall tone is assigned to its article–ticker records; it is not a separate sentiment judgment about each company mentioned. The source has no sentiment labels against which to measure accuracy, so review a sample of your own headlines before interpreting the outputs.

`reports/sentiment_report.json` records the pinned model commit, input column, truncation length, record count, class distribution, mean score, throughput, and elapsed time. `headline_inferences` counts unique titles per chunk (not globally unique articles). Model provenance is also stored in the Parquet schema metadata. This audit is separate from `quality_report.json`, which continues to describe preparation. The two visualization files remain the price table and the enriched news table.

## Ticker coverage filter

Coverage is measured after article date checks, title checks, price matching, and duplicate removal. A covered month contains at least one eligible article. The span includes every calendar month from the first eligible article month through the last, counting both endpoints and all empty months. These are practical starting thresholds for visualization, not statistical guarantees. Defaults in `config/default.json` require:

- a span of at least **24 calendar months**;
- articles in at least **75% of months** in that span;
- no run longer than **3 consecutive empty months**;
- at least **60 distinct publication dates**; and
- at least **100 deduplicated articles**.

Only tickers meeting all five rules appear in the news table. Valid price rows are retained regardless of news coverage. Each price row includes `news_coverage_pass`, `news_coverage_start`, and `news_coverage_end`; the bounds describe the first and last eligible article dates even if a ticker fails. Bounds are null when no eligible articles exist. Use the flag and bounds when comparing news and returns in Power BI; absence of recorded articles does not establish absence of news. If no ticker qualifies, the price table is still written and the news table is empty with its typed schema.

The report separates price tickers from news-qualified (`retained_tickers`) and news-rejected tickers. It includes failed criteria, yearly eligible article counts, monthly coverage metrics, and per-ticker counts before and after publication-date filtering. `price_rows_valid` counts cleaned full-history input; `price_rows_kept` counts the actual exported price rows.

`price_validation_by_ticker` breaks price rejections into parsing failures, nonfinite prices, nonpositive prices, invalid volumes, and inconsistent OHLC bounds. Each invalid row has one reason; duplicate dates are counted separately. `invalid_price_examples` contains at most three original rows per rejection reason across all tickers. Volumes such as `1604200.0` or `1.6042e6` are accepted when they represent exact nonnegative integers within the signed 64-bit range; fractional values are rejected rather than rounded.

`retained_news_quality` reports missing publishers and URLs specifically in the final export, including the missing-publisher fraction (null if no news qualifies). `retained_news_by_year` reports article–ticker record counts and the number of qualifying tickers with articles each year. These counts are not unique articles across tickers. Coverage can vary substantially by year; compare activity within documented coverage periods instead of interpreting aggregate news growth as a consistent measure over time. Missing publishers remain flagged rather than causing otherwise usable records to be dropped.

## Output and calculations

| File | Grain and key | Main fields |
| --- | --- | --- |
| `daily_prices.parquet` | One row per price-file ticker and valid trading date in the window, plus sessions linked by retained news | OHLC, adjusted close, shares traded, adjusted-close return %, volume / mean of up to 20 prior sessions, news coverage flag and bounds |
| `news_events.parquet` | One row per deduplicated article and news-qualified ticker | `published_at_utc`, `article_date_utc`, `next_trading_date`, title, URL, publisher, `has_url`, `has_publisher` |

The `date`, `article_date_utc`, and `next_trading_date` columns are stored as dates; `published_at_utc` is stored as a UTC timestamp. Price and volume fields have numeric types. This makes the files ready for Power BI date relationships without parsing date text. `reports/quality_report.json` records input, rejected, and retained row counts plus ticker coverage; it is not a third visualization table.

Returns use adjacent available trading sessions: `100 × (adjusted close / previous adjusted close − 1)`. The first return and first volume ratio for each ticker are null. The ratio's denominator uses the preceding 1–20 sessions and excludes the current session. Invalid prices, negative volume, duplicate ticker dates, invalid article timestamps, duplicate article URLs, and news without a later price session are recorded in the quality report. Duplicate article URLs are checked within each ticker.

Each article retains its original UTC publication date in `article_date_utc` and receives a separate `next_trading_date`: the first date strictly after publication that appears in that ticker's price history. This skips weekends and market holidays represented by gaps in the price history. Articles before the first price date, without a later session, or with a gap longer than **7 calendar days** are excluded. This gap limit is a data-quality guard, not an exchange holiday calendar. Articles with empty titles are excluded; missing URLs and publishers are flagged and retained. Articles without URLs are deduplicated by publication timestamp and title within each ticker.

Only publication dates must fall in the selected window. A retained late-2020 article can link to an early-2021 session, and that session is included in the price output. The rule excludes same-day reactions and can group several articles on one trading day. Comparing linked news with `return_pct` shows association only: market-wide events, other news, and stock-specific fundamentals also affect prices.
