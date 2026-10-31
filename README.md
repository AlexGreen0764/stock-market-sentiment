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
| `config/default.json` | Default tickers, date range, output directory, and coverage thresholds |
| `Final Project_SP.pdf` | Supplied course project brief |

The two raw input collections are excluded from Git because the article export alone is about 23 GB. Preserve them locally. The origin and licensing of the historical price files and the article export still need to be confirmed by the data provider; article URLs point to individual Nasdaq pages but do not establish how the bulk file was collected.

## Run

Requires Python 3.10 or newer and PyArrow. From the repository root, install the package and its dependency in editable mode once:

```powershell
python -m pip install -e .
```

Then run:

```powershell
python -m stock_market_sentiment
```

The candidate symbols are `AAPL MSFT NVDA AMZN GOOG TSLA`, and the default date window is **January 1, 2009 through December 31, 2020**, inclusive. A prior data-quality review found article publisher and stock ticker fields more complete in this period; the later years should not be presented as equally complete without a fresh audit. The coverage rules below determine which candidates appear in the final files. `TSLA` starts trading in 2010, so it has a shorter available series. Select other candidates with `--tickers`, for example:

```powershell
python -m stock_market_sentiment --tickers AAPL MSFT NVDA
```

The defaults come from `config/default.json`. CLI arguments override the file, for example:

```powershell
python -m stock_market_sentiment --start-date 2018-01-01 --end-date 2020-12-31
```

Each run writes exactly two Parquet tables. The pipeline calculates returns and volume history from the full available price history before applying the requested date range. The date filter limits exported rows, but the pipeline still reads the complete article file to find matching records, which can take time. Use `python -m unittest discover -s tests` to run the focused tests.

## Ticker coverage filter

Coverage is measured after article date checks, duplicate removal, and matching each article to a later price session inside the selected date range. A covered quarter has at least one retained candidate article. The default rules, editable in `config/default.json`, require a ticker to have:

- articles in at least **24 calendar quarters**;
- articles spanning at least **8 distinct years**;
- a **median of at least 3 articles** among its covered quarters; and
- at least **60 distinct publication dates**.

Only tickers meeting all four rules appear in both final tables. The price table still includes their trading days with no news, which allows comparison with days after news. The separate audit report lists every candidate's coverage and pass/fail status. If no ticker qualifies, the run writes the audit report and stops without publishing empty final tables.

## Output and calculations

| File | Grain and key | Main fields |
| --- | --- | --- |
| `daily_prices.parquet` | One row per retained ticker and trading date | OHLC, adjusted close, shares traded, adjusted-close return %, volume / mean of up to 20 prior sessions |
| `news_events.parquet` | One row per deduplicated article and retained ticker | `published_at_utc`, `article_date_utc`, `next_trading_date`, title, URL, publisher |

The `date`, `article_date_utc`, and `next_trading_date` columns are stored as dates; `published_at_utc` is stored as a UTC timestamp. Price and volume fields have numeric types. This makes the files ready for Power BI date relationships without parsing date text. `reports/quality_report.json` records input, rejected, and retained row counts plus ticker coverage; it is not a third visualization table.

Returns use adjacent available trading sessions: `100 × (adjusted close / previous adjusted close − 1)`. The first return and first volume ratio for each ticker are null. The ratio's denominator uses the preceding 1–20 sessions and excludes the current session. Invalid prices, negative volume, duplicate ticker dates, invalid article timestamps, duplicate article URLs, and news without a later price session are recorded in the quality report. Duplicate article URLs are checked within each ticker.

Each article retains its original UTC publication date in `article_date_utc` and receives a separate `next_trading_date`: the first date strictly after publication that appears in that ticker's price history. This skips weekends and market holidays represented by gaps in the price history. It also skips any dates missing for other reasons, so inspect unexpected long gaps before interpreting the linked return. For example, an article dated Friday, December 22, 2023 maps to Tuesday, December 26 because the intervening days include a weekend and Christmas Day. Both the publication date and assigned trading date must fall inside the selected window. This rule avoids treating publication times with uncertain exchange-local timing as same-day news. It may miss same-day reactions and can group several articles on one trading day. Comparing linked news with `return_pct` shows association only: market-wide events, other news, and stock-specific fundamentals also affect prices. The price files end in December 2023 for the default set; the available `GOOGL.csv` ends in April 2020, so the default uses `GOOG.csv`.
