# Stock market data preparation

Prepare stock prices, trading volume, and headline sentiment for a Power BI visualization project.

## Setup

Tested with Python 3.13, PyTorch 2.15.0+cu134, and Transformers 5.22.1. Sentiment scoring uses an NVIDIA GPU with CUDA and BF16; model weights occupy roughly 8 GB plus runtime memory.

Run these commands in PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu134
.\.venv\Scripts\python.exe -m pip install -e .
```

The model downloads roughly 9 GB on first use and is cached locally.

## Input data

Place the source files at:

- `data/raw/full_history/`: one price CSV per ticker.
- `data/raw/nasdaq_exteral_data.csv`: news export.

Raw data, processed data, and reports are excluded from Git.

## Run the pipeline

Run preparation first, then sentiment scoring:

```powershell
.\.venv\Scripts\python.exe -m stock_market_sentiment
.\.venv\Scripts\python.exe -m stock_market_sentiment.sentiment
```

Preparation overwrites the news output, so rerun sentiment scoring after rebuilding the dataset. The scorer reads `data/processed/news_events.parquet` independently of preparation settings and enriches it in place.

## Dataset scope

Defaults in [config/default.json](./config/default.json) select up to **30 tickers** during **2015–2020**. Coverage is measured after article deduplication and price matching.

| Coverage criterion | Minimum or limit |
| --- | --- |
| First-to-last article span | 24 calendar months |
| Months with an article | 75% of the configured window |
| Consecutive empty months | At most 3, including window edges |
| Distinct publication dates | 60 |
| Articles per ticker | 100 |

Eligible tickers rank by covered months, then distinct publication dates, then ticker alphabetically. Both outputs use the selected ticker universe.

Articles are deduplicated per ticker using URL, or publication timestamp and title when the URL is missing. Body text is excluded; articles associated with multiple tickers remain eligible.

Publication timestamps and dates are retained. Each article links to the first valid price session **strictly after** its UTC publication date, within seven calendar days. Trading dates come from observed price histories.

## Outputs

| File | Contents |
| --- | --- |
| `data/processed/daily_prices.parquet` | Daily OHLC prices, adjusted close, volume, return percentage, volume ratio, and coverage fields |
| `data/processed/news_events.parquet` | Article–ticker records with publication dates, next trading date, headline, URL, publisher, and sentiment |
| `reports/quality_report.json` | Preparation counts, validation results, and ticker coverage |
| `reports/sentiment_report.json` | Model settings, label counts, and processing speed |

Dates and numeric columns retain native types. Publication timestamps are UTC. Returns use adjusted close; volume ratios compare today's volume with the mean of up to 20 preceding sessions.

See the [data preparation reference](./docs/data_preparation_reference.md) for report-count definitions, methodology, field types, and limitations.

## Sentiment

[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) evaluates each headline for its associated ticker with thinking disabled.

Each input includes the target's name and security type from [config/securities.json](./config/securities.json), using identities for 2015–2020. Add entries if the selected tickers change; missing metadata stops scoring before the model loads. Reports record the prompt, examples, and target metadata.

| Label | Score |
| --- | --- |
| `negative` | -1 |
| `neutral` | 0 |
| `positive` | +1 |
| `mixed` | null |
| `insufficient_information` | null |

Scores are category codes; confidence values are not exported. Mixed means opposing material implications. Insufficient information means the headline lacks enough target-specific evidence.

Progress reports records scored, speed, and estimated time remaining after each 8,192-row chunk. Model loading and initial compilation precede the first update.

## Power BI

Use date hierarchies for year → quarter → month → day drilldown. Show article counts and insufficient-information counts alongside average sentiment. Exclude null scores from averages; missing news is not neutral.

The tickers form a coverage-selected sample. Sentiment and price movements show associations, not causation, and model labels can be incorrect.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary fixtures and model doubles without GPU inference or model downloads. See [source code](./src/stock_market_sentiment/) and [unit tests](./tests/unit/) for implementation details.
