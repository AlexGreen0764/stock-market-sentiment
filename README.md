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
- `data/raw/symbols_valid_meta.csv`: security names and stock/ETF types, with `Symbol`, `Security Name`, and `ETF` (`Y`/`N`) columns. The [2020 Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?select=symbols_valid_meta.csv) provides this file.

The catalog is the sole source of target names and security types. Its snapshot does not resolve historical symbol changes.

Raw data, intermediate files, processed data, and reports are excluded from Git.

## Run the pipeline

Run preparation first, then sentiment scoring:

```powershell
.\.venv\Scripts\python.exe -m stock_market_sentiment
.\.venv\Scripts\python.exe -m stock_market_sentiment.sentiment
```

Preparation cleans and ranks eligible candidates into `data/interim/prepared/<ticker>/` and records preparation counts in `data/interim/manifest.json`.

Sentiment scoring visits complete tickers in that order until 30 qualify or candidates are exhausted. The model loads once when inference is needed. It publishes matching news/price universes and reports after selection, keeping null-scored articles. A smaller qualifying set is exported if candidates run out; zero qualifiers preserves existing final outputs and raises an error.

Rerun the second command to resume. Completed ticker scores are reused when candidate content, model, prompt and target identity match. Changing final coverage thresholds alone does not repeat inference.

Both commands accept `--root`, `--config`, `--output`, `--start-date` and `--end-date`. Use matching preparation settings. Rebuild candidates when raw data, dates or cleaning rules change; rerun sentiment when the prompt or final coverage thresholds change.

## Dataset scope

Defaults in [config/default.json](./config/default.json) aim for **30 tickers** during **2015–2020**. The first screen uses news coverage after article deduplication and price matching.

| Coverage criterion | Minimum or limit |
| --- | --- |
| First-to-last article span | 24 calendar months |
| Months with an article | 75% of the configured window |
| Consecutive empty months | At most 3, including window edges |
| Distinct publication dates | 60 |
| Articles per ticker | 100 |

Eligible tickers rank by covered months, then distinct publication dates, then ticker alphabetically.

The final screen requires **at least 10 numeric sentiment ratings in at least 80% of all configured publication months**: at least **58 of 72 months** by default. Negative, neutral, and positive count; mixed and insufficient information do not. The first 30 candidates passing this screen form the final universe. These rules measure chart coverage, not model accuracy or market representativeness.

Articles are deduplicated per ticker using URL, or publication timestamp and title when the URL is missing. Body text is excluded; articles associated with multiple tickers remain eligible.

Publication timestamps and dates are retained. Each article links to the first valid price session **strictly after** its UTC publication date, within seven calendar days. Trading dates come from observed price histories.

## Outputs

| File | Contents |
| --- | --- |
| `data/processed/daily_prices.parquet` | Daily OHLC prices, adjusted close, volume, return percentage, volume ratio, and coverage fields |
| `data/processed/news_events.parquet` | Article–ticker records with publication dates, next trading date, headline, URL, publisher, and sentiment |
| `reports/quality_report.json` | Preparation counts, validation results, and final ticker selection |
| `reports/sentiment_report.json` | Model settings, label counts, candidate sentiment coverage, and processing statistics |

Dates and numeric columns retain native types. Publication timestamps are UTC. Returns use adjusted close; volume ratios compare today's volume with the mean of up to 20 preceding sessions.

See the [data preparation reference](./docs/data_preparation_reference.md) for report-count definitions, methodology, field types, and limitations.

## Sentiment

[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) evaluates each headline for its associated ticker with thinking disabled.

Each input includes the target's name and security type. Reports and Parquet metadata record the inference settings and target identities.

Five fixed examples guide classification. An unnamed, uniformly bullish or bearish stock/ETF recommendation can apply through the article–ticker association when its security type matches the target. Other-company news does not transfer to the target; fund flows do not transfer to holdings. Unresolved comparisons remain unknown. Recommendation membership is assumed, not verified from article bodies.

| Label | Score |
| --- | --- |
| `negative` | -1 |
| `neutral` | 0 |
| `positive` | +1 |
| `mixed` | null |
| `insufficient_information` | null |

Scores are category codes; confidence values are not exported. Mixed means opposing material facts about the same target. Insufficient information means no applicable assertion under the attribution rules.

Progress shows headline scoring and candidate acceptance.

## Code layout

| Module | Responsibility |
| --- | --- |
| [config.py](./src/stock_market_sentiment/config.py) | Settings validation and shared command arguments |
| [prepare.py](./src/stock_market_sentiment/prepare.py) | Raw data cleaning, session matching and candidate preparation |
| [sentiment.py](./src/stock_market_sentiment/sentiment.py) | Prompt, model inference and score mapping |
| [pipeline.py](./src/stock_market_sentiment/pipeline.py) | Score reuse, final coverage selection and output publication |

## Power BI

Use date hierarchies for year → quarter → month → day drilldown. Show article counts and insufficient-information counts alongside average sentiment. Exclude null scores from averages; missing news is not neutral.

The tickers form a coverage-selected sample. Sentiment and price movements show associations, not causation, and model labels can be incorrect.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary fixtures and model doubles without GPU inference or model downloads. See [source code](./src/stock_market_sentiment/) and [unit tests](./tests/unit/) for implementation details.

The [prompt evaluation](./docs/sentiment_prompt_evaluation.md) records the bounded GPU comparisons and their limitations.
