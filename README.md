# Stock market data preparation

Python preparation and local SLM sentiment scoring for a Power BI school visualization project: stock prices, trading volume, and news sentiment over time. Sentiment is a model estimate from a headline, not evidence that news caused a price move or a prediction of future returns.

## Repository layout

| Path | Purpose |
| --- | --- |
| [prepare.py](./src/stock_market_sentiment/prepare.py) | Headline loading, price validation, coverage measurement, ticker selection, and export |
| [config.py](./src/stock_market_sentiment/config.py) | Preparation settings and validation |
| [sentiment.py](./src/stock_market_sentiment/sentiment.py) | Separate CUDA SLM scoring stage |
| [config/default.json](./config/default.json) | Date window, output location, ticker count, and coverage thresholds |
| [tests/unit/](./tests/unit/) | Small preparation fixtures and mocked sentiment checks |
| `data/raw/full_history/` | Local per-ticker OHLCV CSV inputs |
| `data/raw/nasdaq_exteral_data.csv` | Local news export; original filename preserved |
| `data/processed/` | Two generated Parquet visualization tables |
| `reports/quality_report.json` | Preparation audit |
| `reports/sentiment_report.json` | Sentiment audit |

Raw data, processed data, and reports stay outside Git. Keep the raw news export if article bodies may be needed later.

## Installation

Tested with **Python 3.13**, **PyTorch 2.15.0+cu134**, and **Transformers 5.22.1**. Dependency constraints are defined in [pyproject.toml](./pyproject.toml).

- Preparation uses PyArrow and does not need a GPU.
- Scoring runs on an NVIDIA GPU with CUDA-enabled PyTorch using BF16. Model weights occupy roughly 8 GB, with additional memory needed during inference.
- Initial package and model downloads need internet access. Inference runs locally.

PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu134
.\.venv\Scripts\python.exe -m pip install -e .
```

The model download is roughly 9 GB and is cached for subsequent runs.

Optional Flash Linear Attention acceleration on Windows uses the following tested package versions:

```powershell
.\.venv\Scripts\python.exe -m pip install "flash-linear-attention==0.6.0" "fla-core==0.6.0" "triton-windows==3.8.0.post29"
```

FLA accelerates the model's linear-attention operations.

## 1. Prepare the study dataset

```powershell
.\.venv\Scripts\python.exe -m stock_market_sentiment
```

Defaults:

| Setting | Value |
| --- | --- |
| Publication window | **2015-01-01 through 2020-12-31**, inclusive |
| `max_tickers` | **30**, or fewer if fewer qualify |
| `output` | `data/processed` |
| Article ticker limit | **None** |
| Body text | **Not retained or exported** |

Preparation evaluates every ticker with a price CSV. News symbols and filename stems are trimmed and uppercased; original paths are preserved, including mixed-case filenames. Normalized filename collisions are rejected. Historical aliases and multiple ticker symbols in one field are not resolved.

Empty headlines and invalid/out-of-window publication timestamps are rejected. Missing publishers and URLs are flagged and retained. Duplicate URLs are removed within each ticker; rows without URLs use publication timestamp plus title. Articles associated with multiple tickers remain eligible.

Each article is matched to the first observed valid price session **strictly after** its UTC publication date, at most seven calendar days later. Articles before the first price date or without a usable later session are rejected. Trading dates come from observed price histories. Publication dates and timestamps are retained separately.

### Coverage and selection

Coverage is measured after price matching and deduplication. A covered month contains at least one eligible article. The denominator is **all configured months**: 72 for 2015–2020. Leading and trailing gaps count toward the empty-month limit.

Defaults require:

- an observed first-to-last article span of at least **24 calendar months**;
- articles in at least **75% of configured months**;
- no gap longer than **3 consecutive empty months**, including window edges;
- at least **60 distinct UTC publication dates**;
- at least **100 deduplicated articles**.

Qualifiers rank by **covered months descending**, then **distinct publication dates descending**, then **ticker alphabetically**. The first 30 form a fixed study universe. Selection does not use sentiment labels, sector balancing, or hardcoded ticker lists. This is a coverage-selected sample, not a representative sample of the market.

Both outputs contain only selected tickers. Prices are exported daily within the window, plus any subsequent sessions needed for retained year-end articles. Full histories are validated one at a time; only selected histories are reread for export. If no ticker qualifies, both files are empty with their typed schemas.

CLI overrides: `--root`, `--config`, `--output`, `--start-date`, `--end-date`. Both date bounds are required in configuration.

```powershell
.\.venv\Scripts\python.exe -m stock_market_sentiment --start-date 2016-01-01 --end-date 2020-12-31
```

### Outputs and calculations

| File | Grain | Main fields |
| --- | --- | --- |
| `daily_prices.parquet` | Selected ticker × valid trading date | OHLC, adjusted close, volume, return %, volume relative to prior sessions, news coverage flag/bounds |
| `news_events.parquet` | Deduplicated article × selected ticker | UTC publication timestamp/date, next trading date, headline, URL, publisher, completeness flags, then sentiment fields |

Dates use Arrow `date32`; publication timestamps use UTC `timestamp[ms]`. Numeric fields retain numeric types for Power BI. Headlines preserve their original case. No article bodies, summaries, or ticker-association counts are exported.

Prices must be finite and positive, have consistent OHLC bounds, and have nonnegative integral int64 volumes. Integral decimal/exponent volume strings are accepted exactly; fractional volumes are rejected. Duplicate price dates keep the first valid row.

Returns are `100 × (adjusted close / previous adjusted close − 1)`. Volume ratios divide today's volume by the mean of up to 20 preceding sessions, excluding today. Both metrics use history before the export window; the first available history row has null metrics.

The quality report records source/filter counts, price rejection reasons and capped examples, all candidate coverage metrics, selection ranking, qualifying-but-unselected tickers, final monthly/yearly article counts, and publisher/URL completeness. `price_rows_valid` describes cleaned full-history candidates; `price_rows_kept` describes selected exports. Counts are article–ticker records, not globally unique articles.

## 2. Score headline sentiment

After preparation:

```powershell
.\.venv\Scripts\python.exe -m stock_market_sentiment.sentiment
```

The scorer uses [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), pinned with its tokenizer to commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It reads `data/processed/news_events.parquet` and writes `reports/sentiment_report.json`. It does not read preparation configuration; a custom preparation output path is not automatically picked up.

The prompt asks for sentiment toward the target ticker using only the supplied headline. Generic roundups, an unidentified target, and insufficient target-specific evidence should produce an abstention.

Routine target-company announcements (meeting dates, earnings schedules, filings, conference appearances, and previews without results) are neutral. Insufficient information means the target cannot be identified or the headline omits its specific situation. Mixed means material opposing implications without a clearly dominant direction; it is distinct from both neutral and missing context.

Fixed settings in [sentiment.py](./src/stock_market_sentiment/sentiment.py): CUDA/BF16, SDPA, batches of 4, 8,192-row chunks, and a 1,024-token prompt budget. Prompts exceeding the budget produce an error.

The scorer uses Qwen3.5's text-only model with thinking disabled. It selects one of five answer tokens in a single forward pass. Identical **ticker/headline pairs** are scored once per chunk; the same headline for different tickers is scored separately.

| Label | `sentiment_score` |
| --- | --- |
| `negative` | -1 |
| `neutral` | 0 |
| `positive` | +1 |
| `mixed` | null |
| `insufficient_information` | null |

**The score is an ordinal category code.** No confidence or class probabilities are exported. Mixed evidence and insufficient information remain separate labels, each with a null score.

The sentiment report records model settings, the prompt, label counts, inference counts, throughput, and elapsed time. Model and prompt details are also stored in Parquet metadata. The mean score excludes null scores; `numeric_score_rows` records its denominator and includes neutral rows.

Progress prints after each 8,192-row chunk, showing records scored, average records per second, and estimated minutes remaining. Model loading and initial GPU kernel compilation occur before the first scoring update.

Scoring replaces the news file after all rows score successfully, preserving dates and non-sentiment columns. An inference failure leaves the original file intact. Preparation overwrites sentiment results, so run scoring after preparation.

## Power BI interpretation

Use year → quarter → month → day date hierarchies on the daily/article-level data. Show article counts and abstention counts alongside sentiment. Missing news is not neutral sentiment. Aggregate article scores directly, or weight preaggregated means by their numeric-score counts.

A coverage-selected ticker universe and source collection patterns affect comparisons. Headline sentiment and subsequent price/volume movements show associations only. Sentiment accuracy requires evaluation against trusted labels. Fixture tests establish implementation behavior.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary fixtures and model doubles and require no GPU or model download. They cover price validation, full-window coverage and edge gaps, deterministic ticker selection, matching output universes, date links, disabled thinking, scoring order, single-token classification, null abstention scores, ticker-specific caching, sentiment-column replacement, and preservation of originals after failures.

Generated reports are standard JSON and should not be manually edited. The reports directory is excluded from Git.
