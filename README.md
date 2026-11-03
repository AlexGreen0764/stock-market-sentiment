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
| `Final Project_SP.pdf` | Supplied course project brief |

The large raw files stay outside Git. Keep the local news export if the team may need article bodies later. URLs alone do not guarantee continued access to original articles. Source provenance and licensing still need confirmation from the data provider.

## Installation

Requires **Python 3.11+**; validation has used Python 3.13. Dependencies are in [pyproject.toml](./pyproject.toml).

- Preparation uses PyArrow and does not need a GPU.
- Scoring requires an NVIDIA GPU supporting CUDA and BF16, a compatible driver, and CUDA-enabled PyTorch. There is no CPU scoring mode.
- The 4B model has approximately 8 GB of BF16 weights, plus runtime and batch memory. The fixed batch size is 4.
- Dependencies and the pinned tokenizer/weights need internet access on first download. Inference runs locally.
- Preparation retains candidate headline metadata in RAM. It does not materialize bodies or maintain global URL association sets. Scoring streams bounded Parquet chunks.

PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu134
.\.venv\Scripts\python.exe -m pip install -e .
```

The PyTorch wheel includes its CUDA runtime; a separate CUDA Toolkit is not required. Transformers 5.22.1 or newer (below 6) is required for the validated Qwen3.5 text loader. The official checkpoint download is roughly 9 GB and is cached for subsequent runs. Transformers may warn that reference PyTorch convolution/linear-attention kernels are used; these support correct inference but can affect throughput.

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

The raw news file is read once with projected columns, respecting quoted multiline fields while skipping body/summary conversion. Empty headlines and invalid/out-of-window publication timestamps are rejected. Missing publishers and URLs are flagged and retained. Duplicate URLs are removed within each ticker; rows without URLs use publication timestamp plus title. Broad articles remain eligible.

Each article is matched to the first observed valid price session **strictly after** its UTC publication date, at most seven calendar days later. Articles before the first price date or without a usable later session are rejected. This uses observed price histories, not a formal exchange calendar. Publication timestamps may be placeholders, so same-day reactions are deliberately excluded.

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

The compact prompt asks for sentiment toward the target ticker using only the supplied headline. Generic roundups, an unidentified target, and insufficient target-specific evidence should produce an abstention. The model receives no bodies or future prices. These instructions do not guarantee correct classifications.

Routine target-company announcements (meeting dates, earnings schedules, filings, conference appearances, and previews without results) are neutral. Insufficient information means the target cannot be identified or the headline omits its specific situation. Mixed means material opposing implications without a clearly dominant direction; it is distinct from both neutral and missing context.

Fixed settings in [sentiment.py](./src/stock_market_sentiment/sentiment.py): CUDA/BF16, SDPA, batches of 4, 8,192-row chunks, and a 1,024-token prompt budget. Overlength prompts fail explicitly rather than silently truncating text. Weights use roughly 8 GB before activation/runtime overhead.

The scorer loads the text-only Qwen3.5 configuration and model, excluding the vision encoder. Thinking is explicitly disabled in the chat template with `enable_thinking=False`. The model selects the highest next-token logit among five permitted answer letters. It performs one forward pass per batch, with no free-form generation, reasoning output, or KV cache. Left padding and token-length grouping reduce padding work; outputs are restored to input order. Identical **ticker/headline pairs** are scored once per chunk; the same headline for different tickers is scored separately.

| Label | `sentiment_score` |
| --- | --- |
| `negative` | -1 |
| `neutral` | 0 |
| `positive` | +1 |
| `mixed` | null |
| `insufficient_information` | null |

**This score is an ordinal category code, not the old FinSense positive-minus-negative probability score.** No confidence or class probabilities are exported. Reruns remove the former FinSense probability columns. Mixed evidence and insufficient information remain separate labels, and neither becomes neutral.

The sentiment report includes model/revision, the exact compact prompt, answer method, `thinking_enabled=false`, `model_input_mode=text_only`, input columns, token/batch limits, counts for all five labels, inference counts, throughput, and elapsed time. The prompt and thinking setting are also stored in Parquet metadata. The mean score excludes null scores; `numeric_score_rows` records its denominator and includes neutral rows. Report the abstention share alongside numeric sentiment summaries.

Scoring stages a partial file and replaces the original only after all rows score successfully. Inference failure preserves the original. Dates and non-sentiment columns are preserved. Data and reports are replaced individually, not as a multi-file transaction. Preparation recreates unscored news; score after your final preparation run.

## Power BI interpretation

Use year → quarter → month → day date hierarchies on the daily/article-level data. Show article counts and abstention counts alongside sentiment. Missing news is not neutral sentiment. Aggregate article scores directly, or weight preaggregated means by their numeric-score counts.

A coverage-selected ticker universe and source collection patterns affect comparisons. Headline sentiment and subsequent price/volume movements show associations only. An SLM replacement is not established as more accurate until evaluated against trusted labels. Fixture tests establish implementation behavior, not financial sentiment accuracy.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests verify thinking is disabled, scoring order is restored, and cache-free single-token classification is used. They use temporary fixtures and model doubles; they do not process the full dataset, download model weights, or require a GPU. They cover price validation, full-window coverage and edge gaps, deterministic selection, retention of broad articles, matching output universes, date links, null abstention scores, ticker-specific caching, old-score replacement, and preservation of originals after failures.

Generated reports are standard JSON and should not be manually edited. The reports directory is excluded from Git.
