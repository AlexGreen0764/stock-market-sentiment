# Data preparation reference

Reference notes for interpreting the project outputs and preparing a Phase 2 explanation. The sections below contain definitions, observed figures, implementation details and information still to be supplied by the team.

**Snapshot:** November 4, 2026. Values are taken from the saved reports available on that date. A later preparation or scoring run can change them. This date is not a dataset download date.

## 1. Evidence locations and source information

| Evidence | Location | What it establishes |
| --- | --- | --- |
| Preparation report | [quality_report.json](../reports/quality_report.json) | Input/output counts, validation results, coverage, selection and date-linking rules |
| Sentiment report | [sentiment_report.json](../reports/sentiment_report.json) | Model provenance, prompt, examples, target identities, label counts and processing statistics |
| Preparation implementation | [prepare.py](../src/stock_market_sentiment/prepare.py) | Exact parsing, validation, deduplication, coverage, calculations and export behavior |
| Preparation configuration | [default.json](../config/default.json) | Default period, maximum ticker count and coverage thresholds |
| Sentiment implementation | [sentiment.py](../src/stock_market_sentiment/sentiment.py) | How model inputs are assembled and predictions become labels and scores |
| Security identities | [securities.json](../config/securities.json) | Target names and stock/ETF types supplied to the model |

The report values describe the completed run. Configuration and code explain the implementation; defaults alone do not prove which settings an earlier run used.

### Inputs and provenance gaps

| Input | Confirmed local facts | Information the team must supply |
| --- | --- | --- |
| Prices | [data/raw/full_history/](../data/raw/full_history/); one CSV per ticker; ticker inferred from filename; daily date, OHLC, adjusted close and volume fields | Original dataset title, publisher/provider, source URL, actual download/export method and acquisition date if known |
| News | [nasdaq_exteral_data.csv](../data/raw/nasdaq_exteral_data.csv); article metadata with publication timestamp, title, ticker, URL and publisher fields | Original dataset title, publisher/provider, source URL, actual download/export method and acquisition date if known |
| Sentiment model | Model identifier `Qwen/Qwen3.5-4B`; exact revision recorded in the sentiment report | Cite the model separately from the datasets if describing generated sentiment |

The preparation code reads CSV files already present locally. It does not download the original price or news datasets. A filename containing “nasdaq” is insufficient evidence of the dataset's original provider or collection method. The code also does not establish upstream price adjustment methodology, currency or licensing; confirm those from the original source documentation where relevant.

The news reader projects five source columns: `Date`, `Article_title`, `Stock_symbol`, `Url` and `Publisher`. Body text, summaries, author information and unrelated columns are not loaded into preparation outputs. Sentiment is generated locally from headlines; it is not an existing source label being cleaned.

## 2. Units of observation and count scopes

| Term | Meaning |
| --- | --- |
| Price row | One observed daily price record for one ticker. The cleaned price table has one row per ticker/date. |
| Article–ticker record | One news record associated with one ticker. The same underlying article can contribute records for several tickers. |
| Candidate ticker | A normalized ticker identified by a price CSV filename. This defines which news tickers can enter preparation. |
| Qualified ticker | A candidate that passes every news-coverage threshold after date filtering, usable price matching and deduplication. |
| Retained ticker | A qualified ticker selected by the ranking and maximum-ticker limit. |
| Model inference | One classification of a ticker/headline pair within a processing chunk. Repeated pairs in that chunk reuse the result. |

**Article–ticker counts are not counts of globally unique articles.** Likewise, unique ticker/headline pairs are not necessarily unique articles: different URLs can share a title.

There are three particularly important scopes:

1. **Full candidate price histories:** every row read from all candidate price files, including dates outside the study period.
2. **Candidate news pool:** rows matching any candidate ticker, narrowed by the date and validation checks at each processing stage.
3. **Final selected dataset:** only records belonging to the retained tickers.

Counters from different scopes should not be treated as if they described the same population.

## 3. Snapshot of the completed run

| Measure | Value | Scope or interpretation |
| --- | ---: | --- |
| Configured publication window | 2015-01-01 to 2020-12-31 | Inclusive; 72 calendar months |
| Candidate tickers | 7,693 | All ticker files considered |
| Qualified tickers | 1,493 | Passed all coverage requirements |
| Rejected tickers | 6,200 | Failed at least one coverage requirement |
| Qualified but not selected | 1,463 | Passed coverage but fell outside the first 30 ranked tickers |
| Retained tickers | 30 | Shared universe for prices and news |
| Price rows read | 29,677,722 | Full histories of all candidate tickers |
| Price rows exported | 45,352 | Retained tickers and permitted export dates |
| News rows read | 15,549,299 | Entire input news CSV |
| News rows before coverage selection | 2,636,759 | Candidate pool after date filtering, usable price matching and deduplication |
| News records exported | 142,266 | Final article–ticker records |
| News records with numeric sentiment | 39,352 | Negative, neutral or positive; mixed/unknown have null scores |

All 30 retained tickers have at least one retained news record in each of the 72 study months. This is stronger than the minimum threshold, but does not imply uniform article volume or usable sentiment in every month.

The retained symbols include stocks and ETFs; “30 tickers” should not be described as “30 companies.” The complete list is in `retained_tickers`.

## 4. Global preparation counters

### 4.1 Price counters: `counts.price_*`

These counters cover the **initial validation pass over full histories of all 7,693 candidates**, except `price_rows_kept`, which describes the final export. Selected histories are read again for export using separate scratch counters, so that reread does not double the reported input totals.

| Counter | Snapshot | Definition |
| --- | ---: | --- |
| `price_rows_read` | 29,677,722 | Every price CSV data row encountered during the initial validation pass. |
| `price_rows_valid` | 28,583,868 | Accepted, unique-date rows after validation and duplicate-date removal; includes dates outside the export window and tickers not ultimately selected. |
| `price_rows_invalid` | 1,093,854 | Rows rejected by the validation checks. Each rejected row receives one reason. |
| `price_rows_duplicate_date` | 0 | Otherwise-valid rows whose date already has an accepted row for that ticker. The first valid occurrence is retained. |
| `price_rows_invalid_parse` | 257 | A required date or numeric value could not be parsed, including blank or malformed input values. |
| `price_rows_invalid_nonfinite` | 0 | A parsed OHLC or adjusted-close value is not finite, for example NaN or infinity. |
| `price_rows_invalid_nonpositive_price` | 1,090,104 | At least one OHLC or adjusted-close value is zero or negative. Zero trading volume is allowed; zero price is rejected. |
| `price_rows_invalid_volume` | 0 | Parsed volume is nonfinite, negative, fractional or outside the signed 64-bit integer range. A volume that cannot be parsed is counted under `invalid_parse`. |
| `price_rows_invalid_ohlc_bounds` | 3,493 | Daily low/high do not contain the open and close, or low is above high. Adjusted close is not compared with unadjusted OHLC bounds. |
| `price_rows_kept` | 45,352 | Rows written to the price output for the final 30 tickers: dates in the configured window plus any dates needed by retained article links. |

Validation order is parsing, finite price checks, positive price checks, volume checks, OHLC bounds, then duplicate-date detection. When multiple problems occur in a row, the first applicable validation reason determines its count. An invalid row does not also increment the duplicate-date count.

Reconciliation:

```text
29,677,722 read = 28,583,868 valid + 1,093,854 invalid + 0 duplicate-date
1,093,854 invalid = 257 parse + 0 nonfinite + 1,090,104 nonpositive
                    + 0 volume + 3,493 OHLC bounds
```

The difference between valid rows and exported rows mostly reflects study scope and ticker selection. It is not an additional invalid-data total. These global figures do not establish the invalid-row rate within just the final 30 tickers; use their individual validation records for that question.

### 4.2 News counters: `counts.article_*`

| Counter | Snapshot | Definition and point of measurement |
| --- | ---: | --- |
| `article_rows_read` | 15,549,299 | Every row encountered in the news input CSV. |
| `article_rows_selected` | 4,906,497 | Rows whose stripped, uppercased ticker matches any candidate price ticker. Counted before date parsing and window filtering. “Selected” here does not mean the final 30 tickers. |
| `article_rows_unmatched_ticker` | 10,642,802 | Rows whose normalized ticker does not match a candidate price file. These rows do not proceed to date or title checks. |
| `article_rows_invalid_date` | 0 | Candidate-matching rows whose publication string fails the expected UTC timestamp format. |
| `article_rows_outside_date_range` | 2,237,136 | Candidate-matching rows with a parsed UTC publication date outside the inclusive study window. |
| `article_rows_missing_title` | 0 | Candidate-matching, in-window rows whose title is empty after stripping surrounding whitespace. |
| `article_rows_missing_url` | 0 | Missing URLs among candidate-matching, in-window, nonempty-title rows, before price matching and deduplication. This is a metadata flag, not a rejection count. |
| `article_rows_missing_publisher` | 847,604 | Missing publishers at the same stage as the preceding URL counter. These rows remain eligible. |
| `article_rows_unusable_price_match` | 24,534 | In-window, nonempty-title rows with no usable subsequent price session: publication precedes usable price history, history is exhausted, no valid history exists, or the next observed session is more than seven calendar days later. The report does not separate those reasons. |
| `article_rows_duplicate` | 8,068 | Usably price-matched rows removed as duplicates within their ticker, using stripped URL or, if absent, publication timestamp and title. |
| `article_rows_before_coverage` | 2,636,759 | All surviving article–ticker records after the preceding filters, before coverage qualification, ranking and the final ticker cap. Includes records for candidates that later fail coverage. |
| `article_rows_kept` | 142,266 | Final article–ticker records exported for retained tickers. |
| `article_rows_kept_missing_publisher` | 89,852 | Final exported rows with an empty publisher; a subset of `article_rows_kept`. |
| `article_rows_kept_missing_url` | 0 | Final exported rows with an empty URL; a subset of `article_rows_kept`. |

The news missing-value checks recognize empty strings after stripping whitespace. They do not establish that every literal placeholder such as `N/A` has been standardized. A zero counter means no cases were found **at the stage where that check runs**, not that the issue was absent from every raw input row.

### 4.3 News count reconciliation

```text
15,549,299 read
  = 10,642,802 unmatched ticker + 4,906,497 candidate-matching

4,906,497 candidate-matching
  - 0 invalid dates
  - 2,237,136 outside the publication window
  - 0 empty titles
  = 2,669,361 ready for price matching

2,669,361
  - 24,534 unusable price matches
  - 8,068 duplicates
  = 2,636,759 before coverage selection

2,636,759 before coverage selection
  - 1,127,551 belonging to tickers failing coverage
  = 1,509,208 belonging to qualified tickers

1,509,208 belonging to qualified tickers
  - 1,366,942 belonging to qualified tickers outside the ranking cap
  = 142,266 final records
```

The qualification-stage article subtotals above are **derived** by summing `coverage_by_ticker[ticker].articles` over the ticker lists. They are not separate named counters in the JSON. Together, failed coverage and the ranking cap account for 2,494,493 records excluded after `article_rows_before_coverage`.

Missing-publisher and missing-URL flags are overlapping descriptions of records. Do not subtract them in this reconciliation or add them to the rejection totals.

## 5. Other preparation report fields

### 5.1 Ticker lists, selection settings and study dates

| Field | Meaning |
| --- | --- |
| `candidate_tickers` | All normalized ticker identifiers available from input price filenames. |
| `price_tickers` | Final selected ticker list used for the price output. Despite the name, this is not the full input-price universe. |
| `retained_tickers` | Final selected ticker list used for the news output; same universe as `price_tickers`. |
| `rejected_tickers` | Candidates failing at least one coverage criterion. Passing candidates outside the cap are not in this list. |
| `qualified_tickers` | All passing candidates, listed alphabetically. |
| `qualified_but_not_selected` | Passing candidates excluded by the final ticker cap. |
| `ticker_ranking` | Passing candidates in actual selection order. |
| `selection.max_tickers` | Maximum selected ticker count: 30 in this run. |
| `selection.ranking_rule` | Covered months descending, then distinct publication dates descending, then ticker alphabetically. Raw article volume and sentiment do not determine this ranking. |
| `selection.article_identity` | Per-ticker duplicate key: exact stripped URL, otherwise publication timestamp and stripped title. |
| `selection.coverage_window` | Coverage is measured across every configured month, including leading and trailing gaps. |
| `start_date`, `end_date` | Inclusive publication window and base price export window. |
| `news_link_rule` | The saved description of subsequent-session matching and permitted extension of linked price dates beyond the study end. |

Ticker-list identities:

```text
7,693 candidates = 6,200 rejected + 1,493 qualified
1,493 qualified = 1,463 qualified-but-not-selected + 30 retained
```

### 5.2 Coverage criteria and rationale

Coverage is evaluated separately for each candidate after date filtering, usable price matching and per-ticker deduplication. A ticker must pass all five tests.

| `coverage_thresholds` field | Run setting | Operational meaning | Reason for the criterion |
| --- | ---: | --- | --- |
| `span_months` | 24 | Inclusive number of calendar months from first to last retained publication month must be at least 24. Empty months between them still count toward span. | Avoid a short burst being treated as a long history. |
| `covered_month_fraction` | 0.75 | At least 75% of all configured months contain one or more usable article–ticker records. Here that means at least 54 of 72 months. | Require temporal breadth. |
| `max_empty_months` | 3 | No run of more than three consecutive empty months, including the beginning and end of the window. | Limit long coverage gaps. |
| `distinct_publication_dates` | 60 | At least 60 distinct UTC publication dates; these are calendar dates, not necessarily trading sessions. | Reduce dependence on a small number of heavily reported days. |
| `articles` | 100 | At least 100 surviving article–ticker records over the whole window. | Require a minimum total volume of observations. |

These are project design choices, not statistically established guarantees of representativeness or model accuracy. There is **no minimum article count per covered month**: one article makes a month “covered.” All passing candidates are ranked, and at most 30 are retained for a manageable visualization dataset.

### 5.3 `coverage_by_ticker`

| Nested field | Meaning |
| --- | --- |
| `coverage_start`, `coverage_end` | Earliest/latest surviving UTC publication dates for that candidate, before the final ticker cap; null when no articles survive. |
| `span_months` | Inclusive first-to-last publication-month span; zero with no surviving articles. |
| `window_months` | Number of calendar months touched by the configured window; 72 in this run. |
| `covered_months` | Months containing at least one surviving record. |
| `covered_month_fraction` | `covered_months / window_months`. |
| `max_empty_months` | Longest consecutive sequence of empty configured months, including both window edges. |
| `distinct_publication_dates` | Number of distinct UTC publication dates among surviving records. |
| `articles` | Number of surviving article–ticker records for that candidate. |
| `articles_by_year` | The same records grouped by UTC publication year. |
| `failed_criteria` | Names of coverage tests not satisfied. An empty list means all passed. |
| `passes` | Whether every coverage criterion passed; this alone does not mean the ticker was finally selected. |

### 5.4 Per-ticker filtering, price validation and date ranges

`article_filter_counts_by_ticker` supplies the following counters for candidate tickers. A counter with no increments can be absent from that ticker's object; treat absent counters in this section as zero.

| Nested counter | Meaning |
| --- | --- |
| `matched_articles` | Raw news rows matching that ticker, before date parsing and window filtering. |
| `invalid_date` | Matching rows with an unparseable publication timestamp. |
| `outside_date_range` | Matching rows with a valid date outside the configured window. |
| `in_date_range` | Matching rows with a valid date inside the window, counted before the empty-title check. This does not yet mean the row was exported. |

For example, ABBV has 6,451 matching rows: 2,865 outside the window and 3,586 inside it. Its 3,585 coverage-stage articles are measured later, after price matching and deduplication. The report does not provide per-ticker counts for every intervening rejection reason.

`price_validation_by_ticker` contains the same read, valid, invalid, duplicate-date and invalid-reason counters defined in section 4.1, scoped to one ticker's full input history. Sum these objects to obtain the corresponding global price counters, excluding `price_rows_kept`.

| `price_date_ranges` field | Meaning |
| --- | --- |
| `first`, `last` | Earliest/latest valid, deduplicated dates in the ticker's full cleaned history; not restricted to the study window. Null when no rows survive. |
| `input_rows` | Number of valid, deduplicated rows in that history. Despite its name, this is not the raw CSV row count; use `price_validation_by_ticker[ticker].price_rows_read` for raw rows. |
| `exported_rows` | Price rows actually exported for that ticker. Zero for candidates not selected. |

`invalid_price_examples` groups illustrative rejected rows by validation reason. Each entry records `ticker`, `source_file` and the original `row` values. Examples are capped at three per reason across the candidate files, taken from processing order. They are not a random sample and need not belong to retained tickers.

### 5.5 Final news quality and temporal summaries

| Field | Meaning |
| --- | --- |
| `retained_news_quality.article_ticker_records` | Final record total; same as `counts.article_rows_kept`. |
| `retained_news_quality.missing_publisher` | Final empty-publisher count; same as `counts.article_rows_kept_missing_publisher`. |
| `retained_news_quality.missing_publisher_fraction` | Final empty-publisher count divided by final record count; null for an empty output. |
| `retained_news_quality.missing_url` | Final empty-URL count; same as `counts.article_rows_kept_missing_url`. |
| `retained_news_by_year[year].article_ticker_records` | Final records by UTC publication year. |
| `retained_news_by_year[year].tickers_with_articles` | Retained tickers with at least one record in that year. This is 30 in each year in this run. |
| `retained_news_by_month[YYYY-MM]` | Final records by UTC publication month, across all retained tickers. This is a volume total, not the number of covered tickers. |

Snapshot annual totals are 18,334 (2015), 22,848 (2016), 25,969 (2017), 28,053 (2018), 24,940 (2019) and 22,122 (2020), summing to 142,266.

## 6. Preparation methodology and decision notes

### Processing order

1. Identify price files and normalize their filename-derived tickers.
2. Stream selected news columns; normalize ticker/text, parse UTC timestamps, match candidate symbols and restrict publication dates.
3. Validate each candidate's full price history, deduplicate dates, sort sessions and calculate price features.
4. For each candidate, find a usable subsequent price session for each article, then deduplicate its articles.
5. Measure news coverage, identify qualifiers, rank them and retain at most 30.
6. Export selected news and prices with explicit types; write the preparation report.
7. Separately score headlines for their target securities and enrich the news file; write the sentiment report.

### Transformations and why they are used

| Operation actually implemented | Method | Purpose and boundary |
| --- | --- | --- |
| Ticker normalization | Strip surrounding whitespace and uppercase the symbol; apply the same rule to price filenames and news tickers. | Make string identifiers comparable. This does not resolve corporate identity changes or historical symbol reuse. |
| Text normalization | Strip surrounding whitespace from titles, URLs and publishers. Headline case and wording are otherwise retained. | Remove superficial whitespace inconsistencies. Publisher aliases, spelling variants and URL variants are not comprehensively standardized. |
| Timestamp parsing | Parse the literal format `YYYY-MM-DD HH:MM:SS UTC`; retain a UTC timestamp and its UTC calendar date. | Support time grouping while retaining the source time field. Parsing does not validate the real precision of publication times. |
| Publication-window restriction | Keep dates from 2015-01-01 through 2020-12-31 inclusive. | Bound the study period and keep graph comparisons within a consistent window. This alone does not prove uniformly high source quality. |
| Price validation | Reject unparsable, nonfinite, nonpositive-price, invalid-volume and inconsistent-OHLC rows; keep first valid row per date. | Prevent malformed records from entering metrics and session matching. Invalid price values are removed rather than imputed. |
| News duplicate removal | Within one ticker, keep the first usable record for a stripped URL; when URL is empty, use publication timestamp plus title. | Limit repeated source records. Article-ticker associations for different tickers remain distinct. Headline-only duplicates with different URLs can remain. |
| Coverage and ranking | Apply all five coverage criteria, then rank qualifiers and cap the result at 30. | Favor continuity and reduce dataset size. The sample is selected for coverage, not representative market sampling. |
| Optional metadata handling | Retain empty publishers/URLs when other requirements pass; export presence flags. | Preserve usable headline observations while making missing metadata visible. No publisher is inferred or filled from another source. |
| Body exclusion | Do not load or export body/summary text. | Keep the preparation and model input focused on headlines and reduce storage/context requirements. The sentiment model cannot use omitted article context. |
| Typed Parquet output | Persist dates, timestamps, numbers and booleans using explicit Arrow schemas. | Preserve field types for Power BI and avoid repeatedly parsing strings on import. |

### Publication date versus linked session

The publication timestamp and date are preserved. A separate `next_trading_date` is computed from the ticker's **observed valid price sessions**:

- Find the first session strictly after the UTC publication date.
- Exclude an article dated before the first valid price-history date.
- Reject a match when no later session exists or the gap is more than seven calendar days.
- Same-day price matching is not used, regardless of publication hour.
- Retain a linked price date beyond the configured end date when required by an in-window article.

Illustrations: a normal Friday publication links to the next observed session, often Monday; if that Monday has no valid price row, it can link to Tuesday. The implementation uses observed history, **not an independently checked exchange holiday calendar**. Missing price records can also cause a date to be skipped. UTC dates are not converted to exchange-local dates, and market-open/close times are not modeled.

This link is an alignment choice for comparing news with subsequent daily prices. It is not proof of the timing or cause of a market reaction.

## 7. Output fields, types and calculated values

### 7.1 [daily_prices.parquet](../data/processed/daily_prices.parquet)

**Grain:** one ticker/date price observation. Expected key: `(ticker, date)`.

| Field | Stored type | Meaning |
| --- | --- | --- |
| `ticker` | string | Normalized security symbol. |
| `date` | date32 | Date of an observed valid daily price session. |
| `open` | float64 | Source daily opening price. |
| `high` | float64 | Source daily high price. |
| `low` | float64 | Source daily low price. |
| `close` | float64 | Source daily closing price. |
| `adj_close` | float64 | Provider-supplied adjusted close, renamed from source `adj close`. Adjustment factors are not recalculated by this project. |
| `volume` | int64 | Source nonnegative integral daily trading volume. |
| `return_pct` | float64 | Percentage change in adjusted close from the preceding valid session, rounded to four decimal places. |
| `volume_vs_prior_20d` | float64 | Current volume divided by mean volume over up to 20 preceding valid sessions, rounded to four decimal places. |
| `news_coverage_pass` | boolean | Ticker passed the coverage requirements. All exported price rows belong to passing, selected tickers, so this is true throughout this output. |
| `news_coverage_start` | date32 | First surviving publication date for the ticker; repeated on its price rows. |
| `news_coverage_end` | date32 | Last surviving publication date for the ticker; repeated on its price rows. |

Formulas:

```text
return_pct = 100 × (current adj_close / preceding valid-session adj_close − 1)

volume_vs_prior_20d = current volume / mean(up to 20 preceding valid-session volumes)
```

The current session is excluded from the volume reference mean. A ratio of 2 means twice the reference average, not a 2% increase. Fewer than 20 prior sessions are used when that is all the history available. With no prior volume history or a zero prior mean, the ratio is null. With no preceding valid adjusted close, return is null.

Features are calculated using full cleaned histories before the export window is applied. Thus the first exported price row can have a return and volume ratio based on earlier, unexported sessions. “Prior 20d” means observed valid sessions, not 20 calendar days; gaps in usable history can span longer periods.

### 7.2 [news_events.parquet](../data/processed/news_events.parquet)

**Grain:** one retained article–ticker record. There is no exported global article ID. URL alone, title alone and `(ticker, title)` are not guaranteed row keys.

| Field | Stored type | Meaning |
| --- | --- | --- |
| `ticker` | string | Normalized associated security symbol. |
| `published_at_utc` | timestamp[ms, UTC] | Parsed publication timestamp; source timestamp format contains seconds. Millisecond storage does not imply millisecond source precision. |
| `article_date_utc` | date32 | UTC calendar date extracted from publication timestamp. |
| `next_trading_date` | date32 | Subsequent observed session used for price alignment, subject to the matching rules above. |
| `title` | string | Stripped headline; model text input. |
| `url` | string | Stripped source URL; may be an empty string when missing. |
| `publisher` | string | Stripped publisher; may be an empty string when missing. |
| `has_url` | boolean | Whether the stripped URL is nonempty. |
| `has_publisher` | boolean | Whether the stripped publisher is nonempty. |
| `sentiment_score` | float32 | Generated category code: −1, 0 or +1; null for mixed or insufficient information. Added by sentiment scoring. |
| `sentiment_label` | string | Generated sentiment category. Added by sentiment scoring. |

The model's `target_name` and `security_type` are lookup inputs recorded in model provenance. They are not additional columns in this news table.

### Using the tables together

- Publication-date charts use `article_date_utc`; subsequent-session comparisons use `next_trading_date`.
- A news record links to prices using its ticker and `next_trading_date` against the price ticker and `date`.
- Several articles can link to one price session. A flattened join repeats that session's price/volume values; summing those repetitions would inflate volume totals.
- Average sentiment excludes null scores. Retain the counts of mixed, insufficient-information and numerically scored records alongside averages.
- Empty publisher strings and intentional null sentiment scores represent different issues and should be described separately.

## 8. Sentiment methodology and report definitions

### Classification method

The scorer loads the pinned text model `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Each input contains `target_ticker`, `target_name`, `security_type` and `headline`. Names and types come from the security lookup; every input ticker must have a nonempty identity entry before model loading proceeds.

The prompt asks for evidence about the target before assigning sentiment. Ten fixed-order user/assistant examples use the same JSON input format. Thinking is disabled. The implementation chooses the highest next-token model logit among the five answer letters A–E; it does not generate or export explanations or confidence values. The exact prompt, examples and identities used are recorded in the sentiment report and Parquet metadata.

| Answer | Saved label | Score | Intended meaning |
| --- | --- | ---: | --- |
| A | `negative` | −1 | Material adverse implications for the target. |
| B | `neutral` | 0 | A concrete target-specific fact with no directional implication. |
| C | `positive` | +1 | Material favorable implications for the target. |
| D | `mixed` | null | Material favorable and adverse implications for the same target, neither dominant. |
| E | `insufficient_information` | null | The headline does not assert enough target-specific information. |

Mixed and insufficient information remain separate labels even though both have null scores. A neutral zero means a classified nondirectional fact; a null does not mean neutral. The label definitions express the intended task, not a guarantee that each model prediction satisfies it.

### Sentiment report fields

| Field | Snapshot or interpretation |
| --- | --- |
| `model`, `model_revision` | Model identifier and exact pinned model/tokenizer revision. |
| `input_file` | Absolute path of the enriched news output. |
| `input_columns` | Source file columns used: `ticker` and `title`. |
| `model_input_fields` | JSON fields supplied to the model, including name/type from the lookup. |
| `target_securities` | Actual ticker-to-name/type mapping used for the scored file. |
| `system_prompt`, `few_shot_messages` | Exact instructions and example chat turns used. |
| `answer_method` | Highest next-token logit among A–E. |
| `thinking_enabled` | False. |
| `model_input_mode` | `text_only`; no article bodies or images are supplied. |
| `max_tokens` | 1,024 tokens for the complete model prompt, including instructions and examples. Oversized inputs raise an error rather than being silently truncated. |
| `batch_size` | Four ticker/headline inputs per model batch. |
| `read_batch_size` | Up to 8,192 news rows per input chunk; also the scope of duplicate-inference reuse. |
| `rows_scored` | 142,266 article–ticker records enriched; predictions are attached to every retained input row. |
| `headline_inferences` | 133,848 classifications after duplicate ticker/headline pairs are collapsed within each chunk. A pair repeated in another chunk can be classified again. This is not a global unique-article or unique-headline count. |
| `label_counts` | Numbers of enriched records assigned to each label; these sum to `rows_scored`, including rows that reused predictions. |
| `numeric_score_rows` | 39,352 records labeled negative, neutral or positive. |
| `mean_sentiment_score` | Approximately 0.3600; arithmetic mean of numeric category codes, excluding null scores. |
| `sentiment_score_definition` | Saved mapping between labels and numeric/null scores. |
| `records_per_second` | 8.6 records/second over processing after initial model loading; includes processing overhead, not just GPU execution. |
| `elapsed_seconds` | 16,639.08 seconds for the reported scoring run, including initial model loading. Timing fields are operational statistics, not quality metrics. |

Within a chunk, repeated `(ticker, title)` pairs share an inference while retaining their separate rows. The same headline attached to different tickers is classified separately. This reuse is distinct from preparation's URL-based duplicate removal: the stages use different keys and operate on different populations.

### Label totals and mean calculation

| Label | Records | Share of all records |
| --- | ---: | ---: |
| Negative | 6,334 | 4.45% |
| Neutral | 12,517 | 8.80% |
| Positive | 20,501 | 14.41% |
| Mixed | 4,228 | 2.97% |
| Insufficient information | 98,686 | 69.37% |
| **Total** | **142,266** | **100%** |

```text
Numeric score rows = 6,334 + 12,517 + 20,501 = 39,352
Null score rows = 4,228 + 98,686 = 102,914
142,266 total = 39,352 numeric + 102,914 null

Mean sentiment = (20,501 − 6,334) / 39,352 = 0.360007...
```

Numeric ratings cover about 27.7% of all retained records. The mean is a summary of assigned categories, not predicted return, percentage price change, probability or certainty. It is weighted by retained article–ticker records, so repeated titles or multiple stories can affect averages. It does not average all 142,266 records with nulls replaced by zero.

Preparation replaces the base news output; sentiment scoring subsequently enriches that file in place. The preparation report therefore describes the base preparation stage, while the sentiment report describes the later generated fields.

## 9. Findings and limitations relevant to interpretation

| Evidence or design choice | Implication |
| --- | --- |
| 89,852 of 142,266 retained records lack a publisher: 63.16%. | Publisher comparisons use an incomplete subset. All retained rows have URLs in this run, but URL presence does not prove that the linked page is still accessible. |
| 30 retained tickers are chosen from 1,493 qualifiers using coverage over the entire window. | Conclusions describe this coverage-selected sample. It is not a random market sample or a universe selected independently at each historical date. |
| The retained universe includes ETFs as well as stocks. | A fund headline and a constituent-company headline can require different target attribution. A headline about fund flows does not automatically describe each holding. |
| Coverage tests precede sentiment scoring. | News presence in a month does not guarantee enough non-null sentiment observations for a stable monthly average. |
| Headlines are associated with tickers by the source dataset; there is no body-level entity verification. | A valid article association can still provide no target-specific evidence in the headline. UNKNOWN is an intended outcome. |
| Sentiment is model-generated from the headline and target identity only. | Missing context, target attribution, neutral/unknown boundaries and MIXED judgments can cause errors. Label frequencies alone do not establish accuracy. |
| Prices are daily observations and links use UTC dates plus subsequent observed sessions. | The data does not measure precise intraday reactions or prove which trading session first incorporated the news. |
| Other events and market conditions are not controlled experimentally. | News, sentiment, volume and returns can be compared as associations; causation is not established. |
| Checks occur at specific processing stages. | Missing-value and rejection counts must be reported with their scope. The report is not an exhaustive missingness/type/label audit of every raw column. |
| Text cleanup strips whitespace and normalizes ticker case. | This is not comprehensive publisher-name standardization, historical security-identity resolution, or duplicate detection across URL variants. |

Useful quality evidence includes invalid-row reasons and examples, explicit duplicate rules, exact filtering counts, typed output definitions and temporal coverage summaries. “Cleaned” here means processed under these rules; it does not certify the original source's completeness or every generated label's correctness.

## 10. Information to assemble for the Phase 2 writeup

This is a fact checklist for the eventual writeup, rather than a presentation script.

| Requirement | Material already available here | Information still to establish |
| --- | --- | --- |
| Data sources | Local input locations, required fields, model identifier and provenance locations | Original price/news dataset names, providers and source URLs; confirm any source-specific units or adjustment definitions |
| Relevance | Daily trends, adjusted-close returns, relative trading activity, headline coverage and target sentiment | Team's exact visualization questions and chosen comparisons |
| Collection | Local CSV ingestion, news-column projection and source-to-output processing order | Actual original download/export procedure and acquisition date/version if available |
| Data quality | Scoped missingness, duplicates, validation failures, coverage, types and known limitations | Any original-source quality claims must be verified from that source; do not infer them from file names or zero scoped counters |
| Cleaning and transformation | Implemented steps, their rationale, calculations and handling of null/blank values | Explain why these design choices fit the team's visualization goals |
| Evidence and documentation | Snapshot counts, count reconciliations, report definitions, field dictionaries and evidence links | Select a concise set of figures and examples for the five-minute presentation |

The reports and processed data are generated local files excluded from Git. This reference is maintained documentation under [docs/](./). Its report/output links work locally once those files exist. Update the snapshot figures when using a later run.
