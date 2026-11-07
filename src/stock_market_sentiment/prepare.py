"""Prepare and rank eligible ticker histories for adaptive sentiment selection.

Candidate files stay in data/interim; scoring selects the final Power BI exports.
Publication dates are UTC and reaction dates come from observed price sessions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
from pyarrow import csv as arrow_csv
from pyarrow import parquet

from .config import CoverageThresholds, PreparationConfig, parse_config_args

PRICE_COLUMNS = ("date", "open", "high", "low", "close", "adj close", "volume")
PRICE_COUNT_KEYS = (
    "price_rows_read", "price_rows_valid", "price_rows_invalid", "price_rows_duplicate_date",
    "price_rows_invalid_parse", "price_rows_invalid_nonfinite", "price_rows_invalid_nonpositive_price",
    "price_rows_invalid_volume", "price_rows_invalid_ohlc_bounds",
)
# Explicit Arrow types keep dates usable as Power BI relationship keys and give
# empty outputs the same schema as populated outputs.
PRICE_SCHEMA = pa.schema([
    ("ticker", pa.string()), ("date", pa.date32()),
    ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()),
    ("close", pa.float64()), ("adj_close", pa.float64()), ("volume", pa.int64()),
    ("return_pct", pa.float64()), ("volume_vs_prior_20d", pa.float64()),
    ("news_coverage_pass", pa.bool_()),
    ("news_coverage_start", pa.date32()), ("news_coverage_end", pa.date32()),
])
EVENT_SCHEMA = pa.schema([
    ("ticker", pa.string()), ("published_at_utc", pa.timestamp("ms", tz="UTC")),
    ("article_date_utc", pa.date32()), ("next_trading_date", pa.date32()),
    ("title", pa.string()), ("url", pa.string()), ("publisher", pa.string()),
    ("has_url", pa.bool_()), ("has_publisher", pa.bool_()),
])


def load_prices(source_path: Path, counts: Counter, invalid_examples: dict[str, list[dict]]) -> list[dict]:
    """Validate and sort one ticker's prices, retaining the first valid row per date.

    Calculate features over full history; count rejections and retain up to three
    examples per reason. Decimal parsing keeps large integral volumes exact.
    """
    ticker = source_path.stem.strip().upper()
    by_date = {}
    with source_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not set(PRICE_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected price columns in {source_path}: {reader.fieldnames}")
        for raw in reader:
            counts["price_rows_read"] += 1
            reason = None
            try:
                day = date.fromisoformat(raw["date"])
                values = {name: float(raw[name]) for name in PRICE_COLUMNS[1:-1]}
                # Accept integral decimal notation without rounding large volumes
                # through a binary float or silently truncating fractional shares.
                volume_value = Decimal(raw["volume"])
            except (ValueError, TypeError, KeyError, OverflowError, InvalidOperation):
                reason = "parse"
            else:
                if not all(math.isfinite(value) for value in values.values()):
                    reason = "nonfinite"
                elif min(values.values()) <= 0:
                    reason = "nonpositive_price"
                elif (not volume_value.is_finite() or volume_value < 0 or volume_value > 2**63 - 1
                      or volume_value != volume_value.to_integral_value()):
                    reason = "volume"
                elif (values["low"] > min(values["open"], values["close"], values["high"])
                      or values["high"] < max(values["open"], values["close"], values["low"])):
                    reason = "ohlc_bounds"
            # Assign one rejection reason per row so audit subtotals reconcile.
            if reason:
                counts["price_rows_invalid"] += 1
                counts[f"price_rows_invalid_{reason}"] += 1
                examples = invalid_examples.setdefault(reason, [])
                if len(examples) < 3:
                    examples.append({"ticker": ticker, "source_file": source_path.name, "row": raw})
                continue
            if day in by_date:
                counts["price_rows_duplicate_date"] += 1
                continue
            by_date[day] = {
                "ticker": ticker, "date": day, "open": values["open"],
                "high": values["high"], "low": values["low"], "close": values["close"],
                "adj_close": values["adj close"], "volume": int(volume_value),
            }
    # Calculate using full history before the export window is applied, so its
    # first exported return can still reference the preceding available session.
    rows = [by_date[day] for day in sorted(by_date)]
    previous_adjusted_close = None
    previous_volumes = deque(maxlen=20)
    for row in rows:
        row["return_pct"] = (
            round(100 * (row["adj_close"] / previous_adjusted_close - 1), 4)
            if previous_adjusted_close is not None else None
        )
        # The current volume enters the deque only after its ratio is calculated.
        average = sum(previous_volumes) / len(previous_volumes) if previous_volumes else 0
        row["volume_vs_prior_20d"] = round(row["volume"] / average, 4) if average else None
        previous_adjusted_close = row["adj_close"]
        previous_volumes.append(row["volume"])
    counts["price_rows_valid"] += len(rows)
    return rows


def next_trading_day(article_day: date, trading_days: list[date]) -> date | None:
    """Return the first observed session strictly after a UTC article date.

    ``trading_days`` must be sorted ascending. Return None for pre-history dates,
    exhausted histories, or gaps over seven calendar days. The gap limit guards
    against stale matches; it does not validate an exchange holiday calendar.
    """
    if not trading_days or article_day < trading_days[0]:
        return None
    # Exclude same-day reactions because source timestamps may be placeholders.
    position = bisect_right(trading_days, article_day)
    if position == len(trading_days):
        return None
    session = trading_days[position]
    return session if (session - article_day).days <= 7 else None


def match_articles(articles: list[dict], trading_days: list[date], counts: Counter) -> list[dict]:
    """Link one ticker's articles to usable sessions, then remove duplicate records."""
    events = []
    seen = set()
    for event in articles:
        session = next_trading_day(event["article_date_utc"], trading_days)
        if session is None:
            counts["article_rows_unusable_price_match"] += 1
            continue
        key = event["url"] or (event["published_at_utc"], event["title"])
        if key in seen:
            counts["article_rows_duplicate"] += 1
            continue
        seen.add(key)
        events.append({**event, "next_trading_date": session})
    return events


def coverage_by_ticker(
    events: list[dict], tickers: list[str], thresholds: CoverageThresholds, start_date: date, end_date: date,
) -> dict[str, dict]:
    """Summarize coverage and failed criteria for every supplied ticker.

    Events must already be deduplicated and matched to usable price sessions.
    The denominator includes every configured month. Empty runs include leading
    and trailing months, so a short burst cannot masquerade as full coverage.
    The separate observed span remains subject to the minimum span requirement.
    """
    months = defaultdict(Counter)
    publication_dates = defaultdict(set)
    years = defaultdict(Counter)
    for event in events:
        day = event["article_date_utc"]
        ticker = event["ticker"]
        # Consecutive month indices make gaps comparable across year boundaries.
        months[ticker][day.year * 12 + day.month - 1] += 1
        publication_dates[ticker].add(day)
        years[ticker][day.year] += 1

    summary = {}
    first_month = start_date.year * 12 + start_date.month - 1
    last_month = end_date.year * 12 + end_date.month - 1
    window_months = last_month - first_month + 1
    for ticker in tickers:
        counts = months[ticker]
        occupied = sorted(counts)
        span = occupied[-1] - occupied[0] + 1 if occupied else 0
        # Sentinel endpoints include gaps at both edges of the requested window.
        longest_gap = max(right - left - 1 for left, right in pairwise([first_month - 1, *occupied, last_month + 1]))
        dates = publication_dates[ticker]
        metrics = {
            "coverage_start": min(dates).isoformat() if dates else None,
            "coverage_end": max(dates).isoformat() if dates else None,
            "span_months": span,
            "window_months": window_months,
            "covered_months": len(counts),
            "covered_month_fraction": len(counts) / window_months,
            "max_empty_months": longest_gap,
            "distinct_publication_dates": len(dates),
            "articles": sum(counts.values()),
            "articles_by_year": dict(sorted(years[ticker].items())),
        }
        checks = {
            "span_months": span >= thresholds.span_months,
            "covered_month_fraction": metrics["covered_month_fraction"] >= thresholds.covered_month_fraction,
            "max_empty_months": longest_gap <= thresholds.max_empty_months,
            "distinct_publication_dates": len(dates) >= thresholds.distinct_publication_dates,
            "articles": metrics["articles"] >= thresholds.articles,
        }
        metrics["failed_criteria"] = [name for name, passes in checks.items() if not passes]
        metrics["passes"] = all(checks.values())
        summary[ticker] = metrics
    return summary


def load_articles(
    path: Path, tickers: set[str], start_date: date, end_date: date,
    counts: Counter, ticker_counts: dict[str, Counter],
) -> dict[str, list[dict]]:
    """Read candidate headlines once, without materializing bodies or summaries.

    Broad articles remain eligible. Missing URLs and publishers are flagged,
    while empty titles and invalid/out-of-window dates are excluded.
    """
    columns = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
    articles = defaultdict(list)
    with arrow_csv.open_csv(
        path, read_options=arrow_csv.ReadOptions(block_size=16 * 1024 * 1024),
        parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
        convert_options=arrow_csv.ConvertOptions(
            include_columns=columns, column_types={name: pa.string() for name in columns},
            strings_can_be_null=False,
        ),
    ) as source:
        for batch in source:
            for published, title, ticker, url, publisher in zip(*(column.to_pylist() for column in batch.columns)):
                counts["article_rows_read"] += 1
                ticker = ticker.strip().upper()
                if ticker not in tickers:
                    counts["article_rows_unmatched_ticker"] += 1
                    continue
                counts["article_rows_selected"] += 1
                ticker_counts[ticker]["matched_articles"] += 1
                try:
                    timestamp = datetime.strptime(published.strip(), "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=UTC)
                except ValueError:
                    counts["article_rows_invalid_date"] += 1
                    ticker_counts[ticker]["invalid_date"] += 1
                    continue
                day = timestamp.date()
                if not start_date <= day <= end_date:
                    counts["article_rows_outside_date_range"] += 1
                    ticker_counts[ticker]["outside_date_range"] += 1
                    continue
                ticker_counts[ticker]["in_date_range"] += 1
                title, url, publisher = title.strip(), url.strip(), publisher.strip()
                if not title:
                    counts["article_rows_missing_title"] += 1
                    continue
                counts["article_rows_missing_url"] += not bool(url)
                counts["article_rows_missing_publisher"] += not bool(publisher)
                articles[ticker].append({
                    "ticker": ticker, "published_at_utc": timestamp, "article_date_utc": day,
                    "title": title, "url": url, "publisher": publisher,
                    "has_url": bool(url), "has_publisher": bool(publisher),
                })
    return articles


def _write_candidate(path: Path, rows: list[dict], schema: pa.Schema) -> str:
    """Commit one typed candidate file and return its exact content fingerprint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        parquet.write_table(pa.Table.from_pylist(rows, schema=schema), partial)
        with partial.open("rb") as source:
            fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
        os.replace(partial, path)
        return fingerprint
    finally:
        partial.unlink(missing_ok=True)


def prepare(root: Path, config: PreparationConfig) -> dict:
    """Clean and rank candidates; commit their preparation manifest last.

    Final selection and production outputs belong to the sentiment stage.
    """
    start_date, end_date, coverage = config.start_date, config.end_date, config.coverage
    source_files = sorted(path for path in (root / "data/raw/full_history").glob("*.csv") if path.is_file())
    price_files = {path.stem.strip().upper(): path for path in source_files}
    if len(price_files) != len(source_files):
        raise ValueError("Multiple price files resolve to the same normalized ticker")
    tickers = sorted(price_files)
    if not tickers:
        raise ValueError("No price CSV files found")
    counts = Counter()
    ticker_counts = defaultdict(Counter)
    news_source = root / "data/raw/nasdaq_exteral_data.csv"
    articles = load_articles(
        news_source, set(tickers), start_date, end_date, counts, ticker_counts,
    )
    coverage_summary = {}
    price_validation = {}
    price_ranges = {}
    invalid_price_examples = {}
    prepared_candidates = {}
    interim = root / "data" / "interim"
    for ticker in tickers:
        price_counts = Counter()
        rows = load_prices(price_files[ticker], price_counts, invalid_price_examples)
        counts.update(price_counts)
        price_validation[ticker] = {key: price_counts[key] for key in PRICE_COUNT_KEYS}
        trading_days = [row["date"] for row in rows]
        price_ranges[ticker] = {
            "first": trading_days[0].isoformat() if rows else None,
            "last": trading_days[-1].isoformat() if rows else None,
            "input_rows": len(rows),
        }
        events = match_articles(articles.pop(ticker, []), trading_days, counts)
        metrics = coverage_by_ticker(events, [ticker], coverage, start_date, end_date)[ticker]
        coverage_summary[ticker] = metrics
        counts["article_rows_before_coverage"] += len(events)
        if metrics["passes"]:
            linked = {event["next_trading_date"] for event in events}
            prepared_prices = []
            for row in rows:
                if start_date <= row["date"] <= end_date or row["date"] in linked:
                    row.update(
                        news_coverage_pass=True,
                        news_coverage_start=date.fromisoformat(metrics["coverage_start"]),
                        news_coverage_end=date.fromisoformat(metrics["coverage_end"]),
                    )
                    prepared_prices.append(row)
            folder = interim / "prepared" / ticker
            prepared_candidates[ticker] = {
                "news_sha256": _write_candidate(folder / "news.parquet", events, EVENT_SCHEMA),
                "prices_sha256": _write_candidate(folder / "prices.parquet", prepared_prices, PRICE_SCHEMA),
                "article_rows": len(events), "price_rows": len(prepared_prices),
            }
    ranked = sorted(prepared_candidates, key=lambda ticker: (
        -coverage_summary[ticker]["covered_months"],
        -coverage_summary[ticker]["distinct_publication_dates"], ticker,
    ))
    report = {
        "prepared_candidates": prepared_candidates,
        "prepared_counts": {
            "news_rows": sum(candidate["article_rows"] for candidate in prepared_candidates.values()),
            "price_rows": sum(candidate["price_rows"] for candidate in prepared_candidates.values()),
        },
        "candidate_tickers": tickers,
        "rejected_tickers": [ticker for ticker in tickers if not coverage_summary[ticker]["passes"]],
        "qualified_tickers": sorted(ranked),
        "ticker_ranking": ranked,
        "selection": {
            "ranking_rule": "Covered months descending, distinct publication dates descending, ticker ascending",
            "article_identity": "Exact stripped URL within ticker, or publication timestamp and title when URL is missing",
            "coverage_window": "All configured months, including leading and trailing gaps",
        },
        "coverage_thresholds": asdict(coverage), "coverage_by_ticker": coverage_summary,
        "article_filter_counts_by_ticker": {ticker: dict(ticker_counts[ticker]) for ticker in tickers},
        "price_validation_by_ticker": price_validation, "invalid_price_examples": invalid_price_examples,
        "price_date_ranges": price_ranges,
        "counts": {key: counts[key] for key in (
            *PRICE_COUNT_KEYS, "article_rows_read", "article_rows_selected",
            "article_rows_unmatched_ticker", "article_rows_invalid_date", "article_rows_missing_title",
            "article_rows_missing_url", "article_rows_missing_publisher", "article_rows_duplicate",
            "article_rows_unusable_price_match", "article_rows_outside_date_range",
            "article_rows_before_coverage",
        )},
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "news_link_rule": "First observed session strictly after the UTC publication date, within 7 calendar days; pre-history articles excluded. Linked sessions may extend beyond end_date.",
    }
    interim.mkdir(parents=True, exist_ok=True)
    manifest_partial = interim / "manifest.json.partial"
    try:
        manifest_partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_partial, interim / "manifest.json")
    finally:
        manifest_partial.unlink(missing_ok=True)
    return report


def main() -> None:
    """Prepare all candidates and print a concise manifest summary."""
    root, config = parse_config_args(__doc__)
    report = prepare(root, config)
    print(f"Prepared {len(report['ticker_ranking']):,} candidates; manifest: {root / 'data/interim/manifest.json'}")
