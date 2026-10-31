"""Prepare stock prices and assign news to the next trading session."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
from pyarrow import parquet

from .config import CoverageThresholds, load_config

PRICE_COLUMNS = ("date", "open", "high", "low", "close", "adj close", "volume")
PRICE_COUNT_KEYS = (
    "price_rows_read", "price_rows_valid", "price_rows_invalid", "price_rows_duplicate_date",
    "price_rows_invalid_parse", "price_rows_invalid_nonfinite", "price_rows_invalid_nonpositive_price",
    "price_rows_invalid_volume", "price_rows_invalid_ohlc_bounds",
)
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
    """Read per-ticker CSV files by header name and validate OHLCV rows."""
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
    rows = [by_date[day] for day in sorted(by_date)]
    previous_adjusted_close = None
    previous_volumes = deque(maxlen=20)
    for row in rows:
        row["return_pct"] = (
            round(100 * (row["adj_close"] / previous_adjusted_close - 1), 4)
            if previous_adjusted_close is not None else None
        )
        average = sum(previous_volumes) / len(previous_volumes) if previous_volumes else 0
        row["volume_vs_prior_20d"] = round(row["volume"] / average, 4) if average else None
        previous_adjusted_close = row["adj_close"]
        previous_volumes.append(row["volume"])
    counts["price_rows_valid"] += len(rows)
    return rows


def next_trading_day(article_day: date, trading_days: list[date]) -> date | None:
    """Find a later session within seven days, excluding pre-history articles."""
    if not trading_days or article_day < trading_days[0]:
        return None
    position = bisect_right(trading_days, article_day)
    if position == len(trading_days):
        return None
    session = trading_days[position]
    return session if (session - article_day).days <= 7 else None


def in_date_range(day: date, start_date: date | None, end_date: date | None) -> bool:
    return (start_date is None or day >= start_date) and (end_date is None or day <= end_date)


def coverage_by_ticker(events: list[dict], tickers: list[str], thresholds: CoverageThresholds) -> dict[str, dict]:
    """Measure monthly continuity, including empty months between first and last."""
    months = defaultdict(Counter)
    publication_dates = defaultdict(set)
    years = defaultdict(Counter)
    for event in events:
        day = event["article_date_utc"]
        ticker = event["ticker"]
        months[ticker][day.year * 12 + day.month - 1] += 1
        publication_dates[ticker].add(day)
        years[ticker][day.year] += 1

    summary = {}
    for ticker in tickers:
        counts = months[ticker]
        occupied = sorted(counts)
        span = occupied[-1] - occupied[0] + 1 if occupied else 0
        longest_gap = max((right - left - 1 for left, right in pairwise(occupied)), default=0)
        dates = publication_dates[ticker]
        metrics = {
            "coverage_start": min(dates).isoformat() if dates else None,
            "coverage_end": max(dates).isoformat() if dates else None,
            "span_months": span,
            "covered_months": len(counts),
            "covered_month_fraction": len(counts) / span if span else 0,
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
    path: Path, tickers: set[str], start_date: date | None, end_date: date | None,
    counts: Counter, ticker_counts: dict[str, Counter],
) -> dict[str, list[dict]]:
    """Read matching in-window article metadata, leaving article bodies on disk."""
    articles = defaultdict(list)
    csv.field_size_limit(100_000_000)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Date", "Article_title", "Stock_symbol", "Url", "Publisher"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected news columns: {reader.fieldnames}")
        for raw in reader:
            counts["article_rows_read"] += 1
            ticker = (raw.get("Stock_symbol") or "").strip().upper()
            if ticker not in tickers:
                counts["article_rows_unmatched_ticker"] += 1
                continue
            counts["article_rows_selected"] += 1
            ticker_counts[ticker]["matched_articles"] += 1
            published = (raw.get("Date") or "").strip()
            try:
                published_at = datetime.strptime(published, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
                article_day = published_at.date()
            except ValueError:
                counts["article_rows_invalid_date"] += 1
                ticker_counts[ticker]["invalid_date"] += 1
                continue
            if not in_date_range(article_day, start_date, end_date):
                counts["article_rows_outside_date_range"] += 1
                ticker_counts[ticker]["outside_date_range"] += 1
                continue
            ticker_counts[ticker]["in_date_range"] += 1
            title = (raw.get("Article_title") or "").strip()
            url = (raw.get("Url") or "").strip()
            if not title:
                counts["article_rows_missing_title"] += 1
                continue
            if not url:
                counts["article_rows_missing_url"] += 1
            publisher = (raw.get("Publisher") or "").strip()
            if not publisher:
                counts["article_rows_missing_publisher"] += 1
            articles[ticker].append({
                "ticker": ticker, "published_at_utc": published_at,
                "article_date_utc": article_day,
                "title": title, "url": url, "publisher": publisher,
                "has_url": bool(url), "has_publisher": bool(publisher),
            })
    return articles


def prepare(
    root: Path,
    output: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    coverage: CoverageThresholds | None = None,
) -> dict:
    """Evaluate every price-file ticker and write prices independently of news coverage."""
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    coverage = coverage or CoverageThresholds()
    price_folder = root / "data" / "raw" / "full_history"
    source_files = sorted(path for path in price_folder.glob("*.csv") if path.is_file())
    price_files = {path.stem.strip().upper(): path for path in source_files}
    if len(price_files) != len(source_files):
        raise ValueError("Multiple price files resolve to the same normalized ticker")
    tickers = sorted(price_files)
    if not tickers:
        raise ValueError(f"No price CSV files found in {price_folder}")
    counts = Counter()
    ticker_counts = defaultdict(Counter)
    articles = load_articles(
        root / "data" / "raw" / "nasdaq_exteral_data.csv", set(tickers),
        start_date, end_date, counts, ticker_counts,
    )
    coverage_summary = {}
    price_ranges = {}
    price_validation = {}
    invalid_price_examples = {}
    price_tickers = []
    output.mkdir(parents=True, exist_ok=True)
    price_path = output / "daily_prices.parquet.partial"
    news_path = output / "news_events.parquet.partial"
    try:
        with parquet.ParquetWriter(price_path, PRICE_SCHEMA) as price_writer, parquet.ParquetWriter(
            news_path, EVENT_SCHEMA
        ) as news_writer:
            for ticker in tickers:
                price_counts = Counter()
                rows = load_prices(price_files[ticker], price_counts, invalid_price_examples)
                counts.update(price_counts)
                price_validation[ticker] = {key: price_counts[key] for key in PRICE_COUNT_KEYS}
                trading_days = [row["date"] for row in rows]
                events = []
                seen = set()
                for event in articles.pop(ticker, []):
                    session = next_trading_day(event["article_date_utc"], trading_days)
                    if session is None:
                        counts["article_rows_unusable_price_match"] += 1
                        continue
                    key = event["url"] or (event["published_at_utc"], event["title"])
                    if key in seen:
                        counts["article_rows_duplicate"] += 1
                        continue
                    seen.add(key)
                    event["next_trading_date"] = session
                    events.append(event)
                metrics = coverage_by_ticker(events, [ticker], coverage)[ticker]
                coverage_summary[ticker] = metrics
                counts["article_rows_before_coverage"] += len(events)
                linked_sessions = set()
                if metrics["passes"]:
                    news_writer.write_table(pa.Table.from_pylist(events, schema=EVENT_SCHEMA))
                    counts["article_rows_kept"] += len(events)
                    counts["article_rows_kept_missing_publisher"] += sum(not event["has_publisher"] for event in events)
                    counts["article_rows_kept_missing_url"] += sum(not event["has_url"] for event in events)
                    linked_sessions = {event["next_trading_date"] for event in events}
                selected_rows = [
                    row for row in rows
                    if in_date_range(row["date"], start_date, end_date) or row["date"] in linked_sessions
                ]
                coverage_start = date.fromisoformat(metrics["coverage_start"]) if metrics["coverage_start"] else None
                coverage_end = date.fromisoformat(metrics["coverage_end"]) if metrics["coverage_end"] else None
                for row in selected_rows:
                    row.update(
                        news_coverage_pass=metrics["passes"],
                        news_coverage_start=coverage_start, news_coverage_end=coverage_end,
                    )
                if selected_rows:
                    price_writer.write_table(pa.Table.from_pylist(selected_rows, schema=PRICE_SCHEMA))
                    price_tickers.append(ticker)
                counts["price_rows_kept"] += len(selected_rows)
                price_ranges[ticker] = {
                    "first": trading_days[0].isoformat() if trading_days else None,
                    "last": trading_days[-1].isoformat() if trading_days else None,
                    "input_rows": len(rows), "exported_rows": len(selected_rows),
                }
        os.replace(price_path, output / "daily_prices.parquet")
        os.replace(news_path, output / "news_events.parquet")
    finally:
        price_path.unlink(missing_ok=True)
        news_path.unlink(missing_ok=True)

    retained_tickers = [ticker for ticker in tickers if coverage_summary[ticker]["passes"]]
    retained_articles_by_year = Counter()
    retained_tickers_by_year = Counter()
    for ticker in retained_tickers:
        retained_articles_by_year.update(coverage_summary[ticker]["articles_by_year"])
        retained_tickers_by_year.update(coverage_summary[ticker]["articles_by_year"].keys())
    report = {
        "candidate_tickers": tickers,
        "price_tickers": price_tickers,
        "retained_tickers": retained_tickers,
        "rejected_tickers": [ticker for ticker in tickers if not coverage_summary[ticker]["passes"]],
        "coverage_thresholds": asdict(coverage),
        "coverage_by_ticker": coverage_summary,
        "article_filter_counts_by_ticker": {ticker: dict(ticker_counts[ticker]) for ticker in tickers},
        "price_validation_by_ticker": price_validation,
        "invalid_price_examples": invalid_price_examples,
        "retained_news_quality": {
            "article_ticker_records": counts["article_rows_kept"],
            "missing_publisher": counts["article_rows_kept_missing_publisher"],
            "missing_publisher_fraction": (
                counts["article_rows_kept_missing_publisher"] / counts["article_rows_kept"]
                if counts["article_rows_kept"] else None
            ),
            "missing_url": counts["article_rows_kept_missing_url"],
        },
        "retained_news_by_year": {
            year: {"article_ticker_records": total, "tickers_with_articles": retained_tickers_by_year[year]}
            for year, total in sorted(retained_articles_by_year.items())
        },
        "counts": {key: counts[key] for key in (
            *PRICE_COUNT_KEYS, "price_rows_kept",
            "article_rows_read", "article_rows_selected", "article_rows_unmatched_ticker", "article_rows_invalid_date",
            "article_rows_missing_title", "article_rows_missing_url", "article_rows_missing_publisher", "article_rows_duplicate",
            "article_rows_unusable_price_match", "article_rows_outside_date_range",
            "article_rows_before_coverage", "article_rows_kept",
            "article_rows_kept_missing_publisher", "article_rows_kept_missing_url",
        )},
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "price_date_ranges": price_ranges,
        "news_link_rule": "First observed session strictly after the UTC publication date, within 7 calendar days; pre-history articles excluded. Linked sessions may extend beyond end_date.",
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "quality_report.json.partial"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(report_path, report_dir / "quality_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    config_path = args.config or args.root / "config" / "default.json"
    try:
        config = load_config(config_path, args.root, {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "output": args.output,
        })
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report = prepare(
        args.root,
        config.output,
        config.start_date,
        config.end_date,
        config.coverage,
    )
    print(json.dumps(report, indent=2))
