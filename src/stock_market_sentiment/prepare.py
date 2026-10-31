"""Prepare stock prices and assign news to the next trading session."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median

import pyarrow as pa
from pyarrow import parquet

from .config import CoverageThresholds, load_config

PRICE_COLUMNS = ("date", "open", "high", "low", "close", "adj close", "volume")
PRICE_OUTPUT = ("ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "return_pct", "volume_vs_prior_20d")
EVENT_OUTPUT = ("ticker", "published_at_utc", "article_date_utc", "next_trading_date", "title", "url", "publisher")


def load_prices(folder: Path, tickers: list[str], counts: Counter) -> dict[str, list[dict]]:
    """Read per-ticker CSV files by header name and validate OHLCV rows."""
    result = {}
    for ticker in tickers:
        source_path = folder / f"{ticker}.csv"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing price file: {source_path}")
        by_date = {}
        with source_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not set(PRICE_COLUMNS).issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected price columns in {source_path}: {reader.fieldnames}")
            for raw in reader:
                counts["price_rows_read"] += 1
                try:
                    day = date.fromisoformat(raw["date"])
                    values = {name: float(raw[name]) for name in PRICE_COLUMNS[1:-1]}
                    volume = int(raw["volume"])
                    if (not all(math.isfinite(value) for value in values.values()) or
                            min(values.values()) <= 0 or volume < 0 or
                            values["low"] > min(values["open"], values["close"], values["high"]) or
                            values["high"] < max(values["open"], values["close"], values["low"])):
                        raise ValueError("Invalid price or volume")
                except (ValueError, TypeError, KeyError, OverflowError):
                    counts["price_rows_invalid"] += 1
                    continue
                if day in by_date:
                    counts["price_rows_duplicate_date"] += 1
                    continue
                by_date[day] = {
                    "ticker": ticker, "date": day, "open": values["open"],
                    "high": values["high"], "low": values["low"], "close": values["close"],
                    "adj_close": values["adj close"], "volume": volume,
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
        counts["price_rows_kept"] += len(rows)
        result[ticker] = rows
    return result


def next_trading_day(article_day: date, trading_days: list[date]) -> date | None:
    """Use the first available trading session strictly after the UTC article date."""
    position = bisect_right(trading_days, article_day)
    return trading_days[position] if position < len(trading_days) else None


def write_parquet(path: Path, fields: tuple[str, ...], rows) -> None:
    types = {
        "ticker": pa.string(), "date": pa.date32(),
        "open": pa.float64(), "high": pa.float64(), "low": pa.float64(),
        "close": pa.float64(), "adj_close": pa.float64(), "volume": pa.int64(),
        "return_pct": pa.float64(), "volume_vs_prior_20d": pa.float64(),
        "published_at_utc": pa.timestamp("ms", tz="UTC"),
        "article_date_utc": pa.date32(), "next_trading_date": pa.date32(),
        "title": pa.string(), "url": pa.string(), "publisher": pa.string(),
    }
    schema = pa.schema([(field, types[field]) for field in fields])
    parquet.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def write_outputs(output: Path, tables: dict[str, tuple[tuple[str, ...], list[dict]]]) -> None:
    """Write both Parquet tables before replacing previous outputs."""
    output.mkdir(parents=True, exist_ok=True)
    paths = [(output / f"{name}.parquet.partial", output / f"{name}.parquet") for name in tables]
    try:
        for (fields, rows), (partial, _) in zip(tables.values(), paths):
            write_parquet(partial, fields, rows)
        for partial, final in paths:
            os.replace(partial, final)
    finally:
        for partial, _ in paths:
            partial.unlink(missing_ok=True)


def in_date_range(day: date, start_date: date | None, end_date: date | None) -> bool:
    return (start_date is None or day >= start_date) and (end_date is None or day <= end_date)


def coverage_by_ticker(events: list[dict], tickers: list[str], thresholds: CoverageThresholds) -> dict[str, dict]:
    """Measure publication breadth and depth using deduplicated eligible articles."""
    quarters = defaultdict(Counter)
    publication_dates = defaultdict(set)
    for event in events:
        day = event["article_date_utc"]
        ticker = event["ticker"]
        quarters[ticker][(day.year, (day.month - 1) // 3 + 1)] += 1
        publication_dates[ticker].add(day)

    summary = {}
    for ticker in tickers:
        counts = quarters[ticker]
        metrics = {
            "covered_quarters": len(counts),
            "covered_years": len({year for year, _ in counts}),
            "median_articles_per_covered_quarter": median(counts.values()) if counts else 0,
            "distinct_publication_dates": len(publication_dates[ticker]),
            "articles": sum(counts.values()),
        }
        metrics["passes"] = (
            metrics["covered_quarters"] >= thresholds.covered_quarters
            and metrics["covered_years"] >= thresholds.covered_years
            and metrics["median_articles_per_covered_quarter"] >= thresholds.median_articles_per_covered_quarter
            and metrics["distinct_publication_dates"] >= thresholds.distinct_publication_dates
        )
        summary[ticker] = metrics
    return summary


def prepare(
    root: Path,
    output: Path,
    tickers: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
    coverage: CoverageThresholds | None = None,
) -> dict:
    """Build visualization tables and a quality report from the supplied raw data."""
    coverage = coverage or CoverageThresholds()
    counts = Counter()
    prices = load_prices(root / "data" / "raw" / "full_history", tickers, counts)
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    price_rows = [row for ticker in tickers for row in prices[ticker]]
    trading_days = {ticker: [row["date"] for row in rows] for ticker, rows in prices.items()}
    seen = set()
    event_rows = []
    csv.field_size_limit(100_000_000)
    with (root / "data" / "raw" / "nasdaq_exteral_data.csv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Date", "Article_title", "Stock_symbol", "Url", "Publisher"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected news columns: {reader.fieldnames}")
        for raw in reader:
            counts["article_rows_read"] += 1
            ticker = (raw.get("Stock_symbol") or "").strip().upper()
            if ticker not in prices:
                continue
            counts["article_rows_selected"] += 1
            published = (raw.get("Date") or "").strip()
            try:
                published_at = datetime.strptime(published, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
                article_day = published_at.date()
            except ValueError:
                counts["article_rows_invalid_date"] += 1
                continue
            if not in_date_range(article_day, start_date, end_date):
                counts["article_rows_outside_date_range"] += 1
                continue
            title = (raw.get("Article_title") or "").strip()
            url = (raw.get("Url") or "").strip()
            if not title:
                counts["article_rows_missing_title"] += 1
            if not url:
                counts["article_rows_missing_url"] += 1
            key = (ticker, url) if url else (ticker, published, title)
            if key in seen:
                counts["article_rows_duplicate"] += 1
                continue
            seen.add(key)
            session = next_trading_day(article_day, trading_days[ticker])
            if session is None:
                counts["article_rows_outside_price_history"] += 1
                continue
            event = {
                "ticker": ticker, "published_at_utc": published_at,
                "article_date_utc": article_day, "next_trading_date": session,
                "title": title, "url": url, "publisher": (raw.get("Publisher") or "").strip(),
            }
            if in_date_range(session, start_date, end_date):
                event_rows.append(event)
            else:
                counts["article_rows_outside_date_range"] += 1

    coverage_summary = coverage_by_ticker(event_rows, tickers, coverage)
    retained_tickers = [ticker for ticker in tickers if coverage_summary[ticker]["passes"]]
    retained = set(retained_tickers)
    counts["article_rows_before_coverage"] = len(event_rows)
    event_rows = [event for event in event_rows if event["ticker"] in retained]
    counts["article_rows_kept"] = len(event_rows)
    selected_price_rows = [
        row for row in price_rows
        if row["ticker"] in retained
        and in_date_range(row["date"], start_date, end_date)
    ]
    tables = {
        "daily_prices": (PRICE_OUTPUT, selected_price_rows),
        "news_events": (EVENT_OUTPUT, event_rows),
    }
    report = {
        "candidate_tickers": tickers,
        "retained_tickers": retained_tickers,
        "rejected_tickers": [ticker for ticker in tickers if ticker not in retained],
        "coverage_thresholds": {
            "covered_quarters": coverage.covered_quarters,
            "covered_years": coverage.covered_years,
            "median_articles_per_covered_quarter": coverage.median_articles_per_covered_quarter,
            "distinct_publication_dates": coverage.distinct_publication_dates,
        },
        "coverage_by_ticker": coverage_summary,
        "counts": {key: counts[key] for key in (
            "price_rows_read", "price_rows_invalid", "price_rows_duplicate_date", "price_rows_kept",
            "article_rows_read", "article_rows_selected", "article_rows_invalid_date",
            "article_rows_missing_title", "article_rows_missing_url", "article_rows_duplicate",
            "article_rows_outside_price_history", "article_rows_outside_date_range",
            "article_rows_before_coverage", "article_rows_kept",
        )},
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "price_date_ranges": {
            ticker: {"first": rows[0]["date"].isoformat(), "last": rows[-1]["date"].isoformat(), "rows": len(rows)} if rows else None
            for ticker, rows in prices.items()
        },
        "news_link_rule": "Each UTC article date is assigned to the first trading date strictly after it. Same-day reaction is excluded.",
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "quality_report.json.partial"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(report_path, report_dir / "quality_report.json")
    if not retained_tickers:
        raise ValueError("No ticker meets the news coverage thresholds; inspect reports/quality_report.json")
    write_outputs(output, tables)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    config_path = args.config or args.root / "config" / "default.json"
    try:
        config = load_config(config_path, args.root, {
            "tickers": args.tickers,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "output": args.output,
        })
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report = prepare(
        args.root,
        config.output,
        list(config.tickers),
        config.start_date,
        config.end_date,
        config.coverage,
    )
    print(json.dumps(report, indent=2))
