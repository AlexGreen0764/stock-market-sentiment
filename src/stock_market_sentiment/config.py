"""Load preparation settings and final sentiment-coverage selection rules."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoverageThresholds:
    """News qualification limits across the configured publication window.

    ``covered_month_fraction`` includes all configured months in its denominator.
    ``max_empty_months`` includes window edges and is an upper bound.
    Publication-date counts use distinct UTC dates rather than trading sessions.
    """

    span_months: int = 24
    covered_month_fraction: float = 0.75
    max_empty_months: int = 3
    distinct_publication_dates: int = 60
    articles: int = 100


@dataclass(frozen=True)
class SentimentCoverageThresholds:
    """Require numeric sentiment depth across all configured publication months."""

    articles_per_month: int = 10
    covered_month_fraction: float = 0.8


@dataclass(frozen=True)
class PreparationConfig:
    """Immutable run settings with parsed dates and a resolved output path."""

    start_date: date
    end_date: date
    output: Path
    coverage: CoverageThresholds
    max_tickers: int
    sentiment_coverage: SentimentCoverageThresholds = SentimentCoverageThresholds()


def _parse_date(value: Any, field_name: str) -> date:
    """Require an ISO date so coverage has a fixed denominator."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD)") from error


def load_config(path: Path, root: Path, overrides: dict[str, Any] | None = None) -> PreparationConfig:
    """Validate JSON settings, apply non-None CLI overrides and resolve output."""
    with path.open(encoding="utf-8") as source:
        values = json.load(source)
    if not isinstance(values, dict):
        raise TypeError(f"Configuration must contain a JSON object: {path}")
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value

    start_date = _parse_date(values.get("start_date"), "start_date")
    end_date = _parse_date(values.get("end_date"), "end_date")
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    max_tickers = values.get("max_tickers", 30)
    if type(max_tickers) is not int or max_tickers < 1:
        raise ValueError("max_tickers must be a positive integer")

    raw_coverage = values.get("coverage", {})
    if not isinstance(raw_coverage, dict):
        raise TypeError("coverage must be an object")
    unknown = set(raw_coverage) - set(CoverageThresholds.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown coverage settings: {', '.join(sorted(unknown))}")
    coverage = CoverageThresholds(**raw_coverage)
    # bool subclasses int in Python, but true/false are not meaningful counts.
    for name in ("span_months", "distinct_publication_dates", "articles"):
        value = getattr(coverage, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"coverage.{name} must be a positive integer")
    if type(coverage.max_empty_months) is not int or coverage.max_empty_months < 0:
        raise ValueError("coverage.max_empty_months must be a non-negative integer")
    if type(coverage.covered_month_fraction) not in (int, float) or not 0 < coverage.covered_month_fraction <= 1:
        raise ValueError("coverage.covered_month_fraction must be greater than 0 and at most 1")

    raw_sentiment = values.get("sentiment_coverage", {})
    if not isinstance(raw_sentiment, dict):
        raise TypeError("sentiment_coverage must be an object")
    unknown = set(raw_sentiment) - set(SentimentCoverageThresholds.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown sentiment_coverage settings: {', '.join(sorted(unknown))}")
    sentiment_coverage = SentimentCoverageThresholds(**raw_sentiment)
    if type(sentiment_coverage.articles_per_month) is not int or sentiment_coverage.articles_per_month < 1:
        raise ValueError("sentiment_coverage.articles_per_month must be a positive integer")
    if (type(sentiment_coverage.covered_month_fraction) not in (int, float)
            or not 0 < sentiment_coverage.covered_month_fraction <= 1):
        raise ValueError("sentiment_coverage.covered_month_fraction must be greater than 0 and at most 1")

    output = Path(values.get("output", "data/processed"))
    if not output.is_absolute():
        output = root / output
    return PreparationConfig(
        start_date=start_date,
        end_date=end_date,
        output=output,
        coverage=coverage,
        sentiment_coverage=sentiment_coverage,
        max_tickers=max_tickers,
    )


def parse_config_args(description: str) -> tuple[Path, PreparationConfig]:
    """Use the same repository paths and overrides for preparation and scoring."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        config = load_config(args.config or root / "config" / "default.json", root, {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "output": args.output,
        })
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return root, config
