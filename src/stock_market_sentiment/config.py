"""Configuration loading for the stock-market preparation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoverageThresholds:
    """Minimum news breadth and depth for retaining a ticker."""

    covered_quarters: int = 24
    covered_years: int = 8
    median_articles_per_covered_quarter: int = 3
    distinct_publication_dates: int = 60


@dataclass(frozen=True)
class PreparationConfig:
    """Validated settings for one preparation run."""

    tickers: tuple[str, ...]
    start_date: date | None
    end_date: date | None
    output: Path
    coverage: CoverageThresholds


def _parse_date(value: Any, field_name: str) -> date | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD)") from error


def _parse_tickers(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("tickers must be a non-empty list")
    tickers = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in value if str(ticker).strip()))
    if not tickers:
        raise ValueError("tickers must contain at least one symbol")
    return tickers


def load_config(path: Path, root: Path, overrides: dict[str, Any] | None = None) -> PreparationConfig:
    """Load JSON settings and apply non-None command-line overrides."""
    with path.open(encoding="utf-8") as source:
        values = json.load(source)
    if not isinstance(values, dict):
        raise TypeError(f"Configuration must contain a JSON object: {path}")
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value

    start_date = _parse_date(values.get("start_date"), "start_date")
    end_date = _parse_date(values.get("end_date"), "end_date")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    raw_coverage = values.get("coverage", {})
    if not isinstance(raw_coverage, dict):
        raise TypeError("coverage must be an object")
    unknown = set(raw_coverage) - set(CoverageThresholds.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown coverage settings: {', '.join(sorted(unknown))}")
    if any(type(value) is not int or value < 1 for value in raw_coverage.values()):
        raise ValueError("coverage thresholds must be positive integers")
    coverage = CoverageThresholds(**raw_coverage)

    output = Path(values.get("output", "data/processed"))
    if not output.is_absolute():
        output = root / output
    return PreparationConfig(
        tickers=_parse_tickers(values.get("tickers", [])),
        start_date=start_date,
        end_date=end_date,
        output=output,
        coverage=coverage,
    )
