"""Configuration loading for the stock-market preparation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoverageThresholds:
    """Minimum news coverage within a ticker's first-to-last article months."""

    span_months: int = 24
    covered_month_fraction: float = 0.75
    max_empty_months: int = 3
    distinct_publication_dates: int = 60
    articles: int = 100


@dataclass(frozen=True)
class PreparationConfig:
    """Validated settings for one preparation run."""

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
    coverage = CoverageThresholds(**raw_coverage)
    for name in ("span_months", "distinct_publication_dates", "articles"):
        value = getattr(coverage, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"coverage.{name} must be a positive integer")
    if type(coverage.max_empty_months) is not int or coverage.max_empty_months < 0:
        raise ValueError("coverage.max_empty_months must be a non-negative integer")
    if type(coverage.covered_month_fraction) not in (int, float) or not 0 < coverage.covered_month_fraction <= 1:
        raise ValueError("coverage.covered_month_fraction must be greater than 0 and at most 1")

    output = Path(values.get("output", "data/processed"))
    if not output.is_absolute():
        output = root / output
    return PreparationConfig(
        start_date=start_date,
        end_date=end_date,
        output=output,
        coverage=coverage,
    )
