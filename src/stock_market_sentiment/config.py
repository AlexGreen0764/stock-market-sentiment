"""Load and validate preparation settings from JSON and CLI overrides.

Sentiment inference uses fixed settings in its own module; these settings only
control preparation. Configuration files remain standard, comment-free JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoverageThresholds:
    """News qualification limits within the first-to-last eligible article span.

    ``covered_month_fraction`` includes empty months in its denominator.
    ``max_empty_months`` is an upper bound; all other limits are lower bounds.
    Publication-date counts use distinct UTC dates rather than trading sessions.
    """

    span_months: int = 24
    covered_month_fraction: float = 0.75
    max_empty_months: int = 3
    distinct_publication_dates: int = 60
    articles: int = 100


@dataclass(frozen=True)
class PreparationConfig:
    """Immutable run settings with parsed dates and a resolved output path."""

    start_date: date | None
    end_date: date | None
    output: Path
    coverage: CoverageThresholds


def _parse_date(value: Any, field_name: str) -> date | None:
    """Parse an ISO date, permitting null or empty values for open bounds."""
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD)") from error


def load_config(path: Path, root: Path, overrides: dict[str, Any] | None = None) -> PreparationConfig:
    """Load preparation settings and apply explicitly supplied CLI overrides.

    Args:
        path: Required JSON configuration file.
        root: Base directory for a relative output path.
        overrides: CLI values; None entries leave configured values unchanged.

    Returns:
        Immutable settings with inclusive date bounds and validated thresholds.

    Raises:
        TypeError: The top-level JSON or coverage section is not an object.
        ValueError: Dates, threshold names, or threshold values are invalid.
        OSError: The file cannot be read.
    """
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
    # bool subclasses int in Python, but true/false are not meaningful counts.
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
