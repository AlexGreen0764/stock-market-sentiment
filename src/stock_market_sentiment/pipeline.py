"""Select a compact ticker universe using complete, reusable sentiment histories."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import monotonic

import pyarrow as pa
from pyarrow import parquet

from . import sentiment
from .config import PreparationConfig
from .prepare import EVENT_SCHEMA, PRICE_SCHEMA

SCORED_SCHEMA = pa.schema([*EVENT_SCHEMA, *sentiment.SENTIMENT_SCHEMA])


def load_securities(root: Path) -> dict[str, dict[str, str]]:
    """Load target names and stock/ETF types from the historical symbol catalog."""
    securities = {}
    catalog = root / "data/raw/symbols_valid_meta.csv"
    with catalog.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not {"Symbol", "Security Name", "ETF"}.issubset(reader.fieldnames or []):
            raise ValueError("symbols_valid_meta.csv requires Symbol, Security Name and ETF columns")
        for row in reader:
            ticker, name, fund = (row[field].strip() for field in ("Symbol", "Security Name", "ETF"))
            if not ticker or not name or fund not in {"Y", "N"}:
                raise ValueError(f"Invalid security catalog entry: {ticker!r}")
            ticker = ticker.upper()
            if ticker in securities:
                raise ValueError(f"Duplicate security catalog entry: {ticker}")
            securities[ticker] = {"target_name": name, "security_type": "ETF" if fund == "Y" else "stock"}
    return securities


def sentiment_coverage(table: pa.Table, config: PreparationConfig) -> dict:
    """Measure numeric ratings against every month in the publication window.

    Neutral is numeric; mixed and insufficient information are abstentions.
    Counts describe retained article records, including repeated headlines with
    distinct article identities. They are not independent market observations.
    """
    first = config.start_date.year * 12 + config.start_date.month - 1
    last = config.end_date.year * 12 + config.end_date.month - 1
    monthly = dict.fromkeys(range(first, last + 1), 0)
    dates = set()
    directional = 0
    for day, score in zip(table["article_date_utc"].to_pylist(), table["sentiment_score"].to_pylist()):
        if day is None or not config.start_date <= day <= config.end_date:
            raise ValueError("Candidate publication date falls outside the prepared window")
        if score is not None:
            monthly[day.year * 12 + day.month - 1] += 1
            dates.add(day)
            directional += score != 0
    counts = list(monthly.values())
    numeric = sum(counts)
    sufficient = sum(count >= config.sentiment_coverage.articles_per_month for count in counts)
    fraction = sufficient / len(counts)
    empty_run = longest = 0
    for count in counts:
        empty_run = empty_run + 1 if count == 0 else 0
        longest = max(longest, empty_run)
    passes = fraction >= config.sentiment_coverage.covered_month_fraction
    return {
        "article_rows": table.num_rows, "numeric_score_rows": numeric,
        "numeric_score_fraction": numeric / table.num_rows if table.num_rows else 0,
        "directional_score_rows": directional, "distinct_publication_dates": len(dates),
        "window_months": len(counts), "months_with_numeric_scores": sum(count > 0 for count in counts),
        "sufficient_months": sufficient, "sufficient_month_fraction": fraction,
        "minimum_monthly_scores": min(counts), "median_monthly_scores": median(counts),
        "max_empty_months": longest,
        "numeric_scores_by_month": {f"{month // 12:04d}-{month % 12 + 1:02d}": count for month, count in monthly.items()},
        "failed_criteria": [] if passes else ["covered_month_fraction"], "passes": passes,
    }


def _fingerprint(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _validate_table(table: pa.Table, schema: pa.Schema, ticker: str, rows: int) -> None:
    if not table.schema.equals(schema, check_metadata=False) or table.num_rows != rows or rows < 1:
        raise ValueError(f"Invalid candidate schema or row count for {ticker}")
    if table["ticker"].unique().to_pylist() != [ticker]:
        raise ValueError(f"Candidate contains an unexpected ticker: {ticker}")
    if "sentiment_label" in schema.names:
        for label, score in zip(table["sentiment_label"].to_pylist(), table["sentiment_score"].to_pylist()):
            if sentiment.summarize_label(label)["sentiment_score"] != score:
                raise ValueError(f"Invalid cached sentiment score for {ticker}: {label}")


def _load_manifest(root: Path, config: PreparationConfig) -> dict:
    manifest = json.loads((root / "data/interim/manifest.json").read_text(encoding="utf-8"))
    if (manifest["start_date"] != config.start_date.isoformat()
            or manifest["end_date"] != config.end_date.isoformat()
            or manifest["coverage_thresholds"] != asdict(config.coverage)):
        raise ValueError("Preparation settings changed; rebuild candidates before scoring")
    ranking = manifest["ticker_ranking"]
    if len(set(ranking)) != len(ranking) or set(ranking) != set(manifest["prepared_candidates"]):
        raise ValueError("Prepared candidate ranking is inconsistent; rebuild candidates")
    for ticker in ranking:
        if (not isinstance(ticker, str) or not ticker or ticker != ticker.strip().upper()
                or ticker in {".", ".."} or any(char in ticker for char in "/\\:")):
            raise ValueError(f"Invalid candidate ticker: {ticker!r}")
    return manifest


def _checkpoint(path: Path, input_hash: str, recipe_hash: str, ticker: str, rows: int) -> tuple[pa.Table, int] | None:
    if not path.exists():
        return None
    metadata = parquet.read_schema(path).metadata or {}
    if (metadata.get(b"candidate_input_sha256") != input_hash.encode()
            or metadata.get(b"sentiment_recipe_sha256") != recipe_hash.encode()):
        return None
    table = parquet.read_table(path)
    _validate_table(table, SCORED_SCHEMA, ticker, rows)
    inferences = int(metadata[b"headline_inferences"])
    if not 0 < inferences <= rows:
        raise ValueError(f"Invalid cached inference count for {ticker}")
    return table, inferences


def _write_checkpoint(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        parquet.write_table(table, partial)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _publish(outputs: dict[Path, pa.Table | dict]) -> None:
    """Stage all outputs, then replace them with rollback on replacement errors.

    Staging beside each destination supports custom output drives. This handles
    ordinary write errors; separate files cannot be crash-atomic as a group.
    """
    with ExitStack() as stack:
        staged = []
        for destination, value in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            folder = Path(stack.enter_context(TemporaryDirectory(prefix=".pipeline-", dir=destination.parent)))
            replacement, backup = folder / "new", folder / "old"
            if isinstance(value, pa.Table):
                parquet.write_table(value, replacement)
            else:
                replacement.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            existed = destination.exists()
            if existed:
                shutil.copyfile(destination, backup)
            staged.append((destination, replacement, backup, existed))
        committed = []
        try:
            for entry in staged:
                os.replace(entry[1], entry[0])
                committed.append(entry)
        except OSError:
            for destination, _, backup, existed in reversed(committed):
                if existed:
                    os.replace(backup, destination)
                else:
                    destination.unlink(missing_ok=True)
            raise


def _quality_report(manifest: dict, news: pa.Table, prices: pa.Table, report: dict, config: PreparationConfig) -> dict:
    """Reconcile final counts without confusing them with the candidate pool."""
    quality = {key: value for key, value in manifest.items() if key != "prepared_candidates"}
    retained = sorted(report["retained_tickers"])
    missing_publisher = news["has_publisher"].to_pylist().count(False)
    missing_url = news["has_url"].to_pylist().count(False)
    annual = Counter()
    annual_tickers = Counter()
    for ticker in retained:
        years = manifest["coverage_by_ticker"][ticker]["articles_by_year"]
        annual.update({int(year): total for year, total in years.items()})
        annual_tickers.update(int(year) for year in years)
    quality.update({
        "output": str(config.output.resolve()), "retained_tickers": retained, "price_tickers": retained,
        "qualified_but_not_selected": sorted(set(manifest["ticker_ranking"]) - set(retained)),
        "selection": {**manifest["selection"], "max_tickers": config.max_tickers,
                      "sentiment_coverage": asdict(config.sentiment_coverage),
                      "final_rule": "Retain first candidates passing monthly numeric sentiment coverage"},
        "counts": {**manifest["counts"], "price_rows_kept": prices.num_rows, "article_rows_kept": news.num_rows,
                   "article_rows_kept_missing_publisher": missing_publisher, "article_rows_kept_missing_url": missing_url},
        "retained_news_quality": {"article_ticker_records": news.num_rows, "missing_publisher": missing_publisher,
                                  "missing_publisher_fraction": missing_publisher / news.num_rows, "missing_url": missing_url},
        "retained_news_by_year": {
            year: {"article_ticker_records": total, "tickers_with_articles": annual_tickers[year]}
            for year, total in sorted(annual.items())
        },
        "retained_news_by_month": dict(sorted(Counter(day.strftime("%Y-%m") for day in news["article_date_utc"].to_pylist()).items())),
        "sentiment_selection": {key: report[key] for key in (
            "target_tickers", "target_reached", "stop_reason", "evaluated_tickers", "rejected_tickers", "unprocessed_tickers",
        )},
    })
    price_counts = Counter(prices["ticker"].to_pylist())
    quality["price_date_ranges"] = {
        ticker: {**metrics, "exported_rows": price_counts[ticker]}
        for ticker, metrics in manifest["price_date_ranges"].items()
    }
    return quality


def run_sentiment(root: Path, config: PreparationConfig) -> dict:
    """Score ranked candidates until the target qualifies or the pool is exhausted.

    Completed ticker checkpoints survive interruptions. Only matching input,
    model, prompt and identity fingerprints permit reuse. Threshold changes do
    not invalidate inference. Final exports contain all news for selected tickers,
    including abstentions, and exactly the same ticker universe as prices.
    """
    started = monotonic()
    manifest = _load_manifest(root, config)
    securities = load_securities(root)
    ranking = manifest["ticker_ranking"]
    retained, evaluated, selected_tables = [], {}, []
    target_securities = {}
    scorer = None
    counts = Counter()
    inference_seconds = 0.0
    for ticker in ranking:
        if ticker not in securities:
            raise ValueError(
                f"Missing security metadata for {ticker} in data/raw/symbols_valid_meta.csv. "
                "Correct the catalog and rerun sentiment; completed scores are retained."
            )
        target_securities[ticker] = securities[ticker]
        candidate = manifest["prepared_candidates"][ticker]
        folder = root / "data/interim/prepared" / ticker
        for filename, field in (("news.parquet", "news_sha256"), ("prices.parquet", "prices_sha256")):
            if _fingerprint(folder / filename) != candidate[field]:
                raise ValueError(f"Prepared {filename} changed for {ticker}; rebuild candidates before scoring")
        # A versioned recipe captures scoring semantics as well as exact prompt/identity.
        recipe = {**sentiment.scoring_provenance({ticker: target_securities[ticker]}),
                  "scoring_version": 1, "deduplication": "exact title within complete ticker history"}
        recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        checkpoint = root / "data/interim/scores" / f"{ticker}.parquet"
        cached = _checkpoint(checkpoint, candidate["news_sha256"], recipe_hash, ticker, candidate["article_rows"])
        if cached is None:
            source = parquet.read_table(folder / "news.parquet")
            _validate_table(source, EVENT_SCHEMA, ticker, candidate["article_rows"])
            if scorer is None:
                scorer = sentiment.SLMScorer()
            scoring_started = monotonic()
            table, inferences = sentiment.score_articles(source, scorer, target_securities[ticker])
            inference_seconds += monotonic() - scoring_started
            table = table.replace_schema_metadata({
                **(table.schema.metadata or {}), b"candidate_input_sha256": candidate["news_sha256"].encode(),
                b"sentiment_recipe_sha256": recipe_hash.encode(), b"headline_inferences": str(inferences).encode(),
            })
            _write_checkpoint(checkpoint, table)
            counts.update(tickers_scored_this_run=1, article_rows_scored_this_run=table.num_rows,
                          new_headline_inferences=inferences)
        else:
            table, inferences = cached
            counts["tickers_reused_this_run"] += 1
        metrics = sentiment_coverage(table, config)
        evaluated[ticker] = metrics
        counts.update(tickers=1, article_rows=table.num_rows, headline_inferences=inferences,
                      numeric_score_rows=metrics["numeric_score_rows"])
        if metrics["passes"]:
            retained.append(ticker)
            selected_tables.append(table.replace_schema_metadata(None))
        print(f"{ticker}: {'retained' if metrics['passes'] else 'rejected'}; "
              f"{metrics['sufficient_months']}/{metrics['window_months']} months meet depth; "
              f"{len(retained)}/{config.max_tickers} tickers retained", flush=True)
        if len(retained) == config.max_tickers:
            break
    if not retained:
        raise ValueError("No candidates meet sentiment coverage; final files were preserved. Review thresholds or metadata.")

    news = pa.concat_tables(selected_tables).replace_schema_metadata(
        sentiment.scoring_metadata({ticker: target_securities[ticker] for ticker in retained}),
    )
    price_tables = []
    for ticker in retained:
        table = parquet.read_table(root / "data/interim/prepared" / ticker / "prices.parquet")
        _validate_table(table, PRICE_SCHEMA, ticker, manifest["prepared_candidates"][ticker]["price_rows"])
        price_tables.append(table)
    prices = pa.concat_tables(price_tables)
    labels = Counter(news["sentiment_label"].to_pylist())
    numeric = [score for score in news["sentiment_score"].to_pylist() if score is not None]
    report = {
        **sentiment.scoring_provenance(target_securities),
        "input_manifest": str((root / "data/interim/manifest.json").resolve()), "output": str(config.output.resolve()),
        "start_date": config.start_date.isoformat(), "end_date": config.end_date.isoformat(),
        "target_tickers": config.max_tickers, "retained_tickers": retained, "evaluated_tickers": list(evaluated),
        "rejected_tickers": [ticker for ticker, metrics in evaluated.items() if not metrics["passes"]],
        "unprocessed_tickers": ranking[len(evaluated):], "target_reached": len(retained) == config.max_tickers,
        "stop_reason": "target_reached" if len(retained) == config.max_tickers else "candidate_pool_exhausted",
        "coverage_thresholds": asdict(config.sentiment_coverage), "coverage_by_ticker": evaluated,
        "candidate_counts": {key: counts[key] for key in (
            "tickers", "article_rows", "numeric_score_rows", "headline_inferences", "tickers_scored_this_run",
            "tickers_reused_this_run", "article_rows_scored_this_run", "new_headline_inferences",
        )},
        "final_counts": {"tickers": len(retained), "article_rows": news.num_rows, "price_rows": prices.num_rows,
                         "numeric_score_rows": len(numeric)},
        "label_counts": {label: labels[label] for label in sentiment.LABELS},
        "mean_sentiment_score": sum(numeric) / len(numeric) if numeric else None,
        "inference_seconds": round(inference_seconds, 2), "elapsed_seconds": round(monotonic() - started, 2),
        "records_per_second": round(counts["article_rows_scored_this_run"] / inference_seconds, 1) if inference_seconds else None,
    }
    quality = _quality_report(manifest, news, prices, report, config)
    _publish({
        config.output / "daily_prices.parquet": prices, config.output / "news_events.parquet": news,
        root / "reports/quality_report.json": quality, root / "reports/sentiment_report.json": report,
    })
    return report
