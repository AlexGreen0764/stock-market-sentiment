"""Exercise adaptive ticker selection with isolated inputs and a model double."""

import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
from pyarrow import parquet

from stock_market_sentiment import sentiment
from stock_market_sentiment.config import (
    CoverageThresholds,
    PreparationConfig,
    SentimentCoverageThresholds,
)
from stock_market_sentiment.pipeline import (
    load_securities,
    run_sentiment,
    sentiment_coverage,
)
from stock_market_sentiment.prepare import EVENT_SCHEMA, prepare

RAW_COVERAGE = CoverageThresholds(1, 1, 0, 1, 1)
PUBLICATION_DATES = ("2020-11-02", "2020-11-03", "2020-12-01", "2020-12-02", "2020-12-31")


class FakeScorer:
    """Record complete inference inputs and return deterministic semantic labels."""

    def __init__(self, labels=None, failure_ticker=None):
        self.labels = labels or {}
        self.failure_ticker = failure_ticker
        self.items = []

    def score(self, items):
        self.items.extend(items)
        if any(item[0] == self.failure_ticker for item in items):
            raise RuntimeError("Injected inference failure")
        return [sentiment.summarize_label(self.labels.get(ticker, ["positive"] * 5)[int(title.rsplit(" ", 1)[1])])
                for ticker, _name, _security_type, title in items]


def make_fixture(root, tickers=("AAA", "BBB", "CCC", "DDD"), target=2):
    """Prepare equal-depth histories so the fixed raw ranking is alphabetical."""
    prices = root / "data/raw/full_history"
    prices.mkdir(parents=True)
    for ticker in tickers:
        with (prices / f"{ticker}.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["date", "open", "high", "low", "close", "adj close", "volume"])
            for day in (*PUBLICATION_DATES[:-1], "2020-11-04", "2020-12-03", "2021-01-04"):
                writer.writerow([day, 10, 11, 9, 10, 10, 100])
    with (prices.parent / "nasdaq_exteral_data.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["Date", "Article_title", "Stock_symbol", "Url", "Publisher"])
        for ticker in tickers:
            for index, day in enumerate(PUBLICATION_DATES):
                writer.writerow([f"{day} 14:00:00 UTC", f"{ticker} news {index}", ticker,
                                 f"https://example.test/{ticker}/{index}", "Test"])
    with (prices.parent / "symbols_valid_meta.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["Symbol", "Security Name", "ETF"])
        writer.writerows((ticker, f"{ticker} Corporation", "N") for ticker in tickers)
    config = PreparationConfig(date(2020, 11, 1), date(2020, 12, 31), root / "data/processed",
                               RAW_COVERAGE, target, SentimentCoverageThresholds(1, 1))
    prepare(root, config)
    return config


def run_with_double(root, config, labels=None, failure_ticker=None):
    """Keep test output concise while checking the real orchestration and I/O."""
    scorer = FakeScorer(labels, failure_ticker)
    with patch("stock_market_sentiment.sentiment.SLMScorer", return_value=scorer) as constructor, redirect_stdout(io.StringIO()):
        report = run_sentiment(root, config)
    return report, constructor, scorer


def preserve_final_results(root, config):
    """Seed byte-identifiable prior results for failure-safety assertions."""
    paths = [config.output / "news_events.parquet", config.output / "daily_prices.parquet",
             root / "reports/quality_report.json", root / "reports/sentiment_report.json"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Prior completed result")
    return paths


class AdaptivePipelineTests(unittest.TestCase):
    """Verify stopping, checkpoint reuse, coverage, and safe publication."""

    def test_replaces_rejected_ticker_and_stops_after_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            labels = {"AAA": ["insufficient_information"] * 5,
                      "BBB": ["negative", "neutral", "positive", "mixed", "insufficient_information"]}
            report, constructor, scorer = run_with_double(root, config, labels)
            self.assertEqual(report["retained_tickers"], ["BBB", "CCC"])
            self.assertTrue(report["target_reached"])
            self.assertEqual(report["target_tickers"], 2)
            self.assertEqual([item[0] for item in scorer.items], ["AAA"] * 5 + ["BBB"] * 5 + ["CCC"] * 5)
            constructor.assert_called_once()
            self.assertFalse((root / "data/interim/scores/DDD.parquet").exists())
            news = parquet.read_table(config.output / "news_events.parquet")
            prices = parquet.read_table(config.output / "daily_prices.parquet")
            self.assertEqual(set(news.column("ticker").to_pylist()), {"BBB", "CCC"})
            self.assertEqual(set(prices.column("ticker").to_pylist()), {"BBB", "CCC"})
            self.assertEqual(news.num_rows, 10)
            self.assertEqual(news.column("sentiment_score").null_count, 2)
            self.assertEqual(report["final_counts"]["numeric_score_rows"], 8)
            self.assertEqual(report["label_counts"]["mixed"], 1)
            for field in ("article_date_utc", "next_trading_date"):
                self.assertEqual(news.schema.field(field).type, pa.date32())
            self.assertEqual(news.schema.field("published_at_utc").type, pa.timestamp("ms", tz="UTC"))
            self.assertIn(date(2021, 1, 4), prices.column("date").to_pylist())
            quality = json.loads((root / "reports/quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["counts"]["article_rows_kept"], news.num_rows)
            self.assertEqual(set(quality["retained_tickers"]), {"BBB", "CCC"})

    def test_cached_rerun_can_change_selection_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            run_with_double(root, config)
            narrower = replace(config, max_tickers=1, sentiment_coverage=SentimentCoverageThresholds(2, 1))
            report, constructor, scorer = run_with_double(root, narrower)
            constructor.assert_not_called()
            self.assertEqual(scorer.items, [])
            self.assertEqual(report["retained_tickers"], ["AAA"])
            expanded = replace(config, max_tickers=3)
            report, constructor, scorer = run_with_double(root, expanded)
            constructor.assert_called_once()
            self.assertEqual([item[0] for item in scorer.items], ["CCC"] * 5)
            self.assertEqual(report["retained_tickers"], ["AAA", "BBB", "CCC"])

    def test_only_changed_candidate_identity_is_rescored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            run_with_double(root, config)
            path = root / "data/raw/symbols_valid_meta.csv"
            path.write_text(path.read_text(encoding="utf-8").replace("BBB Corporation", "BBB Historical Company"),
                            encoding="utf-8")
            _report, constructor, scorer = run_with_double(root, config)
            constructor.assert_called_once()
            self.assertEqual([item[0] for item in scorer.items], ["BBB"] * 5)
            self.assertEqual({item[1] for item in scorer.items}, {"BBB Historical Company"})

    def test_new_input_and_prompt_invalidate_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            run_with_double(root, config)
            raw = root / "data/raw/nasdaq_exteral_data.csv"
            raw.write_text(raw.read_text(encoding="utf-8").replace("AAA news 0", "AAA revised news 0"), encoding="utf-8")
            prepare(root, config)
            _report, _constructor, scorer = run_with_double(root, config)
            self.assertEqual([item[0] for item in scorer.items], ["AAA"] * 5)
            with patch.object(sentiment, "SYSTEM_PROMPT", sentiment.SYSTEM_PROMPT + "\nA revised rule."):
                _report, _constructor, scorer = run_with_double(root, config)
            self.assertEqual([item[0] for item in scorer.items], ["AAA"] * 5 + ["BBB"] * 5)

    def test_pool_exhaustion_publishes_fewer_qualified_tickers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root, ("AAA", "BBB", "CCC"), target=3)
            report, _constructor, _scorer = run_with_double(root, config, {
                "AAA": ["mixed"] * 5, "CCC": ["insufficient_information"] * 5,
            })
            self.assertEqual(report["retained_tickers"], ["BBB"])
            self.assertFalse(report["target_reached"])
            self.assertEqual(report["stop_reason"], "candidate_pool_exhausted")
            self.assertEqual(set(parquet.read_table(config.output / "news_events.parquet").column("ticker").to_pylist()), {"BBB"})

    def test_zero_qualifiers_preserve_existing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root, ("AAA",))
            preserved = preserve_final_results(root, config)
            with self.assertRaises(ValueError):
                run_with_double(root, config, {"AAA": ["insufficient_information"] * 5})
            for path in preserved:
                self.assertEqual(path.read_bytes(), b"Prior completed result")
            self.assertTrue((root / "data/interim/scores/AAA.parquet").is_file())

    def test_inference_failure_preserves_results_and_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            preserved = preserve_final_results(root, config)
            with self.assertRaisesRegex(RuntimeError, "Injected inference failure"):
                run_with_double(root, config, failure_ticker="BBB")
            for path in preserved:
                self.assertEqual(path.read_bytes(), b"Prior completed result")
            self.assertTrue((root / "data/interim/scores/AAA.parquet").is_file())
            self.assertFalse((root / "data/interim/scores/BBB.parquet").exists())

    def test_missing_target_identity_is_explicit_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            path = root / "data/raw/symbols_valid_meta.csv"
            original_catalog = path.read_text(encoding="utf-8")
            path.write_text(original_catalog.replace("BBB,BBB Corporation,N\n", ""), encoding="utf-8")
            preserved = preserve_final_results(root, config)
            with self.assertRaisesRegex(ValueError, "BBB"):
                run_with_double(root, config)
            for output in preserved:
                self.assertEqual(output.read_bytes(), b"Prior completed result")
            self.assertTrue((root / "data/interim/scores/AAA.parquet").is_file())
            path.write_text(original_catalog, encoding="utf-8")
            report, _constructor, scorer = run_with_double(root, config)
            self.assertEqual(report["retained_tickers"], ["AAA", "BBB"])
            self.assertEqual([item[0] for item in scorer.items], ["BBB"] * 5)

    def test_changed_preparation_settings_require_rebuild_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            changed = replace(config, start_date=date(2020, 10, 1))
            with (
                patch("stock_market_sentiment.sentiment.SLMScorer") as constructor,
                self.assertRaisesRegex(ValueError, "rebuild candidates"),
            ):
                run_sentiment(root, changed)
            constructor.assert_not_called()

    def test_changed_prepared_file_cannot_reuse_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            run_with_double(root, config)
            preserved = preserve_final_results(root, config)
            (root / "data/interim/prepared/AAA/news.parquet").write_bytes(b"Changed candidate")
            with (
                patch("stock_market_sentiment.sentiment.SLMScorer") as constructor,
                self.assertRaisesRegex(ValueError, "changed for AAA"),
            ):
                run_sentiment(root, config)
            constructor.assert_not_called()
            for path in preserved:
                self.assertEqual(path.read_bytes(), b"Prior completed result")

    def test_corrupt_matching_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            run_with_double(root, config)
            preserved = preserve_final_results(root, config)
            path = root / "data/interim/scores/AAA.parquet"
            table = parquet.read_table(path)
            field = sentiment.SENTIMENT_SCHEMA.field("sentiment_score")
            invalid = table.set_column(table.schema.get_field_index(field.name), field,
                                       pa.array([2.0] * table.num_rows, type=field.type))
            parquet.write_table(invalid, path)
            with (
                patch("stock_market_sentiment.sentiment.SLMScorer") as constructor,
                self.assertRaisesRegex(ValueError, "Invalid cached sentiment score"),
            ):
                run_sentiment(root, config)
            constructor.assert_not_called()
            for output in preserved:
                self.assertEqual(output.read_bytes(), b"Prior completed result")

    def test_export_failure_rolls_back_both_tables_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_fixture(root)
            preserved = preserve_final_results(root, config)
            real_replace = os.replace

            def fail_report_replacement(source, destination):
                if Path(destination) == root / "reports/quality_report.json":
                    raise OSError("Injected export failure")
                return real_replace(source, destination)

            with (
                patch("stock_market_sentiment.pipeline.os.replace", side_effect=fail_report_replacement),
                self.assertRaisesRegex(OSError, "Injected export failure"),
            ):
                run_with_double(root, config)
            for path in preserved:
                self.assertEqual(path.read_bytes(), b"Prior completed result")
            self.assertEqual(list(root.rglob(".pipeline-*")), [])
            self.assertEqual(list(root.rglob("*.partial")), [])

    def test_coverage_counts_publication_months_and_empty_window_edges(self):
        config = PreparationConfig(date(2020, 10, 1), date(2020, 12, 31), Path("unused"),
                                   RAW_COVERAGE, 1, SentimentCoverageThresholds(1, 0.6))
        rows = []
        for day, label in ((date(2020, 11, 2), "neutral"), (date(2020, 11, 3), "mixed"),
                           (date(2020, 12, 31), "positive")):
            rows.append({"ticker": "AAA", "published_at_utc": datetime(day.year, day.month, day.day, tzinfo=UTC),
                         "article_date_utc": day, "next_trading_date": date(2021, 1, 4), "title": label,
                         "url": "", "publisher": "", "has_url": False, "has_publisher": False,
                         **sentiment.summarize_label(label)})
        table = pa.Table.from_pylist(rows, schema=pa.schema([*EVENT_SCHEMA, *sentiment.SENTIMENT_SCHEMA]))
        metrics = sentiment_coverage(table, config)
        self.assertEqual(metrics["numeric_scores_by_month"], {"2020-10": 0, "2020-11": 1, "2020-12": 1})
        self.assertEqual(metrics["numeric_score_rows"], 2)
        self.assertEqual(metrics["directional_score_rows"], 1)
        self.assertEqual(metrics["sufficient_months"], 2)
        self.assertEqual(metrics["sufficient_month_fraction"], 2 / 3)
        self.assertEqual(metrics["minimum_monthly_scores"], 0)
        self.assertEqual(metrics["median_monthly_scores"], 1)
        self.assertEqual(metrics["max_empty_months"], 1)
        self.assertTrue(metrics["passes"])

    def test_catalog_supplies_normalized_names_and_security_types(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root, ("AAA",))
            catalog = root / "data/raw/symbols_valid_meta.csv"
            catalog.write_text("Symbol,Security Name,ETF\n aaa , Catalog AAA ,N\nBBB,Broad Market Fund,Y\nCCC,CCC Company,N\n", encoding="utf-8")
            identities = load_securities(root)
            self.assertEqual(identities["AAA"], {"target_name": "Catalog AAA", "security_type": "stock"})
            self.assertEqual(identities["BBB"], {"target_name": "Broad Market Fund", "security_type": "ETF"})
            self.assertEqual(identities["CCC"]["security_type"], "stock")


if __name__ == "__main__":
    unittest.main()
