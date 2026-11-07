"""Focused checks for CSV parsing, price validation, and news alignment."""

import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from pyarrow import parquet

from stock_market_sentiment.config import (
    CoverageThresholds,
    PreparationConfig,
    SentimentCoverageThresholds,
    load_config,
    parse_config_args,
)
from stock_market_sentiment.prepare import (
    coverage_by_ticker,
    load_prices,
    main,
    next_trading_day,
    prepare,
)

# Tiny fixtures exercise candidate preparation without changing production defaults.
SMALL_COVERAGE = CoverageThresholds(span_months=1, covered_month_fraction=1, max_empty_months=0, distinct_publication_dates=1, articles=1)


class PreparationTests(unittest.TestCase):
    """Exercise cleaning and coverage rules using isolated temporary inputs."""
    def test_package_entry_point_finds_repository_config(self):
        with patch("sys.argv", ["stock-market-sentiment"]), patch(
            "stock_market_sentiment.prepare.prepare", return_value={"ticker_ranking": []}
        ) as mocked:
            main()
        root = mocked.call_args.args[0]
        self.assertTrue((root / "config" / "default.json").is_file())

    def test_default_config_uses_reliable_article_window(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / "config" / "default.json", root)
        self.assertEqual(config.start_date, date(2015, 1, 1))
        self.assertEqual(config.end_date, date(2020, 12, 31))
        self.assertEqual(config.max_tickers, 30)
        self.assertEqual(config.coverage, CoverageThresholds(24, 0.75, 3, 60, 100))
        self.assertEqual(config.sentiment_coverage, SentimentCoverageThresholds(10, 0.8))

    def test_monthly_coverage_requires_continuity_depth_and_distinct_days(self):
        events = []
        for index in range(24):
            year, month = 2009 + index // 12, index % 12 + 1
            for day in range(1, 7):
                events.append({"ticker": "SAME_DAY", "article_date_utc": date(year, month, 1)})
                # PASS sits exactly on 75%; GAP isolates the continuity failure.
                if index not in (4, 8, 12, 16, 20, 22):
                    events.append({"ticker": "PASS", "article_date_utc": date(year, month, day)})
                if index not in (4, 5, 6, 7):
                    events.append({"ticker": "GAP", "article_date_utc": date(year, month, day)})
                if index % 2 == 0 or index == 23:
                    events.append({"ticker": "SPARSE", "article_date_utc": date(year, month, day)})
            events.append({"ticker": "SHALLOW", "article_date_utc": date(year, month, 1)})
        summary = coverage_by_ticker(events, ["PASS", "SAME_DAY", "SHALLOW", "GAP", "SPARSE", "ABSENT"], CoverageThresholds(), date(2009, 1, 1), date(2010, 12, 31))
        self.assertTrue(summary["PASS"]["passes"])
        self.assertEqual(summary["PASS"]["span_months"], 24)
        self.assertEqual(summary["PASS"]["covered_months"], 18)
        self.assertEqual(summary["PASS"]["covered_month_fraction"], 0.75)
        self.assertEqual(summary["PASS"]["distinct_publication_dates"], 108)
        self.assertEqual(summary["PASS"]["articles_by_year"], {2009: 60, 2010: 48})
        self.assertFalse(summary["SAME_DAY"]["passes"])
        self.assertFalse(summary["SHALLOW"]["passes"])
        self.assertEqual(summary["GAP"]["failed_criteria"], ["max_empty_months"])
        self.assertIn("covered_month_fraction", summary["SPARSE"]["failed_criteria"])
        self.assertFalse(summary["ABSENT"]["passes"])
        self.assertEqual(summary["ABSENT"]["span_months"], 0)
        self.assertIsNone(summary["ABSENT"]["coverage_start"])

    def test_next_session_skips_same_day_and_weekend(self):
        days = [date(2023, 12, 22), date(2023, 12, 26), date(2023, 12, 27)]
        self.assertEqual(next_trading_day(date(2023, 12, 22), days), date(2023, 12, 26))
        self.assertEqual(next_trading_day(date(2023, 12, 24), days), date(2023, 12, 26))
        self.assertEqual(next_trading_day(date(2023, 12, 25), days), date(2023, 12, 26))
        self.assertIsNone(next_trading_day(date(2023, 12, 27), days))
        self.assertIsNone(next_trading_day(date(2023, 12, 21), days))
        self.assertIsNone(next_trading_day(date(2023, 12, 22), [date(2023, 12, 22), date(2024, 1, 2)]))
        self.assertIsNone(next_trading_day(date(2023, 12, 22), []))

    def test_coverage_includes_empty_window_edges(self):
        events = [{"ticker": "ABC", "article_date_utc": date(2020, month, 1)} for month in (5, 6)]
        result = coverage_by_ticker(events, ["ABC"], SMALL_COVERAGE, date(2020, 1, 1), date(2020, 6, 30))["ABC"]
        self.assertEqual(result["window_months"], 6)
        self.assertEqual(result["covered_month_fraction"], 2 / 6)
        self.assertEqual(result["max_empty_months"], 4)
        self.assertIn("covered_month_fraction", result["failed_criteria"])
        self.assertIn("max_empty_months", result["failed_criteria"])

    def test_all_eligible_candidates_ranked_without_replacing_final_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "data/raw/full_history"
            folder.mkdir(parents=True)
            for ticker in ("ABC", "DEF", "XYZ"):
                (folder / f"{ticker}.csv").write_text(
                    "date,open,high,low,close,adj close,volume\n"
                    "2023-12-20,10,11,9,10,10,100\n"
                    "2023-12-21,10,11,9,10,10,100\n"
                    "2023-12-22,10,11,9,10,10,100\n"
                    "2023-12-26,10,11,9,10,10,100\n", encoding="utf-8",
                )
            with (folder.parent / "nasdaq_exteral_data.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"])
                for ticker in ("ABC", "XYZ"):
                    for index, day in enumerate((21, 22, 24)):
                        body = ("First paragraph.\n\nSecond paragraph.", "", "NULL")[index]
                        writer.writerow([f"2023-12-{day} 14:00:00 UTC", f"Item {index}", ticker, f"{ticker}/{index}", "Test", body])
                for index in range(5):
                    writer.writerow(["2023-12-22 14:00:00 UTC", f"DEF {index}", "DEF", f"DEF/{index}", "Test", "Body"])
                # Broad articles remain eligible, including URLs tagged outside the
                # requested window and for tickers without price files.
                for ticker in ("ABC", "OUT1", "OUT2", "OUT3", "OUT4", "OUT5", "OUT5"):
                    published = "2023-12-22" if ticker == "ABC" else "2000-01-01"
                    writer.writerow([f"{published} 14:00:00 UTC", "Broad", ticker, "shared", "Test", "Broad body"])
                # Repeated tags for unmatched tickers have no effect on selection.
                for ticker in ("ABC", "OUT1", "OUT2", "OUT3", "OUT4", "OUT4"):
                    published = "2023-12-22" if ticker == "ABC" else "2000-01-01"
                    writer.writerow([f"{published} 14:00:00 UTC", "At cap", ticker, "at-cap", "Test", "Preserved"])
            output = root / "out"
            output.mkdir()
            reports = root / "reports"
            reports.mkdir()
            preserved = [output / "news_events.parquet", output / "daily_prices.parquet",
                         reports / "quality_report.json", reports / "sentiment_report.json"]
            for path in preserved:
                path.write_bytes(b"Existing final result")
            config = PreparationConfig(date(2023, 12, 1), date(2023, 12, 31), output, SMALL_COVERAGE, 1)
            with patch("stock_market_sentiment.prepare.load_prices", wraps=load_prices) as loader:
                report = prepare(root, config)
            self.assertEqual(loader.call_count, 3)
            self.assertNotIn("retained_tickers", report)
            self.assertEqual(report["ticker_ranking"], ["ABC", "XYZ", "DEF"])
            self.assertEqual(set(report["prepared_candidates"]), {"ABC", "DEF", "XYZ"})
            self.assertNotIn("max_tickers", report["selection"])
            self.assertEqual(report["prepared_counts"], {"news_rows": 13, "price_rows": 12})
            for ticker, candidate in report["prepared_candidates"].items():
                for name, fingerprint in (("news", "news_sha256"), ("prices", "prices_sha256")):
                    path = root / "data/interim/prepared" / ticker / f"{name}.parquet"
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), candidate[fingerprint])
            for path in preserved:
                self.assertEqual(path.read_bytes(), b"Existing final result")
            manifest = json.loads((root / "data/interim/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["ticker_ranking"], ["ABC", "XYZ", "DEF"])
            news = parquet.read_table(root / "data/interim/prepared/ABC/news.parquet", columns=["title"]).to_pylist()
            self.assertEqual([row["title"] for row in news], ["Item 0", "Item 1", "Item 2", "Broad", "At cap"])
            prices = parquet.read_table(root / "data/interim/prepared/ABC/prices.parquet", columns=["ticker"]).to_pylist()
            self.assertEqual({row["ticker"] for row in prices}, {"ABC"})

    def test_pipeline_reads_reordered_prices_and_multiline_news(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data" / "raw" / "full_history").mkdir(parents=True)
            (root / "data" / "raw" / "full_history" / "ABC.csv").write_text(
                "date,volume,open,high,low,close,adj close\n"
                "2023-12-26,200,11,12,10,11,11\n"
                "2023-12-22,100,10,11,9,10,10\n"
                "2023-12-22,100,10,11,9,10,10\n"
                "2023-12-21,90,9,10,8,9,9\n"
                "2023-12-20,-1,10,11,9,10,10\n", encoding="utf-8"
            )
            (root / "data" / "raw" / "full_history" / "DEF.csv").write_text(
                "date,open,high,low,close,adj close,volume\n"
                "2023-12-22,10,11,9,10,10,100\n"
                "2023-12-26,11,12,10,11,11,200\n", encoding="utf-8"
            )
            with (root / "data" / "raw" / "nasdaq_exteral_data.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"])
                writer.writerow(["2023-12-22 14:00:00 UTC", "Good, news", "ABC", "https://example.test/1", "Test", "Two\nlines"])
                writer.writerow(["2023-12-22 14:00:00 UTC", "Good, news", "ABC", "https://example.test/1", "Test", "Duplicate"])
                writer.writerow(["2023-12-26 14:00:00 UTC", "Late", "ABC", "https://example.test/2", "Test", "Outside range"])
                writer.writerow(["2023-12-21 14:00:00 UTC", "Earlier", "ABC", "https://example.test/3", "Test", "Prior publication date"])
                writer.writerow(["2023-12-22 14:00:00 UTC", "Sparse", "DEF", "https://example.test/4", "Test", "One article"])
            output = root / "out"
            config = PreparationConfig(date(2023, 12, 1), date(2023, 12, 31), output,
                                       CoverageThresholds(1, 1, 0, 2, 2), 30)
            report = prepare(root, config)
            self.assertEqual(report["counts"]["price_rows_invalid"], 1)
            self.assertEqual(report["counts"]["price_rows_duplicate_date"], 1)
            self.assertNotIn("article_rows_kept", report["counts"])
            self.assertEqual(report["prepared_counts"]["news_rows"], 2)
            self.assertEqual(report["counts"]["article_rows_duplicate"], 1)
            self.assertEqual(report["ticker_ranking"], ["ABC"])
            self.assertEqual(report["rejected_tickers"], ["DEF"])
            self.assertEqual(report["candidate_tickers"], ["ABC", "DEF"])
            self.assertNotIn("price_tickers", report)
            self.assertNotIn("price_rows_kept", report["counts"])
            self.assertEqual(report["prepared_counts"]["price_rows"], 3)
            self.assertFalse(output.exists())
            self.assertFalse((root / "reports").exists())
            candidate = root / "data/interim/prepared/ABC"
            rows = parquet.read_table(candidate / "prices.parquet").to_pylist()
            self.assertEqual({row["ticker"] for row in rows}, {"ABC"})
            self.assertEqual(rows[2]["date"], date(2023, 12, 26))
            self.assertIsNone(rows[0]["return_pct"])
            self.assertIsNone(rows[0]["volume_vs_prior_20d"])
            self.assertEqual(rows[2]["return_pct"], 10.0)
            self.assertTrue(rows[0]["news_coverage_pass"])
            self.assertEqual(rows[0]["news_coverage_start"], date(2023, 12, 21))
            self.assertTrue(rows[-1]["news_coverage_pass"])
            news = parquet.read_table(candidate / "news.parquet")
            events = news.drop(["published_at_utc"]).to_pylist()
            self.assertEqual(events[0]["title"], "Good, news")
            self.assertEqual(
                news.column("published_at_utc").cast("int64")[0].as_py(),
                int(datetime(2023, 12, 22, 14, tzinfo=UTC).timestamp() * 1000),
            )
            self.assertEqual(events[0]["article_date_utc"], date(2023, 12, 22))
            self.assertEqual(events[0]["next_trading_date"], date(2023, 12, 26))
            news_schema = parquet.read_schema(candidate / "news.parquet")
            self.assertEqual(str(news_schema.field("article_date_utc").type), "date32[day]")
            self.assertEqual(str(news_schema.field("next_trading_date").type), "date32[day]")
            self.assertEqual(str(news_schema.field("published_at_utc").type), "timestamp[ms, tz=UTC]")

            range_output = root / "range-out"
            range_report = prepare(root, replace(config, output=range_output, start_date=date(2023, 12, 26),
                                                  end_date=date(2023, 12, 26), coverage=SMALL_COVERAGE))
            self.assertEqual(range_report["ticker_ranking"], [])
            self.assertEqual(range_report["prepared_candidates"], {})
            self.assertEqual(range_report["prepared_counts"], {"news_rows": 0, "price_rows": 0})
            self.assertFalse(range_output.exists())

            boundary_output = root / "boundary-out"
            boundary_report = prepare(root, replace(config, output=boundary_output, start_date=date(2023, 12, 22),
                                                     end_date=date(2023, 12, 22), coverage=SMALL_COVERAGE))
            self.assertEqual(boundary_report["counts"]["article_rows_outside_date_range"], 2)
            boundary_events = parquet.read_table(candidate / "news.parquet", columns=["title"]).to_pylist()
            self.assertEqual(boundary_report["prepared_counts"]["news_rows"], 2)
            self.assertEqual(len(boundary_events), 1)
            self.assertEqual(boundary_events[0]["title"], "Good, news")
            boundary_prices = parquet.read_table(candidate / "prices.parquet").to_pylist()
            self.assertEqual({row["date"] for row in boundary_prices}, {date(2023, 12, 22), date(2023, 12, 26)})

    def test_config_file_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(
                '{"start_date": "2020-01-01", '
                '"end_date": "2020-12-31", '
                '"coverage": {"covered_month_fraction": 0.9, "max_empty_months": 0}}',
                encoding="utf-8",
            )
            config = load_config(config_path, root, {"end_date": "2021-01-01"})
            self.assertEqual(config.start_date, date(2020, 1, 1))
            self.assertEqual(config.end_date, date(2021, 1, 1))
            self.assertEqual(config.coverage.covered_month_fraction, 0.9)
            self.assertEqual(config.coverage.max_empty_months, 0)
            self.assertEqual(config.sentiment_coverage, SentimentCoverageThresholds())

            with patch("sys.argv", ["prepare", "--root", str(root), "--config", str(config_path),
                                    "--output", "custom", "--start-date", "2020-02-01"]):
                parsed_root, parsed = parse_config_args("Fixture")
            self.assertEqual(parsed_root, root.resolve())
            self.assertEqual(parsed.output, root / "custom")
            self.assertEqual(parsed.start_date, date(2020, 2, 1))

    def test_candidate_write_failure_preserves_manifest_and_final_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "data/raw"
            histories = raw / "full_history"
            histories.mkdir(parents=True)
            (histories / "ABC.csv").write_text(
                "date,open,high,low,close,adj close,volume\n"
                "2020-12-30,10,11,9,10,10,100\n"
                "2020-12-31,10,11,9,10,10,100\n", encoding="utf-8",
            )
            (raw / "nasdaq_exteral_data.csv").write_text(
                "Date,Article_title,Stock_symbol,Url,Publisher\n"
                "2020-12-30 14:00:00 UTC,Headline,ABC,url,Publisher\n", encoding="utf-8",
            )
            preserved = [root / "data/interim/manifest.json", root / "out/news_events.parquet",
                         root / "out/daily_prices.parquet", root / "reports/quality_report.json"]
            for path in preserved:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"Previous successful run")
            with (
                patch("stock_market_sentiment.prepare.parquet.write_table", side_effect=OSError("Disk failure")),
                self.assertRaisesRegex(OSError, "Disk failure"),
            ):
                prepare(root, PreparationConfig(date(2020, 12, 1), date(2020, 12, 31),
                                                root / "out", SMALL_COVERAGE, 30))
            for path in preserved:
                self.assertEqual(path.read_bytes(), b"Previous successful run")
            self.assertEqual(list((root / "data/interim").rglob("*.partial")), [])

    def test_year_end_links_missing_metadata_and_pre_history_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "data" / "raw" / "full_history"
            folder.mkdir(parents=True)
            # Exercise mixed filename casing and integral decimal volumes together.
            (folder / "nEw.csv").write_text(
                "date,open,high,low,close,adj close,volume\n"
                "2020-12-30,10,11,9,10,10,100\n"
                "2020-12-31,11,12,10,11,11,200.0\n"
                "2021-01-04,12,13,11,12,12,3e2\n", encoding="utf-8",
            )
            with (folder.parent / "nasdaq_exteral_data.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"])
                writer.writerow(["2020-12-31 14:00:00 UTC", "Year end", " new ", "", "", ""])
                writer.writerow(["2020-12-31 14:00:00 UTC", "Year end", "NEW", "", "", ""])
                writer.writerow(["2020-12-31 15:00:00 UTC", "", "NEW", "u1", "Test", ""])
                writer.writerow(["2020-12-29 14:00:00 UTC", "Before prices", "NEW", "u2", "Test", ""])
                writer.writerow(["2020-12-31 14:00:00 UTC", "Unmatched", "UNKNOWN", "u3", "Test", ""])
            output = root / "out"
            report = prepare(root, PreparationConfig(date(2020, 12, 1), date(2020, 12, 31),
                                                     output, SMALL_COVERAGE, 30))
            candidate = root / "data/interim/prepared/NEW"
            news = parquet.read_table(candidate / "news.parquet")
            events = news.drop(["published_at_utc"]).to_pylist()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["next_trading_date"], date(2021, 1, 4))
            self.assertFalse(events[0]["has_url"])
            self.assertFalse(events[0]["has_publisher"])
            self.assertEqual(report["candidate_tickers"], ["NEW"])
            self.assertEqual(report["ticker_ranking"], ["NEW"])
            self.assertEqual(report["counts"]["article_rows_duplicate"], 1)
            self.assertEqual(report["counts"]["article_rows_missing_title"], 1)
            self.assertEqual(report["counts"]["article_rows_unusable_price_match"], 1)
            self.assertEqual(report["counts"]["article_rows_unmatched_ticker"], 1)
            self.assertEqual(report["prepared_counts"]["price_rows"], 3)
            self.assertNotIn("retained_news_quality", report)
            self.assertNotIn("retained_news_by_year", report)
            self.assertEqual(report["price_validation_by_ticker"]["NEW"]["price_rows_invalid"], 0)
            prices = parquet.read_table(candidate / "prices.parquet").to_pylist()
            self.assertEqual({row["ticker"] for row in prices}, {"NEW"})
            self.assertEqual([row["volume"] for row in prices], [100, 200, 300])
            self.assertEqual(report["article_filter_counts_by_ticker"]["NEW"]["matched_articles"], 4)
            self.assertEqual(json.loads((root / "data/interim/manifest.json").read_text())["ticker_ranking"], ["NEW"])

    def test_price_validation_accepts_integral_decimals_and_explains_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "mixed.csv"
            cases = [
                ("2020-01-01", "10", "11", "9", "100.0"),
                ("2020-01-02", "10", "11", "9", "3e2"),
                ("2020-01-03", "10", "11", "9", "1.5"),
                ("2020-01-04", "10", "11", "9", "-1"),
                ("2020-01-05", "10", "11", "9", "NaN"),
                ("2020-01-06", "10", "11", "9", "Infinity"),
                ("2020-01-07", "10", "11", "9", "9223372036854775808"),
                ("2020-01-08", "NaN", "11", "9", "100"),
                ("2020-01-09", "0", "11", "9", "100"),
                ("2020-01-10", "10", "9", "9", "100"),
                ("bad-date", "10", "11", "9", "100"),
                ("2020-01-12", "10", "11", "9", "bad-volume"),
                # Above 2**53: a float intermediate would lose the final share.
                ("2020-01-13", "10", "11", "9", "9007199254740993.0"),
            ]
            with source.open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(["date", "open", "high", "low", "close", "adj close", "volume"])
                for day, opening, high, low, volume in cases:
                    writer.writerow([day, opening, high, low, "10", "10", volume])
            counts = Counter()
            examples = {}
            prices = load_prices(source, counts, examples)
            self.assertEqual([row["volume"] for row in prices], [100, 300, 9007199254740993])
            self.assertEqual({row["ticker"] for row in prices}, {"MIXED"})
            self.assertEqual(counts["price_rows_read"], 13)
            self.assertEqual(counts["price_rows_valid"], 3)
            self.assertEqual(counts["price_rows_invalid"], 10)
            self.assertEqual(counts["price_rows_invalid_volume"], 5)
            self.assertEqual(counts["price_rows_invalid_parse"], 2)
            self.assertEqual(counts["price_rows_invalid_nonfinite"], 1)
            self.assertEqual(counts["price_rows_invalid_nonpositive_price"], 1)
            self.assertEqual(counts["price_rows_invalid_ohlc_bounds"], 1)
            self.assertEqual(len(examples["volume"]), 3)
            self.assertEqual(examples["volume"][0]["source_file"], "mixed.csv")

    def test_invalid_monthly_thresholds_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            for overrides in (
                {"span_months": 0}, {"articles": True}, {"distinct_publication_dates": 1.5},
                {"max_empty_months": -1}, {"covered_month_fraction": 0}, {"covered_month_fraction": 1.1},
            ):
                with self.subTest(overrides=overrides):
                    config_path.write_text(json.dumps({"start_date": "2015-01-01", "end_date": "2020-12-31", "coverage": overrides}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(config_path, root)

    def test_invalid_selection_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            for value in (0, -1, True, 1.5):
                with self.subTest(value=value):
                    config_path.write_text(json.dumps({
                        "start_date": "2015-01-01", "end_date": "2020-12-31", "max_tickers": value,
                    }), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(config_path, root)

    def test_invalid_sentiment_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            for raw in (
                {"articles_per_month": 0}, {"articles_per_month": True}, {"articles_per_month": 1.5},
                {"covered_month_fraction": 0}, {"covered_month_fraction": 1.1},
                {"covered_month_fraction": True}, {"unknown": 1},
            ):
                with self.subTest(raw=raw):
                    config_path.write_text(json.dumps({
                        "start_date": "2015-01-01", "end_date": "2020-12-31", "sentiment_coverage": raw,
                    }), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(config_path, root)


if __name__ == "__main__":
    unittest.main()
