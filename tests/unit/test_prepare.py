"""Focused checks for CSV parsing, price validation, and news alignment."""

import csv
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pyarrow import parquet

from stock_market_sentiment.config import CoverageThresholds, load_config
from stock_market_sentiment.prepare import (
    coverage_by_ticker,
    main,
    next_trading_day,
    prepare,
)

SMALL_COVERAGE = CoverageThresholds(span_months=1, covered_month_fraction=1, max_empty_months=0, distinct_publication_dates=1, articles=1)


class PreparationTests(unittest.TestCase):
    def test_package_entry_point_finds_repository_config(self):
        with patch("sys.argv", ["stock-market-sentiment"]), patch(
            "stock_market_sentiment.prepare.prepare", return_value={}
        ) as mocked:
            main()
        root = mocked.call_args.args[0]
        self.assertTrue((root / "config" / "default.json").is_file())

    def test_default_config_uses_reliable_article_window(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / "config" / "default.json", root)
        self.assertEqual(config.start_date, date(2009, 1, 1))
        self.assertEqual(config.end_date, date(2020, 12, 31))
        self.assertEqual(config.coverage, CoverageThresholds(24, 0.75, 3, 60, 100))

    def test_monthly_coverage_requires_continuity_depth_and_distinct_days(self):
        events = []
        for index in range(24):
            year, month = 2009 + index // 12, index % 12 + 1
            for day in range(1, 7):
                events.append({"ticker": "SAME_DAY", "article_date_utc": date(year, month, 1)})
                if index not in (4, 8, 12, 16, 20, 22):
                    events.append({"ticker": "PASS", "article_date_utc": date(year, month, day)})
                if index not in (4, 5, 6, 7):
                    events.append({"ticker": "GAP", "article_date_utc": date(year, month, day)})
                if index % 2 == 0 or index == 23:
                    events.append({"ticker": "SPARSE", "article_date_utc": date(year, month, day)})
            events.append({"ticker": "SHALLOW", "article_date_utc": date(year, month, 1)})
        summary = coverage_by_ticker(events, ["PASS", "SAME_DAY", "SHALLOW", "GAP", "SPARSE", "ABSENT"], CoverageThresholds())
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
            report = prepare(root, output, coverage=CoverageThresholds(1, 1, 0, 2, 2))
            self.assertEqual(report["counts"]["price_rows_invalid"], 1)
            self.assertEqual(report["counts"]["price_rows_duplicate_date"], 1)
            self.assertEqual(report["counts"]["article_rows_kept"], 2)
            self.assertEqual(report["counts"]["article_rows_duplicate"], 1)
            self.assertEqual(report["retained_tickers"], ["ABC"])
            self.assertEqual(report["rejected_tickers"], ["DEF"])
            self.assertEqual(report["candidate_tickers"], ["ABC", "DEF"])
            self.assertEqual(report["price_tickers"], ["ABC", "DEF"])
            self.assertEqual(report["counts"]["price_rows_kept"], 5)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["daily_prices.parquet", "news_events.parquet"])
            self.assertTrue((root / "reports" / "quality_report.json").is_file())
            rows = parquet.read_table(output / "daily_prices.parquet").to_pylist()
            self.assertEqual({row["ticker"] for row in rows}, {"ABC", "DEF"})
            self.assertEqual(rows[2]["date"], date(2023, 12, 26))
            self.assertIsNone(rows[0]["return_pct"])
            self.assertIsNone(rows[0]["volume_vs_prior_20d"])
            self.assertEqual(rows[2]["return_pct"], 10.0)
            self.assertTrue(rows[0]["news_coverage_pass"])
            self.assertEqual(rows[0]["news_coverage_start"], date(2023, 12, 21))
            self.assertFalse(rows[-1]["news_coverage_pass"])
            news = parquet.read_table(output / "news_events.parquet")
            events = news.drop(["published_at_utc"]).to_pylist()
            self.assertEqual(events[0]["title"], "Good, news")
            self.assertEqual(
                news.column("published_at_utc").cast("int64")[0].as_py(),
                int(datetime(2023, 12, 22, 14, tzinfo=timezone.utc).timestamp() * 1000),
            )
            self.assertEqual(events[0]["article_date_utc"], date(2023, 12, 22))
            self.assertEqual(events[0]["next_trading_date"], date(2023, 12, 26))
            news_schema = parquet.read_schema(output / "news_events.parquet")
            self.assertEqual(str(news_schema.field("article_date_utc").type), "date32[day]")
            self.assertEqual(str(news_schema.field("next_trading_date").type), "date32[day]")
            self.assertEqual(str(news_schema.field("published_at_utc").type), "timestamp[ms, tz=UTC]")

            range_output = root / "range-out"
            range_report = prepare(root, range_output, start_date=date(2023, 12, 26), end_date=date(2023, 12, 26), coverage=SMALL_COVERAGE)
            self.assertEqual(range_report["retained_tickers"], [])
            self.assertEqual(parquet.read_table(range_output / "daily_prices.parquet").num_rows, 2)
            self.assertEqual(parquet.read_table(range_output / "news_events.parquet").num_rows, 0)

            boundary_output = root / "boundary-out"
            boundary_report = prepare(root, boundary_output, start_date=date(2023, 12, 22), end_date=date(2023, 12, 22), coverage=SMALL_COVERAGE)
            self.assertEqual(boundary_report["counts"]["article_rows_outside_date_range"], 2)
            boundary_events = parquet.read_table(boundary_output / "news_events.parquet", columns=["title"]).to_pylist()
            self.assertEqual(len(boundary_events), 2)
            self.assertEqual(boundary_events[0]["title"], "Good, news")
            boundary_prices = parquet.read_table(boundary_output / "daily_prices.parquet").to_pylist()
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

    def test_year_end_links_missing_metadata_and_pre_history_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "data" / "raw" / "full_history"
            folder.mkdir(parents=True)
            (folder / "NEW.csv").write_text(
                "date,open,high,low,close,adj close,volume\n"
                "2020-12-30,10,11,9,10,10,100\n"
                "2020-12-31,11,12,10,11,11,200\n"
                "2021-01-04,12,13,11,12,12,300\n", encoding="utf-8",
            )
            with (folder.parent / "nasdaq_exteral_data.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(["Date", "Article_title", "Stock_symbol", "Url", "Publisher"])
                writer.writerow(["2020-12-31 14:00:00 UTC", "Year end", " new ", "", ""])
                writer.writerow(["2020-12-31 14:00:00 UTC", "Year end", "NEW", "", ""])
                writer.writerow(["2020-12-31 15:00:00 UTC", "", "NEW", "u1", "Test"])
                writer.writerow(["2020-12-29 14:00:00 UTC", "Before prices", "NEW", "u2", "Test"])
                writer.writerow(["2020-12-31 14:00:00 UTC", "Unmatched", "UNKNOWN", "u3", "Test"])
            output = root / "out"
            report = prepare(root, output, start_date=date(2020, 1, 1), end_date=date(2020, 12, 31), coverage=SMALL_COVERAGE)
            news = parquet.read_table(output / "news_events.parquet")
            events = news.drop(["published_at_utc"]).to_pylist()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["next_trading_date"], date(2021, 1, 4))
            self.assertFalse(events[0]["has_url"])
            self.assertFalse(events[0]["has_publisher"])
            self.assertEqual(report["counts"]["article_rows_duplicate"], 1)
            self.assertEqual(report["counts"]["article_rows_missing_title"], 1)
            self.assertEqual(report["counts"]["article_rows_unusable_price_match"], 1)
            self.assertEqual(report["counts"]["article_rows_unmatched_ticker"], 1)
            self.assertEqual(report["counts"]["price_rows_kept"], 3)
            self.assertEqual(report["article_filter_counts_by_ticker"]["NEW"]["matched_articles"], 4)
            self.assertEqual(json.loads((root / "reports" / "quality_report.json").read_text())["retained_tickers"], ["NEW"])

    def test_invalid_monthly_thresholds_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            for overrides in (
                {"span_months": 0}, {"articles": True}, {"distinct_publication_dates": 1.5},
                {"max_empty_months": -1}, {"covered_month_fraction": 0}, {"covered_month_fraction": 1.1},
            ):
                with self.subTest(overrides=overrides):
                    config_path.write_text(json.dumps({"coverage": overrides}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(config_path, root)


if __name__ == "__main__":
    unittest.main()
