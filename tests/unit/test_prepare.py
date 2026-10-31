"""Focused checks for CSV parsing, price validation, and news alignment."""

import csv
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

SMALL_COVERAGE = CoverageThresholds(1, 1, 1, 1)


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
        self.assertEqual(config.coverage, CoverageThresholds(24, 8, 3, 60))

    def test_quarterly_coverage_requires_breadth_depth_and_distinct_days(self):
        events = []
        for year in range(2009, 2017):
            for month in (1, 4, 7):
                for day in (1, 2, 3):
                    events.append({"ticker": "PASS", "article_date_utc": date(year, month, day)})
                    events.append({"ticker": "SAME_DAY", "article_date_utc": date(year, month, 1)})
                events.append({"ticker": "SHALLOW", "article_date_utc": date(year, month, 1)})
        summary = coverage_by_ticker(events, ["PASS", "SAME_DAY", "SHALLOW", "ABSENT"], CoverageThresholds())
        self.assertTrue(summary["PASS"]["passes"])
        self.assertEqual(summary["PASS"]["covered_quarters"], 24)
        self.assertEqual(summary["PASS"]["distinct_publication_dates"], 72)
        self.assertFalse(summary["SAME_DAY"]["passes"])
        self.assertFalse(summary["SHALLOW"]["passes"])
        self.assertFalse(summary["ABSENT"]["passes"])

    def test_next_session_skips_same_day_and_weekend(self):
        days = [date(2023, 12, 22), date(2023, 12, 26), date(2023, 12, 27)]
        self.assertEqual(next_trading_day(date(2023, 12, 22), days), date(2023, 12, 26))
        self.assertEqual(next_trading_day(date(2023, 12, 24), days), date(2023, 12, 26))
        self.assertEqual(next_trading_day(date(2023, 12, 25), days), date(2023, 12, 26))
        self.assertIsNone(next_trading_day(date(2023, 12, 27), days))

    def test_pipeline_reads_reordered_prices_and_multiline_news(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data" / "raw" / "full_history").mkdir(parents=True)
            (root / "data" / "raw" / "full_history" / "ABC.csv").write_text(
                "date,volume,open,high,low,close,adj close\n"
                "2023-12-26,200,11,12,10,11,11\n"
                "2023-12-22,100,10,11,9,10,10\n"
                "2023-12-22,100,10,11,9,10,10\n"
                "2023-12-21,-1,10,11,9,10,10\n", encoding="utf-8"
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
            report = prepare(root, output, ["ABC", "DEF"], coverage=CoverageThresholds(1, 1, 2, 2))
            self.assertEqual(report["counts"]["price_rows_invalid"], 1)
            self.assertEqual(report["counts"]["price_rows_duplicate_date"], 1)
            self.assertEqual(report["counts"]["article_rows_kept"], 2)
            self.assertEqual(report["counts"]["article_rows_duplicate"], 1)
            self.assertEqual(report["retained_tickers"], ["ABC"])
            self.assertEqual(report["rejected_tickers"], ["DEF"])
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["daily_prices.parquet", "news_events.parquet"])
            self.assertTrue((root / "reports" / "quality_report.json").is_file())
            rows = parquet.read_table(output / "daily_prices.parquet").to_pylist()
            self.assertEqual({row["ticker"] for row in rows}, {"ABC"})
            self.assertEqual(rows[1]["date"], date(2023, 12, 26))
            self.assertIsNone(rows[0]["return_pct"])
            self.assertIsNone(rows[0]["volume_vs_prior_20d"])
            self.assertEqual(rows[1]["return_pct"], 10.0)
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
            with self.assertRaisesRegex(ValueError, "No ticker meets"):
                prepare(root, range_output, ["ABC"], start_date=date(2023, 12, 26), end_date=date(2023, 12, 26), coverage=SMALL_COVERAGE)

            boundary_output = root / "boundary-out"
            boundary_report = prepare(root, boundary_output, ["ABC"], start_date=date(2023, 12, 22), end_date=date(2023, 12, 26), coverage=SMALL_COVERAGE)
            self.assertEqual(boundary_report["counts"]["article_rows_outside_date_range"], 1)
            boundary_events = parquet.read_table(boundary_output / "news_events.parquet", columns=["title"]).to_pylist()
            self.assertEqual(len(boundary_events), 1)
            self.assertEqual(boundary_events[0]["title"], "Good, news")

    def test_config_file_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(
                '{"tickers": ["AAPL"], "start_date": "2020-01-01", '
                '"end_date": "2020-12-31", '
                '"coverage": {"covered_quarters": 12}}',
                encoding="utf-8",
            )
            config = load_config(config_path, root, {"tickers": ["MSFT"], "end_date": "2021-01-01"})
            self.assertEqual(config.tickers, ("MSFT",))
            self.assertEqual(config.start_date, date(2020, 1, 1))
            self.assertEqual(config.end_date, date(2021, 1, 1))
            self.assertEqual(config.coverage.covered_quarters, 12)


if __name__ == "__main__":
    unittest.main()
