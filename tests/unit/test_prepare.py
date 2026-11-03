"""Focused checks for CSV parsing, price validation, and news alignment."""

import csv
import json
import tempfile
import unittest
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from pyarrow import parquet

from stock_market_sentiment.config import CoverageThresholds, load_config
from stock_market_sentiment.prepare import (
    coverage_by_ticker,
    load_prices,
    main,
    next_trading_day,
    prepare,
)

# Tiny fixtures exercise export behavior without relaxing production defaults.
SMALL_COVERAGE = CoverageThresholds(span_months=1, covered_month_fraction=1, max_empty_months=0, distinct_publication_dates=1, articles=1)


class PreparationTests(unittest.TestCase):
    """Exercise cleaning and coverage rules using isolated temporary inputs."""
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
        self.assertEqual(config.start_date, date(2015, 1, 1))
        self.assertEqual(config.end_date, date(2020, 12, 31))
        self.assertEqual(config.max_tickers, 30)
        self.assertEqual(config.coverage, CoverageThresholds(24, 0.75, 3, 60, 100))

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

    def test_broad_articles_retained_and_top_ticker_selected(self):
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
            report = prepare(root, root / "out", date(2023, 12, 1), date(2023, 12, 31), SMALL_COVERAGE, max_tickers=1)
            self.assertEqual(report["retained_tickers"], ["ABC"])
            self.assertEqual(report["ticker_ranking"], ["ABC", "XYZ", "DEF"])
            self.assertEqual(report["qualified_but_not_selected"], ["DEF", "XYZ"])
            news = parquet.read_table(root / "out/news_events.parquet", columns=["title"]).to_pylist()
            self.assertEqual([row["title"] for row in news], ["Item 0", "Item 1", "Item 2", "Broad", "At cap"])
            prices = parquet.read_table(root / "out/daily_prices.parquet", columns=["ticker"]).to_pylist()
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
            report = prepare(root, output, date(2023, 12, 1), date(2023, 12, 31), CoverageThresholds(1, 1, 0, 2, 2))
            self.assertEqual(report["counts"]["price_rows_invalid"], 1)
            self.assertEqual(report["counts"]["price_rows_duplicate_date"], 1)
            self.assertEqual(report["counts"]["article_rows_kept"], 2)
            self.assertEqual(report["counts"]["article_rows_duplicate"], 1)
            self.assertEqual(report["retained_tickers"], ["ABC"])
            self.assertEqual(report["rejected_tickers"], ["DEF"])
            self.assertEqual(report["candidate_tickers"], ["ABC", "DEF"])
            self.assertEqual(report["price_tickers"], ["ABC"])
            self.assertEqual(report["counts"]["price_rows_kept"], 3)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["daily_prices.parquet", "news_events.parquet"])
            self.assertTrue((root / "reports" / "quality_report.json").is_file())
            rows = parquet.read_table(output / "daily_prices.parquet").to_pylist()
            self.assertEqual({row["ticker"] for row in rows}, {"ABC"})
            self.assertEqual(rows[2]["date"], date(2023, 12, 26))
            self.assertIsNone(rows[0]["return_pct"])
            self.assertIsNone(rows[0]["volume_vs_prior_20d"])
            self.assertEqual(rows[2]["return_pct"], 10.0)
            self.assertTrue(rows[0]["news_coverage_pass"])
            self.assertEqual(rows[0]["news_coverage_start"], date(2023, 12, 21))
            self.assertTrue(rows[-1]["news_coverage_pass"])
            news = parquet.read_table(output / "news_events.parquet")
            events = news.drop(["published_at_utc"]).to_pylist()
            self.assertEqual(events[0]["title"], "Good, news")
            self.assertEqual(
                news.column("published_at_utc").cast("int64")[0].as_py(),
                int(datetime(2023, 12, 22, 14, tzinfo=UTC).timestamp() * 1000),
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
            self.assertEqual(parquet.read_table(range_output / "daily_prices.parquet").num_rows, 0)
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
            report = prepare(root, output, start_date=date(2020, 12, 1), end_date=date(2020, 12, 31), coverage=SMALL_COVERAGE)
            news = parquet.read_table(output / "news_events.parquet")
            events = news.drop(["published_at_utc"]).to_pylist()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["next_trading_date"], date(2021, 1, 4))
            self.assertFalse(events[0]["has_url"])
            self.assertFalse(events[0]["has_publisher"])
            self.assertEqual(report["candidate_tickers"], ["NEW"])
            self.assertEqual(report["retained_tickers"], ["NEW"])
            self.assertEqual(report["counts"]["article_rows_duplicate"], 1)
            self.assertEqual(report["counts"]["article_rows_missing_title"], 1)
            self.assertEqual(report["counts"]["article_rows_unusable_price_match"], 1)
            self.assertEqual(report["counts"]["article_rows_unmatched_ticker"], 1)
            self.assertEqual(report["counts"]["price_rows_kept"], 3)
            self.assertEqual(report["retained_news_quality"], {
                "article_ticker_records": 1, "missing_publisher": 1,
                "missing_publisher_fraction": 1, "missing_url": 1,
            })
            self.assertEqual(report["retained_news_by_year"], {2020: {
                "article_ticker_records": 1, "tickers_with_articles": 1,
            }})
            self.assertEqual(report["price_validation_by_ticker"]["NEW"]["price_rows_invalid"], 0)
            prices = parquet.read_table(output / "daily_prices.parquet").to_pylist()
            self.assertEqual({row["ticker"] for row in prices}, {"NEW"})
            self.assertEqual([row["volume"] for row in prices], [100, 200, 300])
            self.assertEqual(report["article_filter_counts_by_ticker"]["NEW"]["matched_articles"], 4)
            self.assertEqual(json.loads((root / "reports" / "quality_report.json").read_text())["retained_tickers"], ["NEW"])

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
            for name in ("max_tickers",):
                for value in (0, -1, True, 1.5):
                    with self.subTest(name=name, value=value):
                        config_path.write_text(json.dumps({
                            "start_date": "2015-01-01", "end_date": "2020-12-31", name: value,
                        }), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            load_config(config_path, root)


if __name__ == "__main__":
    unittest.main()
