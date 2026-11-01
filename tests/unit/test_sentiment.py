"""Check sentiment math and lossless Parquet enrichment using small fixtures."""

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
from pyarrow import parquet

from stock_market_sentiment.prepare import EVENT_SCHEMA
from stock_market_sentiment.sentiment import (
    MODEL_ID,
    MODEL_REVISION,
    SENTIMENT_SCHEMA,
    score_news,
    summarize_probabilities,
)

PROBABILITIES = {
    "Earnings increased": [0.05, 0.15, 0.8],
    "Annual meeting scheduled": [0.1, 0.8, 0.1],
    "Profit fell": [0.8, 0.15, 0.05],
}


def write_fixture(path: Path, titles: list[str]) -> None:
    """Write typed news rows whose stable identities expose ordering regressions."""
    rows = [{
        "ticker": "ABC", "title": title,
        "published_at_utc": datetime(2020, 12, 31, 14, tzinfo=timezone.utc),
        "article_date_utc": date(2020, 12, 31), "next_trading_date": date(2021, 1, 4),
        "url": f"https://example.test/{index}", "publisher": "Test",
        "has_url": True, "has_publisher": True,
    } for index, title in enumerate(titles)]
    schema = EVENT_SCHEMA.with_metadata({b"source": b"fixture"})
    parquet.write_table(pa.Table.from_pylist(rows, schema=schema), path)


class SentimentTests(unittest.TestCase):
    """Verify score semantics and file safety without weights, CUDA, or network."""
    def test_signed_score_and_confidence_are_distinct(self):
        positive = summarize_probabilities([0.1, 0.2, 0.7])
        self.assertAlmostEqual(positive["sentiment_score"], 0.6)
        self.assertEqual(positive["sentiment_label"], "positive")
        self.assertEqual(positive["sentiment_confidence"], 0.7)
        neutral = summarize_probabilities([0.05, 0.9, 0.05])
        self.assertEqual(neutral["sentiment_score"], 0)
        self.assertEqual(neutral["sentiment_label"], "neutral")

    def test_invalid_model_outputs_are_rejected(self):
        for probabilities in ([float("nan"), 0, 1], [-0.1, 0.1, 1], [0.1, 0.1, 0.1], [0, 1]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                summarize_probabilities(probabilities)

    @patch("stock_market_sentiment.sentiment.FinSenseScorer")
    def test_enrichment_preserves_dates_rows_metadata_and_supports_reruns(self, model_class):
        scorer = model_class.return_value
        scorer.score.side_effect = lambda titles: [summarize_probabilities(PROBABILITIES[title]) for title in titles]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "news.parquet"
            report_path = Path(temp) / "report.json"
            write_fixture(path, list(PROBABILITIES))
            original = parquet.read_table(path)
            report = score_news(path, report_path)
            scored = parquet.read_table(path)
            # Compare Arrow tables directly to avoid timezone database dependencies
            # when converting UTC timestamps to Python objects on Windows.
            self.assertEqual(scored.select(EVENT_SCHEMA.names).replace_schema_metadata(original.schema.metadata), original)
            self.assertEqual(scored.column("title").to_pylist(), list(PROBABILITIES))
            self.assertEqual(scored.column("sentiment_label").to_pylist(), ["positive", "neutral", "negative"])
            for actual, expected in zip(scored.column("sentiment_score").to_pylist(), [0.75, 0, -0.75]):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(scored.schema.metadata[b"sentiment_model"].decode(), MODEL_ID)
            self.assertEqual(scored.schema.metadata[b"sentiment_model_revision"].decode(), MODEL_REVISION)
            self.assertEqual(report["rows_scored"], 3)
            self.assertEqual(report["label_counts"], {"negative": 1, "neutral": 1, "positive": 1})
            self.assertTrue(report_path.is_file())
            self.assertEqual(scorer.score.call_args_list[0].args, (list(PROBABILITIES),))
            self.assertEqual(report["headline_inferences"], 3)
            score_news(path, report_path)
            rescored = parquet.read_schema(path)
            for field in SENTIMENT_SCHEMA:
                self.assertEqual(rescored.names.count(field.name), 1)
                self.assertEqual(rescored.field(field.name).type, field.type)

    @patch("stock_market_sentiment.sentiment.FinSenseScorer")
    def test_inference_failure_keeps_original_file_and_report(self, model_class):
        scorer = model_class.return_value
        scorer.score.side_effect = [[summarize_probabilities([0.1, 0.8, 0.1])], RuntimeError("Model failure")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "news.parquet"
            report_path = Path(temp) / "report.json"
            # Cross a chunk boundary so failure occurs after a partial output write.
            write_fixture(path, ["Annual meeting scheduled"] * 8193)
            original = path.read_bytes()
            report_path.write_text("previous report", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Model failure"):
                score_news(path, report_path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(report_path.read_text(), "previous report")
            self.assertEqual(sorted(file.name for file in Path(temp).iterdir()), ["news.parquet", "report.json"])

    @patch("stock_market_sentiment.sentiment.FinSenseScorer")
    def test_duplicate_headlines_score_once_without_losing_rows(self, model_class):
        scorer = model_class.return_value
        scorer.score.side_effect = lambda titles: [summarize_probabilities(PROBABILITIES[title]) for title in titles]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "news.parquet"
            titles = ["Profit fell", "Earnings increased", "Profit fell", "Annual meeting scheduled", "Profit fell"]
            write_fixture(path, titles)
            report = score_news(path, Path(temp) / "report.json")
            table = parquet.read_table(path)
            self.assertEqual(table.column("title").to_pylist(), titles)
            self.assertEqual(table.column("sentiment_label").to_pylist(), ["negative", "positive", "negative", "neutral", "negative"])
            self.assertEqual(report["rows_scored"], 5)
            self.assertEqual(report["headline_inferences"], 3)
            self.assertEqual(report["label_counts"], {"negative": 3, "neutral": 1, "positive": 1})

    @patch("stock_market_sentiment.sentiment.FinSenseScorer")
    def test_empty_news_rejected_without_loading_model(self, model_class):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "news.parquet"
            write_fixture(path, [])
            with self.assertRaisesRegex(ValueError, "empty"):
                score_news(path, Path(temp) / "report.json")
            model_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
