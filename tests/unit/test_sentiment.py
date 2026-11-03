"""Verify SLM label semantics, ticker-specific caching, and lossless file enrichment."""

import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
from pyarrow import parquet

from stock_market_sentiment.prepare import EVENT_SCHEMA
from stock_market_sentiment.sentiment import (
    LABELS,
    MODEL_ID,
    MODEL_REVISION,
    READ_BATCH_SIZE,
    SENTIMENT_SCHEMA,
    SLMScorer,
    score_news,
    summarize_label,
)

EXAMPLES = {
    "Earnings increased": "positive",
    "Annual meeting scheduled": "neutral",
    "Profit fell": "negative",
}


def write_fixture(path: Path, titles: list[str], tickers: list[str] | None = None) -> None:
    """Write typed rows with unique URLs so caching cannot silently remove records."""
    rows = [{
        "ticker": tickers[index] if tickers else "ABC", "title": title,
        "published_at_utc": datetime(2020, 12, 31, 14, tzinfo=UTC),
        "article_date_utc": date(2020, 12, 31), "next_trading_date": date(2021, 1, 4),
        "url": f"https://example.test/{index}", "publisher": "Test",
        "has_url": True, "has_publisher": True,
    } for index, title in enumerate(titles)]
    parquet.write_table(pa.Table.from_pylist(rows, schema=EVENT_SCHEMA.with_metadata({b"source": b"fixture"})), path)


class SentimentTests(unittest.TestCase):
    """Use model doubles: no weights, network, full dataset, or GPU."""

    def test_abstentions_are_not_neutral_or_numeric(self):
        for label in ("mixed", "insufficient_information"):
            self.assertIsNone(summarize_label(label)["sentiment_score"])
        self.assertEqual(summarize_label("neutral")["sentiment_score"], 0)
        self.assertEqual(summarize_label("negative")["sentiment_score"], -1)
        self.assertEqual(summarize_label("positive")["sentiment_score"], 1)
        with self.assertRaises(ValueError):
            summarize_label("invalid")

    def test_scoring_disables_thinking_and_restores_input_order(self):
        """Exercise the real scoring method with a tokenizer/model double."""
        scorer = SLMScorer.__new__(SLMScorer)
        scorer.tokenizer = MagicMock()
        scorer.tokenizer.apply_chat_template.return_value = "formatted prompt"
        scorer.tokenizer.return_value = {"input_ids": [[11, 12, 13], [14]], "attention_mask": [[1, 1, 1], [1]]}
        scorer.tokenizer.pad.return_value.to.return_value = {"input_ids": "test inputs"}
        scorer.choice_ids = [32, 33, 34, 35, 36]
        scorer.model = MagicMock()
        logits = scorer.model.return_value.logits.__getitem__.return_value
        # Sorted order is second input first. Restore positive, negative for caller.
        logits.argmax.return_value.cpu.return_value.tolist.return_value = [0, 2]
        torch_double = MagicMock()
        torch_double.isfinite.return_value.all.return_value = True
        with patch.dict("sys.modules", {"torch": torch_double}):
            results = scorer.score([("ABC", "Longer headline"), ("DEF", "Short")])
        self.assertEqual([row["sentiment_label"] for row in results], ["positive", "negative"])
        for call in scorer.tokenizer.apply_chat_template.call_args_list:
            self.assertIs(call.kwargs["enable_thinking"], False)
            self.assertIs(call.kwargs["add_generation_prompt"], True)
        self.assertIs(scorer.model.call_args.kwargs["use_cache"], False)
        self.assertEqual(scorer.model.call_args.kwargs["logits_to_keep"], 1)

    @patch("stock_market_sentiment.sentiment.SLMScorer")
    def test_enrichment_preserves_rows_dates_and_replaces_legacy_scores(self, model_class):
        scorer = model_class.return_value
        scorer.score.side_effect = lambda items: [summarize_label(EXAMPLES[title]) for _, title in items]
        with tempfile.TemporaryDirectory() as temp:
            path, report_path = Path(temp) / "news.parquet", Path(temp) / "report.json"
            write_fixture(path, list(EXAMPLES))
            original = parquet.read_table(path)
            legacy = original.append_column("sentiment_confidence", pa.array([0.99] * 3))
            for name in ("sentiment_negative", "sentiment_neutral", "sentiment_positive"):
                legacy = legacy.append_column(name, pa.array([0.1] * 3))
            parquet.write_table(legacy, path)
            report = score_news(path, report_path)
            scored = parquet.read_table(path)
            self.assertEqual(scored.select(EVENT_SCHEMA.names).replace_schema_metadata(original.schema.metadata), original)
            self.assertEqual(scored.column("sentiment_label").to_pylist(), ["positive", "neutral", "negative"])
            self.assertEqual(scored.column("sentiment_score").to_pylist(), [1, 0, -1])
            self.assertNotIn("sentiment_confidence", scored.schema.names)
            self.assertNotIn("sentiment_positive", scored.schema.names)
            self.assertEqual(scored.schema.metadata[b"sentiment_model"].decode(), MODEL_ID)
            self.assertEqual(scored.schema.metadata[b"sentiment_model_revision"].decode(), MODEL_REVISION)
            self.assertEqual(scored.schema.metadata[b"sentiment_thinking_enabled"], b"false")
            self.assertFalse(report["thinking_enabled"])
            self.assertEqual(report["model_input_mode"], "text_only")
            self.assertEqual(report["rows_scored"], 3)
            self.assertEqual(report["numeric_score_rows"], 3)
            self.assertEqual(report["mean_sentiment_score"], 0)
            self.assertEqual(report["headline_inferences"], 3)
            self.assertEqual(sum(report["label_counts"].values()), 3)
            score_news(path, report_path)
            for field in SENTIMENT_SCHEMA:
                self.assertEqual(parquet.read_schema(path).names.count(field.name), 1)

    @patch("stock_market_sentiment.sentiment.SLMScorer")
    def test_same_title_for_different_tickers_is_scored_separately(self, model_class):
        scorer = model_class.return_value
        scorer.score.side_effect = lambda items: [
            summarize_label("positive" if ticker == "ABC" else "negative") for ticker, _ in items
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "news.parquet"
            write_fixture(path, ["A wins business from B"] * 3, ["ABC", "DEF", "ABC"])
            report = score_news(path, Path(temp) / "report.json")
            self.assertEqual(scorer.score.call_args.args[0], [("ABC", "A wins business from B"), ("DEF", "A wins business from B")])
            self.assertEqual(report["headline_inferences"], 2)
            self.assertEqual(parquet.read_table(path).column("sentiment_score").to_pylist(), [1, -1, 1])

    @patch("stock_market_sentiment.sentiment.SLMScorer")
    def test_all_abstentions_have_null_mean(self, model_class):
        model_class.return_value.score.return_value = [
            summarize_label("mixed"), summarize_label("insufficient_information"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "news.parquet"
            write_fixture(path, ["Mixed evidence", "Roundup"])
            report = score_news(path, Path(temp) / "report.json")
            self.assertEqual(report["numeric_score_rows"], 0)
            self.assertIsNone(report["mean_sentiment_score"])
            self.assertEqual(parquet.read_table(path).column("sentiment_score").to_pylist(), [None, None])
            self.assertEqual(set(report["label_counts"]), set(LABELS))

    @patch("stock_market_sentiment.sentiment.SLMScorer")
    def test_inference_failure_keeps_original_file_and_report(self, model_class):
        model_class.return_value.score.side_effect = [
            [summarize_label("neutral")], RuntimeError("Model failure"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path, report_path = Path(temp) / "news.parquet", Path(temp) / "report.json"
            write_fixture(path, ["Annual meeting scheduled"] * (READ_BATCH_SIZE + 1))
            original = path.read_bytes()
            report_path.write_text("previous report", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Model failure"):
                score_news(path, report_path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(report_path.read_text(), "previous report")
            self.assertEqual(sorted(file.name for file in Path(temp).iterdir()), ["news.parquet", "report.json"])

    @patch("stock_market_sentiment.sentiment.SLMScorer")
    def test_empty_news_rejected_before_model_loading(self, model_class):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "news.parquet"
            write_fixture(path, [])
            with self.assertRaisesRegex(ValueError, "empty"):
                score_news(path, Path(temp) / "report.json")
            model_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
