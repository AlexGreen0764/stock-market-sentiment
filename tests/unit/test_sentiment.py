"""Verify SLM answers, distinct-headline reuse, and lossless table enrichment."""

import json
import unittest
from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pyarrow as pa

from stock_market_sentiment.prepare import EVENT_SCHEMA
from stock_market_sentiment.sentiment import (
    FEW_SHOT_EXAMPLES,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    READ_BATCH_SIZE,
    SENTIMENT_SCHEMA,
    SLMScorer,
    score_articles,
    scoring_metadata,
    scoring_provenance,
    summarize_label,
)

EXAMPLES = {
    "Earnings increased": "positive",
    "Annual meeting scheduled": "neutral",
    "Profit fell": "negative",
}
SECURITIES = {
    "ABC": {"target_name": "ABC Corporation", "security_type": "stock"},
    "DEF": {"target_name": "DEF Equity Fund", "security_type": "ETF"},
}


def make_fixture(titles: list[str], tickers: list[str] | None = None) -> pa.Table:
    """Build typed rows with unique URLs so prediction reuse cannot remove records."""
    rows = [{
        "ticker": tickers[index] if tickers else "ABC", "title": title,
        "published_at_utc": datetime(2020, 12, 31, 14, tzinfo=UTC),
        "article_date_utc": date(2020, 12, 31), "next_trading_date": date(2021, 1, 4),
        "url": f"https://example.test/{index}", "publisher": "Test",
        "has_url": True, "has_publisher": True,
    } for index, title in enumerate(titles)]
    return pa.Table.from_pylist(rows, schema=EVENT_SCHEMA.with_metadata({b"source": b"fixture"}))


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
            results = scorer.score([
                ("ABC", "ABC Corporation", "stock", "Longer headline"),
                ("DEF", "DEF Equity Fund", "ETF", "Short"),
            ])
        self.assertEqual([row["sentiment_label"] for row in results], ["positive", "negative"])
        for call in scorer.tokenizer.apply_chat_template.call_args_list:
            self.assertIs(call.kwargs["enable_thinking"], False)
            self.assertIs(call.kwargs["add_generation_prompt"], True)
            messages = call.args[0]
            self.assertEqual(
                [message["role"] for message in messages],
                ["system"] + ["user", "assistant"] * len(FEW_SHOT_EXAMPLES) + ["user"],
            )
            for message in messages[1::2]:
                self.assertEqual(set(json.loads(message["content"])), {
                    "target_ticker", "target_name", "security_type", "headline",
                })
            answers = [message["content"] for message in messages[2:-1:2]]
            self.assertEqual(answers, [example[-1] for example in FEW_SHOT_EXAMPLES])
        calls = scorer.tokenizer.apply_chat_template.call_args_list
        self.assertEqual(calls[0].args[0][:-1], calls[1].args[0][:-1])
        self.assertEqual(json.loads(calls[0].args[0][-1]["content"]), {
            "target_ticker": "ABC", "target_name": "ABC Corporation", "security_type": "stock", "headline": "Longer headline",
        })
        self.assertEqual(json.loads(calls[1].args[0][-1]["content"]), {
            "target_ticker": "DEF", "target_name": "DEF Equity Fund", "security_type": "ETF", "headline": "Short",
        })
        self.assertIs(scorer.model.call_args.kwargs["use_cache"], False)
        self.assertEqual(scorer.model.call_args.kwargs["logits_to_keep"], 1)

    def test_over_budget_prompt_rejected_before_model_call(self):
        """Reject the entire batch before padding or inference if any prompt is too long."""
        scorer = SLMScorer.__new__(SLMScorer)
        scorer.tokenizer = MagicMock()
        scorer.tokenizer.apply_chat_template.return_value = "formatted prompt"
        scorer.tokenizer.return_value = {"input_ids": [[11], [12] * (MAX_LENGTH + 1)]}
        scorer.model = MagicMock()
        with (
            patch.dict("sys.modules", {"torch": MagicMock()}),
            self.assertRaisesRegex(ValueError, f"exceeds {MAX_LENGTH} tokens; no headlines were truncated"),
        ):
            scorer.score([
                ("ABC", "ABC Corporation", "stock", "Short headline"),
                ("DEF", "DEF Equity Fund", "ETF", "Over-budget headline"),
            ])
        scorer.tokenizer.pad.assert_not_called()
        scorer.model.assert_not_called()

    def test_enrichment_preserves_rows_types_and_validates_score_semantics(self):
        scorer = MagicMock()
        scorer.score.side_effect = lambda items: [
            {"sentiment_label": EXAMPLES[title], "sentiment_score": 999} for *_, title in items
        ]
        original = make_fixture(list(EXAMPLES))
        scored, inferences = score_articles(original, scorer, SECURITIES["ABC"])
        self.assertEqual(scored.select(EVENT_SCHEMA.names).replace_schema_metadata(original.schema.metadata), original)
        self.assertEqual(scored.column("sentiment_label").to_pylist(), ["positive", "neutral", "negative"])
        self.assertEqual(scored.column("sentiment_score").to_pylist(), [1, 0, -1])
        self.assertEqual(inferences, 3)
        for field in SENTIMENT_SCHEMA:
            self.assertEqual(scored.schema.field(field.name).type, field.type)
        self.assertEqual(scored.schema.metadata[b"source"], b"fixture")
        provenance = json.loads(scored.schema.metadata[b"sentiment_provenance"])
        self.assertEqual(provenance, scoring_provenance({"ABC": SECURITIES["ABC"]}))

    def test_repeated_headlines_are_scored_once_across_chunks(self):
        scorer = MagicMock()
        scorer.score.side_effect = lambda items: [summarize_label(EXAMPLES[title]) for *_, title in items]
        titles = ["Earnings increased", "Annual meeting scheduled", "Earnings increased", "Profit fell", "Earnings increased"]
        with patch("stock_market_sentiment.sentiment.READ_BATCH_SIZE", 2):
            scored, inferences = score_articles(make_fixture(titles), scorer, SECURITIES["ABC"])
        self.assertEqual(inferences, 3)
        self.assertEqual(scorer.score.call_count, 2)
        self.assertEqual([item[-1] for call in scorer.score.call_args_list for item in call.args[0]], list(EXAMPLES))
        self.assertEqual(scored.column("sentiment_score").to_pylist(), [1, 0, 1, -1, 1])
        self.assertEqual(scored.column("url").to_pylist(), [f"https://example.test/{index}" for index in range(5)])

    def test_same_headline_is_scored_for_each_target_identity(self):
        scorer = MagicMock()
        scorer.score.side_effect = lambda items: [
            summarize_label("positive" if ticker == "ABC" else "negative") for ticker, *_ in items
        ]
        headline = "A wins business from B"
        stock, stock_count = score_articles(make_fixture([headline]), scorer, SECURITIES["ABC"])
        fund, fund_count = score_articles(make_fixture([headline], ["DEF"]), scorer, SECURITIES["DEF"])
        self.assertEqual(scorer.score.call_args_list[0].args[0], [("ABC", "ABC Corporation", "stock", headline)])
        self.assertEqual(scorer.score.call_args_list[1].args[0], [("DEF", "DEF Equity Fund", "ETF", headline)])
        self.assertEqual((stock_count, fund_count), (1, 1))
        self.assertEqual(stock.column("sentiment_score").to_pylist(), [1])
        self.assertEqual(fund.column("sentiment_score").to_pylist(), [-1])

    def test_abstentions_remain_rows_with_typed_null_scores(self):
        scorer = MagicMock()
        scorer.score.return_value = [summarize_label("mixed"), summarize_label("insufficient_information")]
        scored, inferences = score_articles(make_fixture(["Mixed evidence", "Roundup"]), scorer, SECURITIES["ABC"])
        self.assertEqual(scored.num_rows, 2)
        self.assertEqual(inferences, 2)
        self.assertEqual(scored.column("sentiment_score").to_pylist(), [None, None])
        self.assertEqual(scored.schema.field("sentiment_score").type, pa.float32())
        self.assertEqual(scored.column("sentiment_label").to_pylist(), ["mixed", "insufficient_information"])

    def test_invalid_inputs_are_rejected_before_inference(self):
        cases = (
            (make_fixture([]), SECURITIES["ABC"], "empty"),
            (make_fixture(["Headline"]).drop(["title"]), SECURITIES["ABC"], "EVENT_SCHEMA"),
            (make_fixture(["Headline", "Other"], ["ABC", "DEF"]), SECURITIES["ABC"], "one nonempty ticker"),
            (make_fixture(["Headline"], [" "]), SECURITIES["ABC"], "one nonempty ticker"),
            (make_fixture(["Headline"], [None]), SECURITIES["ABC"], "one nonempty ticker"),
            (make_fixture([" "]), SECURITIES["ABC"], "headlines must be nonempty"),
            (make_fixture([None]), SECURITIES["ABC"], "headlines must be nonempty"),
            (make_fixture(["Headline"]), {}, "nonempty target_name and security_type"),
            (make_fixture(["Headline"]), {"target_name": "ABC", "security_type": " "}, "nonempty target_name and security_type"),
        )
        for table, security, message in cases:
            scorer = MagicMock()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                score_articles(table, scorer, security)
            scorer.score.assert_not_called()

    def test_invalid_model_answers_are_rejected(self):
        for results, error_type, message in (
            ([], ValueError, "output count"),
            ([{"sentiment_label": "invalid"}], ValueError, "Invalid sentiment label"),
            ([{}], ValueError, "Invalid sentiment label"),
            ([None], TypeError, "must contain sentiment labels"),
        ):
            scorer = MagicMock()
            scorer.score.return_value = results
            with self.subTest(results=results), self.assertRaisesRegex(error_type, message):
                score_articles(make_fixture(["Headline"]), scorer, SECURITIES["ABC"])

    def test_inference_failure_does_not_modify_the_input_table(self):
        scorer = MagicMock()
        scorer.score.side_effect = [[summarize_label("neutral")], RuntimeError("Model failure")]
        original = make_fixture(["First headline", "Second headline"])
        expected = make_fixture(["First headline", "Second headline"])
        with patch("stock_market_sentiment.sentiment.READ_BATCH_SIZE", 1), self.assertRaisesRegex(RuntimeError, "Model failure"):
            score_articles(original, scorer, SECURITIES["ABC"])
        self.assertEqual(original, expected)
        self.assertEqual(original.schema, EVENT_SCHEMA.with_metadata({b"source": b"fixture"}))

    def test_provenance_matches_footer_and_runtime_settings(self):
        provenance = scoring_provenance(SECURITIES)
        metadata = scoring_metadata(SECURITIES)
        self.assertEqual(provenance, json.loads(metadata[b"sentiment_provenance"]))
        self.assertEqual(provenance["model"], MODEL_ID)
        self.assertEqual(provenance["model_revision"], MODEL_REVISION)
        self.assertEqual(provenance["max_tokens"], MAX_LENGTH)
        self.assertEqual(provenance["read_batch_size"], READ_BATCH_SIZE)
        self.assertEqual(provenance["model_input_fields"], ["target_ticker", "target_name", "security_type", "headline"])
        self.assertFalse(provenance["thinking_enabled"])
        self.assertEqual(provenance["model_input_mode"], "text_only")


if __name__ == "__main__":
    unittest.main()
