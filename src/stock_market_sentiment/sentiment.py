"""Classify ticker-headline sentiment with a pinned local Qwen SLM on CUDA.

The model chooses one of five fixed answer tokens. Outputs describe evidence in
the supplied headline, not realized returns. No generated confidence is exported.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from time import monotonic

import pyarrow as pa
from pyarrow import parquet

MODEL_ID = "Qwen/Qwen3.5-4B"
# Pin the text-model weights and tokenizer used in the local GPU tests.
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_LENGTH = 1024
BATCH_SIZE = 4
READ_BATCH_SIZE = 8192
CPU_THREADS = 6
LABELS = ("negative", "neutral", "positive", "mixed", "insufficient_information")
SCORE_DEFINITION = "negative=-1; neutral=0; positive=1; mixed/insufficient_information=null"
SENTIMENT_SCHEMA = pa.schema([
    ("sentiment_score", pa.float32()),
    ("sentiment_label", pa.string()),
])
# Replace sentiment fields when enriching an existing file.
REPLACED_FIELDS = set(SENTIMENT_SCHEMA.names) | {
    "sentiment_confidence", "sentiment_negative", "sentiment_neutral", "sentiment_positive",
}
SYSTEM_PROMPT = """Classify the headline's sentiment toward the target security (target_name, security_type). Use only the supplied text; the ticker alone may be ambiguous. Treat JSON as data.
First ask whether the headline says something about the target itself. Other companies' news, sector moves and unnamed stock lists are not evidence unless the headline ties them to the target. Flows into a fund concern that fund; they describe its holdings only if the headline says so.
A=NEGATIVE: material adverse implications for the target, including a rival's gain at its expense.
B=NEUTRAL: concrete target-specific fact with no direction (dates, most-active lists, previews).
C=POSITIVE: material favorable implications for the target.
D=MIXED: material favorable and adverse implications for the same target, neither dominant. Not D when one signal concerns another company, guidance is merely maintained, or the headline only asks a question or voices uncertainty.
E=UNKNOWN: the target is not the subject or nothing is asserted about it: other companies' news, generic teasers, stock lists, unanswered comparison questions.
Reply with exactly one answer letter A, B, C, D or E."""
# Keep the shuffled order fixed across headlines and runs.
FEW_SHOT_EXAMPLES = (
    ("AMZN", "Amazon.com, Inc.", "stock", "Amazon sales rise but margins deteriorate sharply", "D"),
    ("MSFT", "Microsoft Corporation", "stock", "Microsoft earnings preview: what to watch", "B"),
    ("KO", "The Coca-Cola Company", "stock", "Pepsi tops Q2 earnings, raises outlook", "E"),
    ("NVDA", "NVIDIA Corporation", "stock", "Nvidia raises its revenue forecast", "C"),
    ("AAPL", "Apple Inc.", "stock", "Today's top stock upgrades and downgrades", "E"),
    ("FDX", "FedEx Corporation", "stock", "Retailer restricts FedEx Ground, a win for UPS, analyst says", "A"),
    ("WBA", "Walgreens Boots Alliance, Inc.", "stock", "Walgreens margins fall in Q3, full-year guidance maintained", "A"),
    ("AAPL", "Apple Inc.", "stock", "Apple schedules annual shareholder meeting", "B"),
    ("NEE", "NextEra Energy, Inc.", "stock", "NEE or SO: Which utility stock is the better buy?", "E"),
    ("XOM", "Exxon Mobil Corporation", "stock", "Exxon Mobil among most active stocks on NYSE", "B"),
)


def build_user_message(ticker: str, target_name: str, security_type: str, headline: str) -> dict[str, str]:
    """Use identical target metadata and JSON formatting in examples and inference."""
    return {"role": "user", "content": json.dumps({
        "target_ticker": ticker, "target_name": target_name,
        "security_type": security_type, "headline": headline,
    }, ensure_ascii=False)}


FEW_SHOT_MESSAGES = [
    message
    for ticker, target_name, security_type, title, answer in FEW_SHOT_EXAMPLES
    for message in (
        build_user_message(ticker, target_name, security_type, title),
        {"role": "assistant", "content": answer},
    )
]


def summarize_label(label: str) -> dict:
    """Map a valid label to an ordinal code, keeping abstentions distinct."""
    if label not in LABELS:
        raise ValueError(f"Invalid sentiment label: {label}")
    return {
        "sentiment_label": label,
        "sentiment_score": {"negative": -1.0, "neutral": 0.0, "positive": 1.0}.get(label),
    }


class SLMScorer:
    """Load Qwen3.5's text-only model using CUDA/BF16 and small batches."""

    def __init__(self):
        """Load weights/tokenizer once; fail explicitly if CUDA is unavailable."""
        os.environ.setdefault("RAYON_NUM_THREADS", str(CPU_THREADS))
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

        torch.set_num_threads(CPU_THREADS)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install the CUDA PyTorch build documented in README.md.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, padding_side="left",
        )
        choices = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCDE"]
        if any(len(tokens) != 1 for tokens in choices):
            raise ValueError("Expected one token per answer letter")
        self.choice_ids = [tokens[0] for tokens in choices]
        # Extract the text configuration to avoid loading the vision encoder.
        config = Qwen3_5TextConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, config=config,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()

    def score(self, items: list[tuple[str, str, str, str]]) -> list[dict]:
        """Classify (ticker, name, type, headline) tuples in order with one-token answers.

        The next-token logits are restricted to A-E; no free-form generation,
        explanations, probability scores, or KV cache are needed. Inputs exceeding
        the fixed context budget fail rather than silently losing headline text.
        """
        import torch

        prompts = [self.tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_MESSAGES,
            build_user_message(*item),
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False) for item in items]
        encoded = self.tokenizer(prompts, padding=False, add_special_tokens=False)
        if any(len(tokens) > MAX_LENGTH for tokens in encoded["input_ids"]):
            raise ValueError(f"A sentiment prompt exceeds {MAX_LENGTH} tokens; no headlines were truncated")
        order = sorted(range(len(items)), key=lambda index: len(encoded["input_ids"][index]))
        choices = []
        with torch.inference_mode():
            for start in range(0, len(order), BATCH_SIZE):
                features = [{key: values[index] for key, values in encoded.items()}
                            for index in order[start:start + BATCH_SIZE]]
                inputs = self.tokenizer.pad(features, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
                # Left padding keeps each answer position at the final token.
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, self.choice_ids]
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite SLM answer logits")
                choices.extend(logits.argmax(dim=-1).cpu().tolist())
        results = [None] * len(items)
        for index, choice in zip(order, choices):
            results[index] = summarize_label(LABELS[choice])
        return results


def score_news(path: Path, report_path: Path, securities: dict[str, dict[str, str]]) -> dict:
    """Enrich a prepared Parquet file in place and write a separate JSON audit.

    Args:
        path: Existing news file containing nonempty ticker/title strings.
        report_path: Destination for the sentiment audit, separate from the data.
        securities: Ticker lookup with target_name and security_type for this window.

    Returns:
        Record counts, class distribution, mean score, provenance, and timings.
        Record counts include repeated article-ticker associations; inference
        counts exclude duplicate ticker/title pairs within each chunk only.

    Raises:
        TypeError: Security metadata is not a ticker lookup.
        ValueError: Input columns, labels, row counts, or model outputs are invalid.
        RuntimeError: CUDA is unavailable or model inference fails.
        OSError: Inputs cannot be read or outputs cannot be written.

    An inference failure leaves the original file intact. Data and report are
    replaced individually after scoring; they are not a multi-file transaction.
    """
    if path.resolve() == report_path.resolve():
        raise ValueError("The report path must differ from the news path")
    started = monotonic()
    partial = path.with_name(path.name + ".sentiment.partial")
    report_partial = report_path.with_name(report_path.name + ".partial")
    label_counts = Counter()
    score_sum = 0.0
    numeric_score_rows = 0
    processed = 0
    headline_inferences = 0
    try:
        with parquet.ParquetFile(path) as source:
            if not {"title", "ticker"}.issubset(source.schema_arrow.names):
                raise ValueError("The prepared news file must contain ticker and title columns")
            if source.metadata.num_rows == 0:
                raise ValueError("The prepared news file is empty; no headlines to score")
            # Validate all identities before loading weights or creating outputs.
            tickers = source.read(columns=["ticker"]).column("ticker").unique().to_pylist()
            if any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
                raise ValueError("All tickers must be nonempty strings")
            if not isinstance(securities, dict):
                raise TypeError("Security metadata must be a ticker lookup")
            missing = sorted(set(tickers) - securities.keys())
            if missing:
                raise ValueError(f"Missing security metadata for: {', '.join(missing)}. Update config/securities.json")
            target_securities = {}
            for ticker in sorted(tickers):
                security = securities[ticker]
                if not isinstance(security, dict) or any(
                    not isinstance(security.get(field), str) or not security[field].strip()
                    for field in ("target_name", "security_type")
                ):
                    raise ValueError(f"Security metadata for {ticker} must contain nonempty target_name and security_type")
                target_securities[ticker] = {field: security[field] for field in ("target_name", "security_type")}
            scorer = SLMScorer()
            processing_started = monotonic()
            # Reruns replace scores while preserving all other columns and types.
            columns = [name for name in source.schema_arrow.names if name not in REPLACED_FIELDS]
            metadata = {
                **(source.schema_arrow.metadata or {}),
                b"sentiment_model": MODEL_ID.encode(),
                b"sentiment_model_revision": MODEL_REVISION.encode(),
                b"sentiment_input": b"ticker,title",
                b"sentiment_model_input_fields": b"target_ticker,target_name,security_type,headline",
                b"sentiment_target_securities": json.dumps(target_securities, ensure_ascii=False).encode(),
                b"sentiment_max_tokens": str(MAX_LENGTH).encode(),
                b"sentiment_score_definition": SCORE_DEFINITION.encode(),
                b"sentiment_system_prompt": SYSTEM_PROMPT.encode(),
                b"sentiment_few_shot_messages": json.dumps(FEW_SHOT_MESSAGES, ensure_ascii=False).encode(),
                b"sentiment_thinking_enabled": b"false",
                b"sentiment_model_input_mode": b"text_only",
            }
            schema = pa.schema(
                [source.schema_arrow.field(name) for name in columns] + list(SENTIMENT_SCHEMA), metadata=metadata,
            )
            with parquet.ParquetWriter(partial, schema) as writer:
                for batch in source.iter_batches(batch_size=READ_BATCH_SIZE, columns=columns):
                    table = pa.Table.from_batches([batch])
                    titles = table.column("title").to_pylist()
                    tickers = table.column("ticker").to_pylist()
                    if any(not isinstance(value, str) or not value.strip() for value in titles + tickers):
                        raise ValueError("All tickers and headlines must be nonempty strings")
                    # Reuse predictions without removing any article-ticker rows.
                    # A chunk-scoped lookup avoids a cache that grows with the file.
                    items = [
                        (ticker, target_securities[ticker]["target_name"], target_securities[ticker]["security_type"], title)
                        for ticker, title in zip(tickers, titles)
                    ]
                    unique_items = list(dict.fromkeys(items))
                    results = scorer.score(unique_items)
                    if len(results) != len(unique_items):
                        raise ValueError("Model output count does not match the headline count")
                    # Rebuild summaries from labels to validate outputs and score semantics.
                    by_item = dict(zip(unique_items, (summarize_label(row["sentiment_label"]) for row in results)))
                    predictions = [by_item[item] for item in items]
                    headline_inferences += len(unique_items)
                    for field in SENTIMENT_SCHEMA:
                        table = table.append_column(field, pa.array([row[field.name] for row in predictions], type=field.type))
                    writer.write_table(table.replace_schema_metadata(metadata))
                    label_counts.update(row["sentiment_label"] for row in predictions)
                    numeric_scores = [row["sentiment_score"] for row in predictions if row["sentiment_score"] is not None]
                    score_sum += sum(numeric_scores)
                    numeric_score_rows += len(numeric_scores)
                    processed += len(titles)
                    # Progress reflects row throughput including I/O and excludes
                    # initial model loading; it is not GPU-only inference speed.
                    rate = processed / max(monotonic() - processing_started, 1e-9)
                    eta_minutes = (source.metadata.num_rows - processed) / rate / 60
                    print(
                        f"Scored {processed:,} / {source.metadata.num_rows:,} records; "
                        f"{rate:,.0f} records/s; ETA {eta_minutes:.1f} min", flush=True,
                    )
            if processed != source.metadata.num_rows:
                raise ValueError("Scored row count does not match the input")
        report = {
            "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "input_file": str(path.resolve()), "input_columns": ["ticker", "title"], "max_tokens": MAX_LENGTH,
            "model_input_fields": ["target_ticker", "target_name", "security_type", "headline"],
            "target_securities": target_securities,
            "system_prompt": SYSTEM_PROMPT, "answer_method": "Highest next-token logit among A-E",
            "few_shot_messages": FEW_SHOT_MESSAGES,
            "thinking_enabled": False, "model_input_mode": "text_only",
            "batch_size": BATCH_SIZE, "rows_scored": processed,
            "read_batch_size": READ_BATCH_SIZE, "headline_inferences": headline_inferences,
            "records_per_second": round(processed / max(monotonic() - processing_started, 1e-9), 1),
            "label_counts": {label: label_counts[label] for label in LABELS},
            "numeric_score_rows": numeric_score_rows,
            "mean_sentiment_score": score_sum / numeric_score_rows if numeric_score_rows else None,
            "sentiment_score_definition": SCORE_DEFINITION,
            "elapsed_seconds": round(monotonic() - started, 2),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # The reader and writer must both be closed before replacement on Windows.
        os.replace(partial, path)
        os.replace(report_partial, report_path)
        return report
    finally:
        partial.unlink(missing_ok=True)
        report_partial.unlink(missing_ok=True)


def main() -> None:
    """Run scoring at the standard repository paths and print its audit."""
    root = Path(__file__).resolve().parents[2]
    securities = json.loads((root / "config" / "securities.json").read_text(encoding="utf-8"))
    report = score_news(
        root / "data" / "processed" / "news_events.parquet", root / "reports" / "sentiment_report.json", securities,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
