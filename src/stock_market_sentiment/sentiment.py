"""Add FinSense headline sentiment to the prepared news Parquet file."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from time import monotonic

import pyarrow as pa
from pyarrow import parquet

MODEL_ID = "AnkitAI/FinSense-ModernBERT-Financial-News-Sentiment-Analysis"
MODEL_REVISION = "fa8b21b6b6549ca70263ee96b9ec52aba67cd2b4"
MAX_LENGTH = 128
BATCH_SIZE = 128
READ_BATCH_SIZE = 8192
CPU_THREADS = 6
LABELS = ("negative", "neutral", "positive")
SENTIMENT_SCHEMA = pa.schema([
    ("sentiment_score", pa.float32()),
    ("sentiment_label", pa.string()),
    ("sentiment_confidence", pa.float32()),
    ("sentiment_negative", pa.float32()),
    ("sentiment_neutral", pa.float32()),
    ("sentiment_positive", pa.float32()),
])


def summarize_probabilities(probabilities: list[float]) -> dict:
    """Convert class probabilities to a signed score and the winning label."""
    if (len(probabilities) != 3 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
            or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)):
        raise ValueError("Invalid sentiment probabilities")
    winner = max(range(3), key=probabilities.__getitem__)
    return {
        "sentiment_score": probabilities[2] - probabilities[0],
        "sentiment_label": LABELS[winner],
        "sentiment_confidence": probabilities[winner],
        **{f"sentiment_{name}": value for name, value in zip(LABELS, probabilities)},
    }


class FinSenseScorer:
    """Load the pinned FinSense model for BF16 inference on the RTX 5070."""

    def __init__(self):
        os.environ.setdefault("RAYON_NUM_THREADS", str(CPU_THREADS))
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(CPU_THREADS)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install the CUDA PyTorch build documented in README.md.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, attn_implementation="sdpa", dtype=torch.bfloat16,
        ).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        if self.model.config.id2label != dict(enumerate(LABELS)):
            raise ValueError("Unexpected FinSense label mapping")

    def score(self, headlines: list[str]) -> list[dict]:
        """Tokenize a chunk once, group similar lengths, and restore input order."""
        import torch

        encoded = self.tokenizer(headlines, padding=False, truncation=True, max_length=MAX_LENGTH)
        order = sorted(range(len(headlines)), key=lambda index: len(encoded["input_ids"][index]))
        batches = []
        with torch.inference_mode():
            for start in range(0, len(order), BATCH_SIZE):
                features = [{name: values[index] for name, values in encoded.items()} for index in order[start:start + BATCH_SIZE]]
                inputs = self.tokenizer.pad(features, padding=True, pad_to_multiple_of=8, return_tensors="pt")
                inputs = inputs.to("cuda")
                batches.append(self.model(**inputs).logits.float().softmax(dim=-1))
            probabilities = torch.cat(batches).cpu().tolist()
        results = [None] * len(headlines)
        for index, row in zip(order, probabilities):
            results[index] = summarize_probabilities(row)
        return results


def score_news(path: Path, report_path: Path) -> dict:
    """Stream existing news rows and replace the input only after all rows score."""
    if path.resolve() == report_path.resolve():
        raise ValueError("The report path must differ from the news path")
    started = monotonic()
    partial = path.with_name(path.name + ".sentiment.partial")
    report_partial = report_path.with_name(report_path.name + ".partial")
    label_counts = Counter()
    score_sum = 0.0
    processed = 0
    headline_inferences = 0
    try:
        with parquet.ParquetFile(path) as source:
            if "title" not in source.schema_arrow.names:
                raise ValueError("The prepared news file must contain a title column")
            if source.metadata.num_rows == 0:
                raise ValueError("The prepared news file is empty; no headlines to score")
            scorer = FinSenseScorer()
            processing_started = monotonic()
            columns = [name for name in source.schema_arrow.names if name not in SENTIMENT_SCHEMA.names]
            metadata = {
                **(source.schema_arrow.metadata or {}),
                b"sentiment_model": MODEL_ID.encode(),
                b"sentiment_model_revision": MODEL_REVISION.encode(),
                b"sentiment_input": b"title",
                b"sentiment_max_tokens": str(MAX_LENGTH).encode(),
                b"sentiment_score_definition": b"P(positive) - P(negative)",
            }
            schema = pa.schema(
                [source.schema_arrow.field(name) for name in columns] + list(SENTIMENT_SCHEMA), metadata=metadata,
            )
            with parquet.ParquetWriter(partial, schema) as writer:
                for batch in source.iter_batches(batch_size=READ_BATCH_SIZE, columns=columns):
                    table = pa.Table.from_batches([batch])
                    titles = table.column("title").to_pylist()
                    if any(not isinstance(title, str) or not title.strip() for title in titles):
                        raise ValueError("All prepared headlines must be nonempty strings")
                    unique_titles = list(dict.fromkeys(titles))
                    results = scorer.score(unique_titles)
                    if len(results) != len(unique_titles):
                        raise ValueError("Model output count does not match the headline count")
                    by_title = dict(zip(unique_titles, results))
                    predictions = [by_title[title] for title in titles]
                    headline_inferences += len(unique_titles)
                    for field in SENTIMENT_SCHEMA:
                        table = table.append_column(field, pa.array([row[field.name] for row in predictions], type=field.type))
                    writer.write_table(table.replace_schema_metadata(metadata))
                    label_counts.update(row["sentiment_label"] for row in predictions)
                    score_sum += sum(row["sentiment_score"] for row in predictions)
                    processed += len(titles)
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
            "input_file": str(path.resolve()), "input_column": "title", "max_tokens": MAX_LENGTH,
            "batch_size": BATCH_SIZE, "rows_scored": processed,
            "read_batch_size": READ_BATCH_SIZE, "headline_inferences": headline_inferences,
            "records_per_second": round(processed / max(monotonic() - processing_started, 1e-9), 1),
            "label_counts": {label: label_counts[label] for label in ("negative", "neutral", "positive")},
            "mean_sentiment_score": score_sum / processed,
            "sentiment_score_definition": "P(positive) - P(negative); range [-1, 1]",
            "elapsed_seconds": round(monotonic() - started, 2),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, path)
        os.replace(report_partial, report_path)
        return report
    finally:
        partial.unlink(missing_ok=True)
        report_partial.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report = score_news(root / "data" / "processed" / "news_events.parquet", root / "reports" / "sentiment_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
