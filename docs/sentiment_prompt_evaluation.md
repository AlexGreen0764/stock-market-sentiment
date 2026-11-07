# Sentiment prompt evaluation

Evaluation date: November 6, 2026. This records a bounded local prompt comparison; the full processed dataset was not rescored.

## Selected configuration

[sentiment.py](../src/stock_market_sentiment/sentiment.py) uses the target-first prompt with **five fixed examples**, the pinned Qwen3.5-4B text model, thinking disabled and the existing one-pass A–E logit selection. The batch size remains four.

The prompt permits uniformly bullish or bearish recommendations about unnamed stocks/ETFs to use the article–ticker association when the security type matches. It retains guards for other-company events, fund flows attributed to holdings, unnamed sector/commodity stories, and unresolved buy/avoid roles. Named trading-activity facts can be neutral. Questions or unchanged guidance do not themselves supply an opposing material fact.

This association rule is an inference assumption. It does not verify that the ticker is actually one of an article's recommended picks.

## Cases and reference judgments

The [evaluation fixture](../tests/fixtures/sentiment_prompt_evaluation.json) preserves cases, both assistant annotations, reference answers, candidate prompts, predictions, timings, source fingerprints and article-role checks.

- Initial sample: 120 distinct real ticker/headline pairs across all 30 selected tickers, plus 20 synthetic regression cases.
- Real-case selection: 30 ticker-balanced random cases, 30 year-balanced random cases and 60 cases targeting recommendations, activity lists, questions, opposing signals and funds.
- Initial development partition: 60 real cases plus all 20 synthetic cases. Two ambiguous/disagreed references were excluded, leaving 78 clear cases.
- Initial validation partition: 60 other real cases, with 59 clear references.
- Additional validation partition: 40 real cases disjoint from the initial sample: 20 random, 10 screened by previous negative predictions and 10 activity-list cases. There were 37 clear references.
- Total: **180 cases, 174 agreed reference labels and six excluded ambiguous/disagreed cases**.

Two separate assistant reviews labeled the supplied headline/target inputs without seeing model predictions. These are assistant judgments, not human-adjudicated ground truth. The extra negative-screen partition deliberately checks regressions among previous negative predictions; it is not an unbiased sample of all negative news.

References use the revised association-aware rubric. The original prompt used a stricter rule for unnamed recommendations. Consequently, agreement improvements include alignment with the revised task definition.

The validation partitions were later used for additional finalist comparisons. Their results are model-selection evidence, not an untouched independent final-test accuracy estimate. The fixture is now a regression set; future claims of generalization need fresh evaluation cases.

## Development comparison

All nine configurations were tested on the same development partition. Throughput excludes model loading and a warmup batch; all runs use batch size four.

| Configuration | Examples | Agreement on clear development cases | Macro-F1 | Pairs/second |
| --- | ---: | ---: | ---: | ---: |
| Original prompt | 10 | 55/78 (70.5%) | 0.670 | 7.22 |
| Compact rules | 0 | 61/78 (78.2%) | 0.746 | 21.92 |
| Compact rules | 4 | 58/78 (74.4%) | 0.652 | 12.65 |
| Compact rules | 6 | 64/78 (82.1%) | 0.805 | 10.02 |
| Compact rules | 8 | 62/78 (79.5%) | 0.785 | 8.55 |
| Target-first rules | 0 | 56/78 (71.8%) | 0.666 | 19.09 |
| Target-first rules | 6 | 67/78 (85.9%) | 0.822 | 9.78 |
| Target-first targeted wording revision | 6 | 66/78 (84.6%) | 0.802 | 9.61 |
| **Target-first rules, MIXED example removed** | **5** | **65/78 (83.3%)** | **0.798** | **11.04** |

The targeted wording revision tried to address unchanged-guidance and omitted-target-list mistakes. It produced more synthetic regressions and was rejected. Additional examples and stricter wording did not consistently improve results.

## Finalist comparisons

Five finalists were evaluated on both real validation partitions. Rates below combine their 100 scored cases; agreement excludes the four ambiguous/disagreed references, leaving 96. Combined throughput is total cases divided by total timed scoring seconds, not an average of the two rates.

| Configuration | Validation agreement | Validation macro-F1 | Agreement across all 174 clear cases | Combined validation pairs/second |
| --- | ---: | ---: | ---: | ---: |
| Original, 10 examples | 68/96 (70.8%) | 0.608 | 123/174 (70.7%) | 7.20 |
| Compact, no examples | 70/96 (72.9%) | 0.576 | 131/174 (75.3%) | 22.15 |
| Compact, 6 examples | 73/96 (76.0%) | 0.648 | 137/174 (78.7%) | 10.22 |
| Target-first, 6 examples | 76/96 (79.2%) | 0.653 | 143/174 (82.2%) | 10.19 |
| **Selected: target-first, 5 examples** | **77/96 (80.2%)** | **0.645** | **142/174 (81.6%)** | **11.30** |

The five-example variant differed from six examples by one agreement across 174 clear cases and ran about 11% faster on the combined validation cases. Its agreement was close in this small experiment; formal statistical equivalence is not established. It was selected as a practical speed/accuracy compromise, rather than the configuration with the highest development agreement.

Against the original prompt, selected-prompt throughput was approximately **57% higher** on the same combined validation cases. This is a bounded inference measurement, not a promised full-dataset runtime improvement. The no-example configuration was substantially faster but lost more reference agreement and failed both reference-MIXED validation cases.

## Important class-level tradeoffs

The selected prompt did not improve every class. These are correct predictions divided by reference-class cases in the 96 clear real validation examples:

| Reference class | Original | Selected |
| --- | ---: | ---: |
| Negative | 8/10 | 5/10 |
| Neutral | 8/16 | 15/16 |
| Positive | 9/21 | 19/21 |
| Mixed | 1/2 | 0/2 |
| Insufficient information | 42/47 | 38/47 |

Negative headlines and MIXED remain weaknesses. Some declines are incorrectly classified as neutral or mixed, and some named activity lists excluding the target are incorrectly classified as neutral. The rare-class counts are too small for stable population estimates. Higher total agreement should not be interpreted as proof of improved bearish-event detection.

The six-example configuration initially matched 52/59 clear cases in the first validation partition, but matched only 24/37 in the later negative/activity-heavy partition. Reporting only the initial 88.1% would hide this variation; both partitions are included above.

## Recommendation-attribution checks

Five development recommendation articles were inspected to check whether a ticker tag establishes membership in the recommended group:

| Target and source article | Observed role |
| --- | --- |
| NKE — [3 Big-Brand Stocks to Buy in February](https://www.nasdaq.com/articles/3-big-brand-stocks-buy-february-2017-02-15) | Nike is one of the recommended picks. |
| AMGN — [3 Big Biotech Stocks to Buy in January](https://www.fool.com/investing/2018/01/06/3-big-biotech-stocks-to-buy-in-january.aspx) | Amgen is one of the recommended picks. |
| AMGN — [2 Top Dividend Stocks to Buy in 2018](https://www.fool.com/investing/2018/01/07/2-top-dividend-stocks-to-buy-in-2018.aspx) | Amgen is one of the recommended picks. |
| BABA — [Companies Powering the Winter Olympics: 3 Stocks to Buy Now](https://www.nasdaq.com/articles/companies-powering-winter-olympics-3-stocks-buy-now-2018-02-12) | Alibaba is mentioned as an Olympic sponsor; the numbered picks are Intel, Visa and Panasonic. |
| NEE — [3 Top Energy Stocks to Buy Now](https://www.nasdaq.com/articles/3-top-energy-stocks-to-buy-now-2019-09-01) | The picks include NEP, not NEE; NEE is discussed as its parent and appears in disclosure text. |

These cases demonstrate both the value and the risk of inferred recommendation sentiment. They do not estimate the population frequency of incorrect associations. Frozen headline-only policy references were not relabeled using the article bodies; the body checks are recorded separately. Agreement with an association-based reference is therefore not proof of actual recommendation membership.

## Runtime, integration and preservation checks

- GPU runs used the cached pinned model with CUDA/BF16, thinking disabled, A–E next-token selection and no KV cache.
- Repeated batch tests on the six-example candidate measured about 10.37 pairs/second at batch four versus 9.46 at batch eight; peak allocated memory was about 8.28 versus 8.64 GiB. The five-example version retains batch four; batch eight was not separately retested for that final ablation.
- The final source prompt and examples were checked for exact equality with the selected tested configuration.
- Cached-tokenizer checks on the 100 longest tokenized input payloads in the saved dataset found a maximum final prompt length of **653 tokens**, within the unchanged 1,024-token limit. No truncation was used.
- At the time of this prompt evaluation, all 21 fixture unit tests passed, including JSON input format, target identity/type, disabled thinking/cache, restored input order, oversized-prompt rejection before model calls, typed data preservation, deduplication and failure preservation. These checks preceded adaptive ticker selection and do not validate that integration.
- Ruff checks passed for that evaluated revision. Fixture tests used model doubles and did not rerun the GPU evaluation.
- SHA-256 checks confirmed both production Parquet files and both production reports were unchanged. The existing sentiment report still describes the earlier full run.

To build the candidate histories and select the final universe with this prompt:

```powershell
.\.venv\Scripts\python.exe -m stock_market_sentiment
.\.venv\Scripts\python.exe -m stock_market_sentiment.sentiment
```

Once candidates are prepared, rerun only the second command to resume scoring or apply a prompt change. Target identities come from the historical symbol catalog described in the [README](../README.md#input-data). Evaluation results refer to the target names saved in the fixture.

The evaluation fixture contains the measured comparisons. It is not automatically executed by unit-test discovery; replaying its real-model cases requires explicit GPU inference through the existing scorer.
