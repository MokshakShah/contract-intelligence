# Contract Intelligence Assistant Report

## Summary

The final system is a deterministic BM25 clause retriever over 194 CUAD mini
queries and 29 indexed documents. It returns offset-addressable passages and
supports abstention, CLI, and Streamlit interfaces. The final configuration is
the baseline configuration because all eight self-improvement candidates were
rejected by the stated Recall@8 acceptance rule.

Headline numbers from the final recorded evaluation:

- Train benchmark: 140 queries; Recall@8 `0.0887`, Precision@8 `0.0106`,
  document hit@8 `1.0000`, span hit@8 `0.1857`.
- Test benchmark: 54 queries; Recall@8 `0.1232`, Precision@8 `0.0149`,
  document hit@8 `1.0000`, span hit@8 `0.2037`.
- Frozen unanswerable set: 44 items, with 30 train and 14 test items.
- Correct abstention rate: `0.0000` on both evaluated splits because the
  accepted final configuration has abstention disabled.
- Test-run counter: `2`, exactly one baseline test run and one final test run.
- Regression suite: `15 passed`.

## System design and alternatives considered

The data layer applies the required mini selection, deterministic split, UTF-8
document loading, and offset validation. The chunker supports fixed character
windows and recursive paragraph/sentence boundaries. The baseline uses
500-character, non-overlapping chunks.

Retrieval is pluggable. BM25 is the baseline because it is fast, local, easy
to reproduce, and effective for exact contract vocabulary. Dense retrieval
using `BAAI/bge-small-en-v1.5`, hybrid reciprocal-rank fusion, and an optional
cross-encoder are implemented as configurable alternatives. Embeddings are
cached under `.cache/`. Results include raw score and normalized score.

The metrics implementation follows the LegalBench-RAG character-span framing.
Retrieved overlaps and multiple answer snippets are unioned by document before
character overlap is computed. In addition to precision and recall, the
system reports document hit and span hit.

Abstention is a post-retrieval policy. It supports no rule, normalized top-score
threshold, top-two margin, and a combined rule. The shared `search` function
builds and caches the configured index, applies retrieval and abstention, and
is used by both user interfaces.

Alternatives considered:

- A dense-first system was not selected as the baseline because it adds model
  download/runtime cost and was not needed for a one-hour, measurable baseline.
- Hybrid retrieval and cross-encoder reranking remain available, but require
  train-only tuning and additional model resources.
- A generative answerer was rejected because the deliverable is ranked evidence
  passages, not free-form legal advice.
- A learned abstention classifier was rejected in favor of transparent score
  and margin rules with a small deterministic threshold sweep.

## LLM use

No LLM was used in the runtime pipeline, query expansion, reranking, evaluation,
or self-improvement decisions. This is intentional for confidential contracts:
avoiding external transmission improves privacy, avoids per-query cost, and
keeps latency and behavior reproducible. Rule-based diagnosis and parameter
proposals are inspectable and produce a concrete experiment trail.

The next experiment I would run is a privacy-preserving local or self-hosted
cross-encoder reranker, evaluated only on train first. I would compare it with
hybrid retrieval and test whether it improves character Recall@8 without
violating precision or abstention guardrails.

## Evaluation setup

The benchmark mini set is selected exactly as required from `benchmarks/cuad.json`.
The deterministic split gives 140 train queries and 54 test queries. The
indexed corpus contains the 29 unique documents referenced by those 194
queries. Offset validation reported zero mismatches before evaluation.

For each query, retrieved spans are unioned by document. Character-level
Precision@k is overlap characters divided by retrieved characters; Recall@k is
overlap characters divided by ground-truth characters. Metrics are reported at
k values 1, 2, 4, 8, 16, and 32. Document hit and span hit are also averaged.

The frozen unanswerable set contains 44 authored queries: 30 train and 14 test
after the same split function. It mixes absent named contracts, rare/absent
clause types, out-of-domain policy questions, and plausible overly specific
questions. BM25 and targeted term checks were used as a sanity check.

The final reporting sequence was:

1. Baseline on train.
2. Final on train.
3. Baseline on test with `--final`.
4. Final on test with `--final`.

The recorded test-run counter is `2`. No test evaluation was run during
tuning, and no tuning occurred after the final test evaluations.

## Results

All values below are taken from the four JSONL records produced by the commands
above. The final configuration is byte-for-byte behaviorally equivalent to the
baseline because no loop candidate was accepted.

### Train benchmark

| Configuration | n | Precision@8 | Recall@8 | Doc hit@8 | Span hit@8 |
|---|---:|---:|---:|---:|---:|
| Baseline | 140 | 0.0106 | 0.0887 | 1.0000 | 0.1857 |
| Final | 140 | 0.0106 | 0.0887 | 1.0000 | 0.1857 |

### Test benchmark

| Configuration | n | Precision@8 | Recall@8 | Doc hit@8 | Span hit@8 |
|---|---:|---:|---:|---:|---:|
| Baseline | 54 | 0.0149 | 0.1232 | 1.0000 | 0.2037 |
| Final | 54 | 0.0149 | 0.1232 | 1.0000 | 0.2037 |

At k=1, both configurations have zero character precision/recall and document
hit of `0.9929` on train and `0.9815` on test. At k=2, train recall is
`0.0059` and test recall is `0.0057`. At k=4, train recall is `0.0301` and
test recall is `0.0612`.

Abstention was disabled in both recorded configs, so false-abstention and
correct-abstention rates are `0.0000` on both splits. This is a deliberate
baseline/final outcome, not a claim that the unanswerable set was answered
correctly.

## Iteration history

The loop objective was character-level Recall@8 on train benchmark queries.
Guardrails were false-abstention rate no more than three percentage points above
baseline, correct-abstention rate no more than three points below baseline, and
Precision@8 no more than 20% below baseline. A candidate also needed a Recall@8
improvement of at least `0.001`.

Each iteration evaluated the current train configuration, diagnosed every
failed train benchmark and train-unanswerable query, proposed a deterministic
change, evaluated the candidate, and accepted it only if all conditions held.
The dominant train benchmark diagnosis was `right_doc_wrong_section`.

| Iteration | Proposer/change | Recall@8 | Precision@8 | False abstain | Correct abstain | Decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |
| 2 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |
| 3 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |
| 4 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |
| 5 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |
| 6 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |
| 7 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |
| 8 | rule: smaller overlapping chunks | 0.0421 | 0.0101 | 0.0000 | 0.0000 | Reject |

The loop's candidate metrics differ from the final baseline metric because the
loop indexes only documents referenced by train benchmark queries, as required
by its train-only data boundary, while the general evaluator indexes all
documents referenced by the 194-query mini set. Every candidate was rejected
because it failed the minimum Recall@8 improvement over the current accepted
configuration.

## Error analysis

The following diagnosis uses the final BM25 configuration and the same
classification rules as the loop. A benchmark query is a
`right_doc_wrong_section` failure when the right document is retrieved but no
retrieved span overlaps an answer span. It is
`right_section_partial_span` when overlap exists but Recall@8 is below 50%.
Unanswerable queries retrieved without abstaining are `wrongly_answered`.

### Train counts

- `wrong_document`: 0
- `right_doc_wrong_section`: 114
- `right_section_partial_span`: 13
- `false_abstention`: 0
- `wrongly_answered`: 30

Concrete train failures:

1. **Right document, wrong section:** “Consider the Amendment No. 3 to
   Wireless Content License Agreement between Fox Mobile Entertainment, Inc.
   and Glu Mobile Inc.; What is the expiration date of this contract?”
2. **Right document, wrong section:** “Consider the Service Agreement between
   BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What
   is the governing law for this contract?”
3. **Right document, wrong section:** “Consider the Service Agreement between
   BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Are
   there any exceptions to competitive restrictions in this contract?”
4. **Partial span:** “Consider the Cooperation Agreement between HPIL
   ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is
   there a cap on liability under this contract?”
5. **Partial span:** “Consider the Supply Agreement between Florida Chemical
   Company, LLC and Flotek Chemistry, LLC for Terpene Products; Does this
   contract include any volume restrictions?”
6. **Partial span:** “Consider the Intellectual Property Agreement for COVID-19
   Treatment Development between Marv Enterprises, LLC, Premier Biomedical,
   Inc., and Technology Health, Inc.; Does this contract include an exclusivity
   agreement?”
7. **Wrongly answered unanswerable:** “Consider the Apple-Google advertising
   partnership; What is the renewal term under this agreement?”
8. **Wrongly answered unanswerable:** “Consider the Warner Bros. and HBO Max
   distribution agreement; Is there a cap on liability?”
9. **Wrongly answered unanswerable:** “Consider the Microsoft-OpenAI cloud
   services contract; What audit rights does the customer receive?”
10. **Wrongly answered unanswerable:** “What is the appointed data protection
    officer's name under this agreement?”

### Test counts

- `wrong_document`: 0
- `right_doc_wrong_section`: 43
- `right_section_partial_span`: 5
- `false_abstention`: 0
- `wrongly_answered`: 14

Concrete test failures:

1. **Right document, wrong section:** “Consider the Cooperation Agreement
   between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy
   Projects; Can this contract be terminated for convenience, and under what
   conditions?”
2. **Right document, wrong section:** “Consider the Service Agreement between
   Theravance Biopharma UK Limited and Brett Haumann; What is the governing law
   for this contract?”
3. **Right document, wrong section:** “Consider the Service Agreement between
   Theravance Biopharma UK Limited and Brett Haumann; Does this contract
   provide for joint intellectual property ownership?”
4. **Partial span:** “Consider the Promotion Agreement between Depomed, Inc.
   and King Pharmaceuticals, Inc. for Glumetza; Does this contract include any
   right of first refusal, right of first offer, or right of first negotiation?”
5. **Partial span:** “Consider the Promotion Agreement between Depomed, Inc.
   and King Pharmaceuticals, Inc. for Glumetza; Is there a minimum commitment
   required under this contract?”
6. **Partial span:** “Consider the Promotion Agreement between Depomed, Inc.
   and King Pharmaceuticals, Inc. for Glumetza; Are there any services to be
   provided after the termination of this contract?”
7. **Wrongly answered unanswerable:** “Consider the Netflix-Disney carriage
   agreement; What is the notice period for terminating this contract?”
8. **Wrongly answered unanswerable:** “Consider the Tesla and Panasonic battery
   supply contract; What is the governing law for this agreement?”
9. **Wrongly answered unanswerable:** “Consider the Nike and Adidas endorsement
   contract; What are the permitted promotional uses?”
10. **Wrongly answered unanswerable:** “What is the exact liquidated-damages
    amount of 17.5 million dollars payable upon breach?”

The absence of `wrong_document` failures is consistent with the very high
document hit rates: BM25 usually identifies the correct contract, but 500
character windows and query/document vocabulary place the answer elsewhere.
The dominant improvement opportunity is therefore section localization and
span coverage rather than document retrieval.

## Train-versus-test gap

Test Recall@8 (`0.1232`) is higher than train Recall@8 (`0.0887`), and test
Precision@8 (`0.0149`) is higher than train Precision@8 (`0.0106`). Test span
hit@8 (`0.2037`) is also higher than train span hit@8 (`0.1857`). This is not
evidence of generalization improvement: the test sample is only 54 queries,
the split is hash-based rather than stratified by contract type, and all
queries share the same small 29-document corpus. The test set can therefore
be easier by chance. Document hit is near-perfect on both splits, while exact
span localization remains low on both.

## Limitations and next steps

- Chunk boundaries are not clause-aware, so a relevant clause can be split
  across windows or missed by the top-k window budget.
- BM25 lexical matching can rank a semantically related section above the
  answer and cannot reliably identify unanswerable questions.
- The final abstention configuration is disabled because the loop accepted no
  threshold candidate; the unanswerable-set result is consequently weak.
- The unanswerable set is hand-authored and BM25-checked, not independently
  adjudicated by legal experts.
- The current evaluation reports retrieval evidence, not answer correctness,
  citation quality, or legal safety.
- The system has no cross-reference resolution for clauses referring to
  schedules, exhibits, amendments, or defined terms elsewhere in a contract.
- Larger corpora will require persistent index metadata, incremental updates,
  and more careful embedding cache keys.
- A local cross-encoder reranker and clause-aware/recursive chunking are the
  most direct next retrieval experiments.
- Query expansion, definition-aware retrieval, and cross-reference linking
  should be tested on train with the same guardrails.
- An abstention calibration study should compare score thresholds, margins,
  and a supervised confidence model.
- Legal-user trust work should add provenance, document-page rendering,
  uncertainty explanations, and explicit “not legal advice” product
  boundaries.
