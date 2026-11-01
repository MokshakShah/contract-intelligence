# Contract Intelligence Assistant

## Overview

This project is a small, measurable clause-search system over the LegalBench-RAG
CUAD mini benchmark. Given a natural-language contract question, it returns
ranked passages with the source document, character offsets, text, and scores.
It can abstain when the configured confidence rule says that no result is
relevant enough.

The indexed data is restricted to `benchmarks/cuad.json` and the contracts
referenced by the 194-query mini selection under `corpus/cuad/`. Ground-truth
spans are used for evaluation only and are never supplied to retrieval.

## Setup

Use Python 3.11. Create and activate a virtual environment, then install the
declared dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download only the allowed benchmark and referenced corpus files:

```powershell
python scripts\download_data.py
```

The download is written under `data/raw/`. Downloads and local caches are
ignored by Git. No API key is required for the public dataset.

## How to run

Start the web application:

```powershell
python -m streamlit run app.py
```

Run a CLI search:

```powershell
python -m src.cli "What is the governing law for this contract?" --config configs\final.yaml --k 5
```

Run evaluation on train:

```powershell
python -m src.eval --config configs\baseline.yaml --split train
```

Test evaluation is protected and requires an explicit final flag:

```powershell
python -m src.eval --config configs\final.yaml --split test --final
```

Run the train-only self-improvement loop:

```powershell
python -m src.loop --config configs\baseline.yaml --iterations 8
```

Run the threshold sweep:

```powershell
python -m src.tune_threshold --split train
```

Run tests:

```powershell
python -m pytest -q
```

## Architecture and configuration

`src/data.py` implements the exact mini selection and SHA-256 split assignment,
loads UTF-8 documents, and validates every benchmark span. `src/chunking.py`
creates fixed or recursive chunks while preserving character offsets.

`src/retrieval.py` contains BM25, dense, hybrid, optional cross-encoder
reranking, per-document diversity limits, and the shared cached
`search(query, config, k)` service used by both interfaces. Dense embeddings
are cached under `.cache/`.

`src/abstention.py` implements no abstention, normalized score thresholds,
score margins, and combined rules. `src/metrics.py` implements character-level
precision/recall and document/span hit metrics. `src/eval.py` evaluates a
selected split and logs JSONL records. `src/diagnose.py` and `src/loop.py`
implement train-only error diagnosis and guarded configuration improvement.

All pipeline choices live in YAML:

- `configs/baseline.yaml` is the reproducible baseline.
- `configs/final.yaml` is the accepted best configuration produced by the loop.

The CLI is in `src/cli.py`; the Streamlit UI is in `app.py`. Experiment records
are in `experiments/log.jsonl` and `experiments/LOG.md`.

## Reproducing results

From a clean environment with the data already downloaded, run:

```powershell
python -m pytest -q
python -m src.eval --config configs\baseline.yaml --split train
python -m src.eval --config configs\final.yaml --split train
python -m src.eval --config configs\baseline.yaml --split test --final
python -m src.eval --config configs\final.yaml --split test --final
```

On the development machine used for this submission, a BM25 train evaluation
typically takes roughly 15–45 seconds and a test evaluation roughly
10–30 seconds, depending on cache state and filesystem load. The dense and
reranking options require model downloads on their first run and can take
longer. The exact commands above append records to the experiment log; test
evaluations also increment `experiments/test_runs.json`.

## Assumptions

- Python 3.11 is the supported runtime.
- The dataset is the public `awinml/legalbench-rag` Hugging Face dataset.
- Only the exact `mini()` function and the exact SHA-256 split function are
  used.
- The mini benchmark contains 194 queries; split assignment is deterministic.
- Documents are read in default text mode with UTF-8 and offsets are Python
  character offsets.
- Every snippet span is checked against the loaded document before evaluation.
- Multiple ground-truth snippets are unioned by document before scoring.
- Multiple retrieved spans are also unioned by document before scoring.
- Precision and recall are character-level and are reported for k values
  1, 2, 4, 8, 16, and 32.
- Document hit means at least one retrieved document matches a ground-truth
  document. Span hit means any positive character overlap.
- Retrieval scores are raw method scores plus a candidate-list sigmoid
  z-score used by abstention. The scale is five standard deviations to reduce
  saturation across candidate lists.
- Abstention is applied after retrieval. An abstained benchmark query has zero
  recall in the abstained-as-zero view.
- The unanswerable set is synthetic, frozen, contains 44 items, and is split
  with the same deterministic function. Items were authored in benchmark-like
  language, then checked against the indexed corpus with BM25 and targeted
  term searches. The check was used to remove obvious answer-bearing items;
  it is not a proof of semantic impossibility.
- Tuning and the self-improvement loop use train benchmark queries and train
  unanswerable queries only. Test is reserved for final reporting.
- The improvement loop accepts a candidate only if Recall@8 improves by at
  least its minimum delta and all precision/abstention guardrails hold.
- The loop uses a fixed random seed and deterministic rule/grid proposals.
- The default baseline is BM25 with fixed 500-character chunks and no overlap.
- The final interface returns passages rather than legal advice.
- No source data or downloaded corpus is committed to the repository.

## AI tool usage

AI assistance was used to scaffold files, implement and review the retrieval
pipeline, write tests, run commands, and draft this documentation. The
retrieval and tuning decisions are explicit Python/YAML rules; no LLM was used
at inference, for query expansion, for reranking, or to choose evaluation
results. All reported numbers were produced by the repository's code.
