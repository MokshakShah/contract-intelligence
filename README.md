# Contract Intelligence Assistant

## Self-improvement objective

`python -m src.loop --config configs/baseline.yaml --iterations 8` tunes only
the exact `train` split. The primary objective is character-level Recall@8 on
train benchmark queries. Guardrails are false-abstention rate no more than
three percentage points above baseline, correct-abstention rate on train
unanswerable queries no more than three points below baseline, and Precision@8
no more than 20% below baseline. A candidate is accepted only when Recall@8
improves by at least the configured minimum delta and every guardrail holds.
The loop writes accepted best configuration to `configs/final.yaml` and logs
each diagnosis, proposal, metric result, and decision to `experiments/`.
