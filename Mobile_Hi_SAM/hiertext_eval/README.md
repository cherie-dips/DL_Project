# hiertext_eval (vendored)

Copied verbatim from [ymy-k/Hi-SAM](https://github.com/ymy-k/Hi-SAM/tree/main/hiertext_eval),
originally from [google-research-datasets/hiertext](https://github.com/google-research-datasets/hiertext).
This is the evaluator behind the numbers in Hi-SAM's paper, so results from it are
directly comparable to their table; `evaluation/metrics.py` is a correct but
independent implementation and is not.

**Metric logic is unmodified.** Two changes only, both recorded here:

1. `eval.py` — `apache_beam` is imported lazily. It is used solely by `main()` to
   run a parallel map over images; no metric function touches it, and it is a very
   large dependency to install for a parallel for-loop.
   `evaluation/run_hiertext_eval.py` calls the same functions serially.
2. `evaluator/polygon_ops.py` — `dtype=np.float` → `dtype=float`. `np.float` was
   removed in NumPy 1.20 and was only ever an alias for the builtin, so this is
   not a behaviour change.

## Use

```bash
# 1. predictions in HierText's format
python -m Mobile_Hi_SAM.evaluation.run_hiertext_eval predict \
    --run_dir <training run> --root Data/HierText --split validation --out preds.json

# 2. official metrics
python -m Mobile_Hi_SAM.evaluation.run_hiertext_eval score \
    --gt Data/gt/validation.jsonl --result preds.json
```

`--mask_stride 2` roughly quarters scoring time at a small accuracy cost.

## Verification

Ground truth scored against itself: Det-PQ 1.0, Det-Fscore 1.0, precision and
recall 1.0 at word, line and paragraph (word PQ reads 0.99999998 — float
rounding). Shifting every predicted polygon degrades it as expected:

| shift | word PQ | line PQ | paragraph PQ |
|---|---|---|---|
| 0 px | 1.000 | 1.000 | 1.000 |
| 5 px | 0.713 | 0.692 | 0.699 |
| 25 px | 0.149 | 0.215 | 0.131 |

Requires `absl-py` and `six` (both small); `apache_beam` is not needed.
