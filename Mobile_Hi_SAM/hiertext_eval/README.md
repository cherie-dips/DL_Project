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

## Inference protocol

Predictions come from `evaluation/auto_mask_generator.py`, which reproduces
Hi-SAM's `hi_sam/modeling/auto_mask_generator.py`. Their protocol is **not** a
prompt grid:

1. ModalAligner -> S-Decoder -> predicted text foreground;
2. up to `fg_points_num` (600) points sampled uniformly at random **from that
   foreground**, so every prompt lands on predicted text;
3. H-Decoder over those points in batches of `batch_points_num` (100);
4. drop predictions whose **line** score is below `score_thresh` (0.5);
5. Matrix NMS (SOLOv2, gaussian, sigma 2.0) on the **line** masks; keep
   `updated_score > nms_thresh` (0.5);
6. group lines into paragraphs by pairwise IoU of their predicted paragraph
   masks (`get_para_iou`), union-find above `para_thresh` (0.5).

`matrix_nms` and `get_para_iou` are copied verbatim. Verified: a duplicate mask
decays 0.80 -> 0.108 while a disjoint one holds 0.85 -> 0.850, and the affinity
grouping recovers the correct paragraph partition.

This matters. A grid puts most prompts on background, where the decoder still
emits a mask, which is why the earlier grid-prompted numbers understated the
model so badly. Prompt-free inference therefore **requires the S-Decoder** -
without it there is no foreground to sample from, and the generator raises
rather than silently falling back.
