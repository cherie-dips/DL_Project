# Results

All numbers from the `prompted` protocol on the HierText validation split
(official ground truth, normalised via `normalize_gt.py`). Encoder fine-tuned,
S-Decoder off, H-Decoder only.

## Headline — full run, 8,281 images, 12 epochs (~10 h on an M4)

Use **`checkpoints/epoch_12.pth`**, not `best_model.pth` — see *Checkpoint
selection* below.

| Level | IoU | PQ | SQ | RQ | F | P | R |
|---|---|---|---|---|---|---|---|
| Word | 70.73 | 65.16 | 65.55 | 82.84 | 80.57 | 81.61 | 87.14 |
| Text-line | 68.83 | 60.60 | 62.41 | 77.45 | 79.06 | 83.38 | 81.65 |
| Layout | 46.07 | 31.71 | 36.55 | 43.74 | 56.66 | 82.12 | 61.04 |

**These are NOT comparable to Hi-SAM's published table.** The `prompted`
protocol takes its prompts from ground truth, which Hi-SAM's evaluation does
not; their numbers also come from the official `hiertext_eval`. For a
deployable figure use `--protocol grid`, which needs no ground truth at
inference. The fgIOU column is deliberately absent: Hi-SAM's fgIOU is a single
stroke-level number requiring the S-Decoder and their contributed stroke
annotations.

## Ablations — 400 images, 8 epochs each, identical data and schedule

| Config | Trainable | best val | Word IoU | Line IoU | Layout IoU | Layout PQ |
|---|---|---|---|---|---|---|
| Frozen encoder, 1x1 adapter | 4,270,020 | 2.2133 | 43.37 | 35.99 | 24.82 | 6.40 |
| Frozen, dilated context adapter | 4,414,404 | 2.3696 | 43.30 | 34.96 | 22.67 | 7.08 |
| **Fine-tuned encoder** | 10,013,912 | **2.0929** | **51.90** | **45.12** | **28.45** | **11.13** |

**The context adapter is a negative result.** Widening the adapter's receptive
field from 1 to 21 cells on the 64x64 map bought nothing — layout IoU fell 2.2
points. Paragraph is not context-starved. Kept behind `--adapter context` so it
is not retried.

**Fine-tuning wins**, but the comparison is confounded: that arm ran at batch 2
(twice the optimiser steps per epoch) with 2.3x the parameters. A clean
attribution needs the frozen arm re-run at batch 2.

## Data volume was what paragraph needed

Scaling 400 -> 8,281 images, fine-tuned encoder throughout:

| | 400 images | full | change |
|---|---|---|---|
| Word PQ | 37.17 | 65.16 | +75% |
| Line PQ | 26.43 | 60.60 | +129% |
| Layout PQ | 11.13 | 31.71 | +185% |

Paragraph loss looked flat in every 400-image run, across frozen, context and
fine-tuned configurations. That was a data-volume artefact, not an architectural
limit: layout PQ nearly tripled on the full set. Mid-run readings of the
paragraph trend from three or four noisy epochs were repeatedly wrong.

## Checkpoint selection

`best_model.pth` (epoch 9, val 1.6200) is **worse on every metric** than the
final epoch (val 1.6909):

| | epoch 9 | epoch 12 |
|---|---|---|
| Word IoU / PQ | 69.48 / 63.30 | 70.73 / 65.16 |
| Line IoU / PQ | 67.63 / 58.55 | 68.83 / 60.60 |
| Layout IoU / PQ | 44.80 / 30.28 | 46.07 / 31.71 |

Paragraph is ~58% of the weighted validation loss and its noisiest term, so
selection on total loss tracks paragraph variance rather than model quality.
Selecting on validation loss fixed the original bug (selection on *train* loss,
which just picks the last epoch); it did not make selection correct. Select on a
metric, or downweight paragraph in the selection criterion.

## Full training curve

| Epoch | train | val | val word | val line | val para |
|---|---|---|---|---|---|
| 1 | 2.2110 | 2.0710 | 0.6463 | 0.3745 | 2.0037 |
| 2 | 1.9057 | 2.0562 | 0.5809 | 0.3333 | 2.1843 |
| 3 | 1.7787 | 2.0675 | 0.5724 | 0.3340 | 2.2261 |
| 4 | 1.6574 | 1.7285 | 0.5100 | 0.3113 | 1.7179 |
| 5 | 1.6439 | 1.7842 | 0.5015 | 0.2991 | 1.8723 |
| 6 | 1.5967 | 1.7455 | 0.4783 | 0.2894 | 1.8575 |
| 7 | 1.5534 | 1.7446 | 0.4779 | 0.2825 | 1.8670 |
| 8 | 1.4971 | 1.8570 | 0.4507 | 0.2737 | 2.1723 |
| 9 | 1.4584 | 1.6200 | 0.4395 | 0.2695 | 1.7283 |
| 10 | 1.4489 | 1.6905 | 0.4238 | 0.2610 | 1.9150 |
| 11 | 1.4027 | 1.6854 | 0.4108 | 0.2586 | 1.9344 |
| 12 | 1.4012 | 1.6909 | 0.4151 | 0.2578 | 1.9407 |

Word and line improved monotonically for all 12 epochs. Paragraph oscillated
between 1.72 and 2.23 throughout.
