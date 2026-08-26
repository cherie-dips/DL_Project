# Running Mobile-Hi-SAM

Measured on an Apple M4, 16 GB, macOS 15.7, PyTorch 2.9 (MPS).

## 0. One-time setup

```bash
pip install -r requirements.txt          # torch, torchvision, opencv, scipy, timm
pip install mobile_sam                   # TinyViT encoder
curl -L -o weights/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

Verify the pipeline before touching data — this runs every gate from
`REMEDIATION_PLAN.md` as an assertion and needs no dataset:

```bash
python -m Mobile_Hi_SAM.tests.test_pipeline      # expect 20/20
```

## 1. Annotations

Training needs the nested word → line → paragraph tree at
`<root>/gt/<split>.jsonl`.

**Preferred — HierText's own ground truth.** Download `train.jsonl` and
`validation.jsonl` from
[google-research-datasets/hiertext](https://github.com/google-research-datasets/hiertext)
into `Data/gt/`. Required for any number you intend to publish.

**Fallback — rebuild from the union masks in `Data/Masks/`.** If the real `gt` is
unavailable, reconstruct an approximate tree by connected components:

```bash
python -m Mobile_Hi_SAM.train.masks_to_tree \
    --masks_root Data/Masks --split train      --out Data/gt/train.jsonl
python -m Mobile_Hi_SAM.train.masks_to_tree \
    --masks_root Data/Masks --split validation --out Data/gt/validation.jsonl
```

This works — containment holds and the counts are sane — but union masks discard
instance boundaries, so touching words merge into one. Counts are lower bounds,
paragraphs suffer most, and **any gap to Hi-SAM measured this way confounds the
encoder swap with annotation damage.** Fine for training and iteration; not for
the writeup. See the module docstring for the measured numbers.

## 2. Train

The default configuration is H-Decoder only: 4,270,020 trainable parameters, with
the TinyViT encoder frozen.

```bash
cd Mobile_Hi_SAM/train

python train_hierarchical_v2.py \
    --root ../../Data \
    --checkpoint_encoder ../../weights/mobile_sam.pt \
    --batch_size 4 \
    --samples_per_image 1 \
    --num_workers 4 \
    --epochs 20 \
    --max_val_samples 200 \
    --run_name laptop
```

The device is chosen automatically (CUDA → MPS → CPU); override with `--device`.
`--use_amp` is CUDA-only and is ignored elsewhere with a warning — on MPS the
frozen encoder already keeps activation memory near 0.2 GB at batch 4, so there
is little to gain.

**Timings (M4, measured).** ~0.18 s/image, essentially flat from batch 1 to 4, so
batch size trades gradient quality against nothing. Data loading is 18 ms/image,
so 4 workers keep the GPU saturated with room to spare.

**Memory.** Annotations are read from line-delimited JSONL by byte offset and
parsed one record at a time, so worker count costs almost no RAM. This matters:
parsing the reconstructed train split into memory takes **2.55 GB**, and each
DataLoader worker would hold its own copy — about 10 GB across 4 workers on a
16 GB machine. If you substitute HierText's official `gt`, note that it ships as
one large JSON object rather than JSONL; the loader detects that and falls back
to a full in-memory parse, so drop to `--num_workers 2` in that case, or convert
it to JSONL first.

| Setup | Samples/epoch | Epoch | 20 epochs |
|---|---|---|---|
| Full train, 1 sample/image | 8,281 | ~25 min | ~8 h |
| Full train, 2 samples/image | 16,562 | ~50 min | ~17 h |
| `--max_samples 2000`, 2/image | 4,000 | ~12 min | ~4 h |

Start with `--max_samples 2000 --epochs 20` to get a curve in an afternoon, then
scale up once the validation curve looks sane.

`--samples_per_image` draws that many nested instances per image per epoch. It
changes the effective epoch size, so hold it fixed across any runs you compare.

Checkpoints land in `hierarchical_training_<run_name>/checkpoints/`;
`best_model.pth` is selected on **validation** loss.

## 3. Evaluate

```bash
cd Mobile_Hi_SAM/evaluation

# Segmentation quality given a correct prompt. Uses GT at inference,
# so it is comparable across model variants but not deployable.
python evaluate_hisam_metrics.py \
    --run_dir ../train/hierarchical_training_laptop \
    --root ../../Data --split validation --max_samples 200 \
    --protocol prompted

# No ground truth at inference: grid prompts, NMS, instance matching.
# This is the honest number. Slower - budget ~30 s/image at n_side=16.
python evaluate_hisam_metrics.py \
    --run_dir ../train/hierarchical_training_laptop \
    --root ../../Data --split validation --max_samples 50 \
    --protocol grid --n_side 12
```

Reported metrics mean what they are named. `IoU` is true IoU, `PQ` is
instance-matched. The old `fgIOU` was recall — both are now reported separately.

**On comparing to Hi-SAM's table:** their `fgIOU` is a single *stroke-level*
number aggregated over the split as `sum(I)/sum(U)`, which is why 74.86 appears
on both the Word and Text-line rows. Reproducing it needs the S-Decoder plus
Hi-SAM's contributed stroke annotations, which are a separate download. Without
those, report PQ / F / P / R per level and leave the fgIOU column empty rather
than filling it with a different quantity.

## 4. Optional: the S-Decoder

Off by default. It predicts stroke-level text, and HierText has no stroke labels
of its own — Hi-SAM's authors contributed them separately (see their
`datasets/data_preparation.md`). Enabling it without them is a hard error,
because filled word polygons are a blob where the target is letter strokes.

```bash
python train_hierarchical_v2.py ... \
    --enable_s_decoder --stroke_gt_root ../../Data
```

This takes the model to 9,692,905 trainable parameters and adds the ModalAligner,
which is what makes prompt-free inference (`--prompts modal`) meaningful.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `AssertionError: N trainable params receive no gradient` | A module is built but not in the forward path. Working as intended — do not widen the allow-list, find the module. |
| `Checkpoint does not match the model` | Architecture changed since the checkpoint. Re-train, or `--allow_partial_load` only if you know why. |
| `--enable_s_decoder needs --stroke_gt_root` | See section 4. |
| `No usable records for split` | `Data/gt/<split>.jsonl` is missing or empty. See section 1. |
| MPS out of memory | Lower `--batch_size`. Memory is ~0.2 GB at batch 4, so suspect another process. |
