"""
Evaluate Mobile-Hi-SAM on HierText.

Two protocols, because they answer different questions:

  --protocol prompted
      Prompt at each ground-truth word centroid and score the returned mask
      against that instance. Measures segmentation quality given a correct
      prompt. Comparable across model variants, but it uses ground truth at
      inference, so it is NOT a deployable number.

  --protocol grid
      Prompt from a regular grid over the image (or from the ModalAligner's
      tokens with --prompts modal), deduplicate overlapping predictions, then
      match against the ground-truth instances. Needs no ground truth at
      inference. This is the honest number.

Reported metrics mean what they are named - see evaluation/metrics.py. In
particular the old ``fgIOU`` was recall; both are now reported separately.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.environ.get(
    "MOBILE_HISAM_ROOT",
    str(Path(__file__).resolve().parents[2]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM  # noqa: E402
from Mobile_Hi_SAM.evaluation.metrics import (  # noqa: E402
    MetricAccumulator,
    binarize,
    panoptic_quality,
    pixel_stats,
    recall,
    to_instances,
)
from Mobile_Hi_SAM.train.hiertext_hierarchical_dataset import (  # noqa: E402
    HierTextEvalDataset,
    eval_collate_fn,
    rasterize_polys,
)

LEVELS = ("word", "line", "para")
LEVEL_LABEL = {"word": "Word", "line": "Text-line", "para": "Layout Analysis"}
METRIC_KEYS = ("IoU", "PQ", "SQ", "RQ", "P", "R", "F", "Recall_fg")


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------
def load_model(run_dir: str, device, encoder_ckpt=None, strict=True):
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")

    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    encoder_ckpt = encoder_ckpt or config.get("checkpoint_encoder")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = MobileHiSAM(
        checkpoint_path=encoder_ckpt,
        img_size=1024,
        embed_dim=256,
        enable_hierarchical=True,
        enable_s_decoder=config.get("enable_s_decoder", False),
        transformer_mlp_dim=config.get("transformer_mlp_dim", 2048),
        init_decoder_from_sam=False,   # trained weights are about to overwrite these
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # strict=False silently loads a renamed module as random weights and still
    # reports a plausible score. Fail instead.
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint does not match the model.\n"
            f"  missing ({len(missing)}): {list(missing)[:10]}\n"
            f"  unexpected ({len(unexpected)}): {list(unexpected)[:10]}\n"
            "Re-run with --allow_partial_load only if this is understood."
        )
    if missing or unexpected:
        print(f"[warn] partial load: {len(missing)} missing, {len(unexpected)} unexpected")

    model.eval()
    return model, config


# ----------------------------------------------------------------------
# Prompt generation
# ----------------------------------------------------------------------
def grid_points(input_size, n_side: int, device):
    """A regular grid over the unpadded region, in padded-image coordinates."""
    nh, nw = input_size
    ys = torch.linspace(nh / (2 * n_side), nh - nh / (2 * n_side), n_side)
    xs = torch.linspace(nw / (2 * n_side), nw - nw / (2 * n_side), n_side)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(device)


def dedupe(masks, scores, iou_threshold=0.7):
    """NMS over masks: keep the highest-scoring of any overlapping group."""
    if not masks:
        return []
    order = np.argsort(-np.asarray(scores))
    kept = []
    for idx in order:
        candidate = masks[idx]
        area = candidate.sum()
        if area == 0:
            continue
        duplicate = False
        for k in kept:
            inter = np.logical_and(candidate, k).sum()
            if inter == 0:
                continue
            union = np.logical_or(candidate, k).sum()
            if union > 0 and inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


# ----------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_prompted(model, loader, device, max_prompts_per_image=32):
    """Oracle-prompted: one prompt per GT word, scored against that instance."""
    acc = {lvl: MetricAccumulator(METRIC_KEYS) for lvl in LEVELS}

    for batch in tqdm(loader, desc="prompted"):
        image = batch["image"].to(device)
        word_polys = batch["word_polys"][0]
        if not word_polys:
            continue

        gt = {
            "word": rasterize_polys(word_polys, model_word_size(model)),
            "line": rasterize_polys(batch["line_polys"][0], model_mask_size(model)),
            "para": rasterize_polys(batch["para_polys"][0], model_mask_size(model)),
        }
        if not gt["word"]:
            continue

        # Prompt at each word centroid, in padded-image coordinates.
        wsize = model_word_size(model)
        centres = torch.tensor(
            [[p[:, 0].mean() * 1024.0 / wsize, p[:, 1].mean() * 1024.0 / wsize]
             for p in word_polys],
            dtype=torch.float32, device=device,
        )[:max_prompts_per_image]

        n = centres.shape[0]
        out = model.forward_hierarchical({
            "image": image.expand(n, -1, -1, -1),
            "point_coords": centres.unsqueeze(1),
            "point_labels": torch.ones(n, 1, dtype=torch.int64, device=device),
        })

        for i in range(n):
            for lvl, key in (("word", "word_hr"), ("line", "line"), ("para", "para")):
                pred = binarize(out[key][i, 0])
                target = pick_target(gt[lvl], centres[i], model, lvl)
                if target is None:
                    continue
                target_t = torch.from_numpy(target).float().to(pred.device)
                stats = pixel_stats(pred, target_t)
                stats["Recall_fg"] = recall(pred, target_t)
                stats.update(panoptic_quality(to_instances(pred), [target]))
                acc[lvl].update(stats)

    return {lvl: acc[lvl].mean() for lvl in LEVELS}


def model_mask_size(model):
    return 256


def model_word_size(model):
    return 384


def pick_target(instances, centre, model, level):
    """The GT instance the prompt point falls inside, else the nearest."""
    if not instances:
        return None
    size = model_word_size(model) if level == "word" else model_mask_size(model)
    x = int(centre[0].item() * size / 1024.0)
    y = int(centre[1].item() * size / 1024.0)
    x = min(max(x, 0), size - 1)
    y = min(max(y, 0), size - 1)
    for inst in instances:
        if inst[y, x]:
            return inst
    return None


@torch.no_grad()
def evaluate_grid(model, loader, device, n_side=16, nms_iou=0.7, batch_prompts=32,
                  use_modal=False):
    """Deployable: prompts come from a grid (or the ModalAligner), never from GT."""
    acc = {lvl: MetricAccumulator(METRIC_KEYS) for lvl in LEVELS}

    for batch in tqdm(loader, desc="grid"):
        image = batch["image"].to(device)
        input_size = batch["input_size"][0]

        gt = {
            "word": rasterize_polys(batch["word_polys"][0], model_word_size(model)),
            "line": rasterize_polys(batch["line_polys"][0], model_mask_size(model)),
            "para": rasterize_polys(batch["para_polys"][0], model_mask_size(model)),
        }

        collected = {lvl: ([], []) for lvl in LEVELS}
        union = {lvl: None for lvl in LEVELS}

        if use_modal:
            out = model.forward_hierarchical({"image": image}, use_modal_aligner=True)
            chunks = [out]
        else:
            points = grid_points(input_size, n_side, device)
            chunks = []
            for start in range(0, points.shape[0], batch_prompts):
                chunk = points[start:start + batch_prompts]
                n = chunk.shape[0]
                chunks.append(model.forward_hierarchical({
                    "image": image.expand(n, -1, -1, -1),
                    "point_coords": chunk.unsqueeze(1),
                    "point_labels": torch.ones(n, 1, dtype=torch.int64, device=device),
                }))

        for out in chunks:
            scores = out["iou"].detach().cpu().numpy()
            for j, (lvl, key) in enumerate((("word", "word_hr"), ("line", "line"), ("para", "para"))):
                logits = out[key]
                for i in range(logits.shape[0]):
                    pred = binarize(logits[i, 0])
                    if pred.sum() == 0:
                        continue
                    union[lvl] = pred if union[lvl] is None else torch.maximum(union[lvl], pred)
                    for inst in to_instances(pred):
                        collected[lvl][0].append(inst)
                        collected[lvl][1].append(float(scores[i, min(j, scores.shape[1] - 1)]))

        for lvl in LEVELS:
            instances = dedupe(collected[lvl][0], collected[lvl][1], nms_iou)
            stats = panoptic_quality(instances, gt[lvl])

            # Pixel-level numbers over the union of everything predicted.
            size = model_word_size(model) if lvl == "word" else model_mask_size(model)
            gt_union = np.zeros((size, size), bool)
            for inst in gt[lvl]:
                gt_union |= inst
            pred_union = union[lvl]
            if pred_union is None:
                pred_union = torch.zeros((size, size))
            gt_t = torch.from_numpy(gt_union).float().to(pred_union.device)
            stats.update(pixel_stats(pred_union, gt_t))
            stats["Recall_fg"] = recall(pred_union, gt_t)
            acc[lvl].update(stats)

    return {lvl: acc[lvl].mean() for lvl in LEVELS}


# ----------------------------------------------------------------------
def print_table(results, protocol):
    print("\n" + "=" * 92)
    print(f"RESULTS  ({protocol} protocol)")
    print("=" * 92)
    header = f"{'Level':<18} | {'IoU':>7} | {'PQ':>7} | {'SQ':>7} | {'RQ':>7} | {'F':>7} | {'P':>7} | {'R':>7}"
    print(header)
    print("-" * 92)
    for lvl in LEVELS:
        m = results[lvl]
        print(f"{LEVEL_LABEL[lvl]:<18} | {m['IoU']:>7.2f} | {m['PQ']:>7.2f} | {m['SQ']:>7.2f} | "
              f"{m['RQ']:>7.2f} | {m['F']:>7.2f} | {m['P']:>7.2f} | {m['R']:>7.2f}")
    print("=" * 92)
    print("All values are percentages. IoU is true IoU, PQ is instance-matched.")
    if protocol == "prompted":
        print("NOTE: prompts come from ground truth. Not a deployable number.")
    print(
        "NOTE: Hi-SAM's published table quotes one fgIOU (74.86) for both Word and\n"
        "      Text-line, which suggests a single shared pixel-level metric from the\n"
        "      S-Decoder. Confirm the definition before quoting a gap against it."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--root", required=True, help="HierText dataset root")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--protocol", choices=["prompted", "grid"], default="prompted")
    parser.add_argument("--prompts", choices=["grid", "modal"], default="grid",
                        help="grid protocol only: prompt source")
    parser.add_argument("--n_side", type=int, default=16)
    parser.add_argument("--nms_iou", type=float, default=0.7)
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--allow_partial_load", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model(
        args.run_dir, device, args.encoder_ckpt, strict=not args.allow_partial_load
    )

    dataset = HierTextEvalDataset(
        args.root, split=args.split, max_items=args.max_samples, img_size=1024
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=eval_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    if args.protocol == "prompted":
        results = evaluate_prompted(model, loader, device)
    else:
        results = evaluate_grid(
            model, loader, device, n_side=args.n_side,
            nms_iou=args.nms_iou, use_modal=(args.prompts == "modal"),
        )

    print_table(results, args.protocol)

    out_path = Path(args.run_dir) / f"validation_results_{args.protocol}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
