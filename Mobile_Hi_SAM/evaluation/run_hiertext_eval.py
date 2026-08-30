"""
Run the official HierText evaluator on Mobile-Hi-SAM predictions.

Two steps, either of which can be run alone:

  1. ``predict``  - run inference over a split and write a prediction JSON in
                    HierText's format (paragraphs -> lines -> words, each word a
                    polygon in ORIGINAL image coordinates).
  2. ``score``    - hand that JSON and the ground-truth JSON to the vendored
                    ``hiertext_eval`` and print its metrics.

    python -m Mobile_Hi_SAM.evaluation.run_hiertext_eval predict \\
        --run_dir ../train/hierarchical_training_x --root Data/HierText \\
        --split validation --out preds.json
    python -m Mobile_Hi_SAM.evaluation.run_hiertext_eval score \\
        --gt Data/HierText/gt/validation.jsonl --result preds.json

The scoring path calls the official functions serially instead of through
apache_beam. The metric code itself is untouched, so the numbers are the ones
Hi-SAM's table reports - unlike evaluation/metrics.py, which is a correct but
independent implementation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.environ.get(
    "MOBILE_HISAM_ROOT", str(Path(__file__).resolve().parents[2])
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HIERTEXT_EVAL_DIR = str(Path(__file__).resolve().parents[1] / "hiertext_eval")
if HIERTEXT_EVAL_DIR not in sys.path:
    sys.path.insert(0, HIERTEXT_EVAL_DIR)


# ----------------------------------------------------------------------
# Scoring: the official pipeline, serially
# ----------------------------------------------------------------------
def score(gt_path: str, result_path: str, output: Optional[str],
          eval_lines: bool, eval_paragraphs: bool, mask_stride: int) -> str:
    import eval as ht  # the vendored hiertext_eval/eval.py
    from evaluator import evaluator

    word_evaluator = evaluator.HierTextEvaluator(
        text_box_type=evaluator.TextBoxRep.POLY, evaluate_text=False
    )
    line_evaluator = evaluator.HierTextEvaluator() if eval_lines else None
    paragraph_evaluator = evaluator.HierTextEvaluator() if eval_paragraphs else None

    annotations = ht.load_annotations(gt_path, result_path)

    # beam.Map -> a loop; beam.CombineGlobally(dict_add) -> one dict_add call.
    per_image = []
    for anno in tqdm(annotations, desc="scoring"):
        parsed = ht.parse_annotation_dict(anno, eval_lines, eval_paragraphs, mask_stride)
        per_image.append(
            ht.evaluate_one_image(
                *parsed, word_evaluator, line_evaluator, paragraph_evaluator
            )
        )

    summed = ht.dict_add(per_image)
    metrics = ht.compute_eval_metrics(
        *summed, word_evaluator, line_evaluator, paragraph_evaluator
    )
    text = ht.metric_format(metrics)
    if isinstance(text, list):
        text = "\n".join(text)
    print(text)
    if output:
        Path(output).write_text(text)
        print(f"\nSaved: {output}")
    return text


# ----------------------------------------------------------------------
# Prediction: model output -> HierText prediction JSON
# ----------------------------------------------------------------------
def mask_to_polygon(mask: np.ndarray, scale_x: float, scale_y: float,
                    min_area: int = 8) -> Optional[List[List[int]]]:
    """Largest external contour of a mask, in original image coordinates."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    epsilon = 0.004 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return [[int(round(x * scale_x)), int(round(y * scale_y))] for x, y in approx]


def group_lines_into_paragraphs(para_masks: List[np.ndarray],
                                iou_threshold: float = 0.5) -> List[List[int]]:
    """Union-find over predicted paragraph masks.

    Hi-SAM merges paragraph predictions by pairwise IoU > 0.5 with union-find;
    two lines whose predicted paragraphs agree belong to the same paragraph.
    """
    n = len(para_masks)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    areas = [m.sum() for m in para_masks]
    for i in range(n):
        if areas[i] == 0:
            continue
        for j in range(i + 1, n):
            if areas[j] == 0:
                continue
            inter = np.logical_and(para_masks[i], para_masks[j]).sum()
            if inter == 0:
                continue
            iou = inter / (areas[i] + areas[j] - inter)
            if iou > iou_threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


@torch.no_grad()
def predict(model, dataset, device, n_side: int, min_score: float,
            nms_iou: float, batch_prompts: int, max_images: Optional[int]) -> dict:
    """Grid-prompt inference -> HierText prediction annotations."""
    from Mobile_Hi_SAM.evaluation.metrics import binarize, to_instances
    from Mobile_Hi_SAM.evaluation.evaluate_hisam_metrics import grid_points, dedupe

    annotations = []
    total = min(len(dataset), max_images) if max_images else len(dataset)

    for idx in tqdm(range(total), desc="predicting"):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        nh, nw = sample["input_size"]
        H, W = sample["original_size"]

        embeddings = model.encode(image)
        points = grid_points((nh, nw), n_side, device)

        line_masks, para_masks, word_masks, scores = [], [], [], []
        for start in range(0, points.shape[0], batch_prompts):
            chunk = points[start:start + batch_prompts]
            n = chunk.shape[0]
            out = model.decode_prompts(
                embeddings, chunk.unsqueeze(1),
                torch.ones(n, 1, dtype=torch.int64, device=device),
            )
            iou = out["iou"].detach().cpu().numpy()
            for i in range(n):
                if float(iou[i, 1]) < min_score:      # line-token quality
                    continue
                line = binarize(out["line"][i, 0]).cpu().numpy().astype(bool)
                if not line.any():
                    continue
                line_masks.append(line)
                para_masks.append(binarize(out["para"][i, 0]).cpu().numpy().astype(bool))
                word_masks.append(binarize(out["word_hr"][i, 0]).cpu().numpy().astype(bool))
                scores.append(float(iou[i, 1]))

        keep = _nms_indices(line_masks, scores, nms_iou)
        line_masks = [line_masks[i] for i in keep]
        para_masks = [para_masks[i] for i in keep]
        word_masks = [word_masks[i] for i in keep]

        paragraphs = []
        if line_masks:
            # word_hr masks are 384 across the padded 1024 square, and the image
            # content occupies (nh, nw) of it, so a mask pixel maps to original
            # pixels by 1024/384 then W/nw (and H/nh).
            for group in group_lines_into_paragraphs(para_masks):
                lines_out = []
                for li in group:
                    words = []
                    for inst in to_instances(torch.from_numpy(word_masks[li]), min_area=8):
                        poly = mask_to_polygon(
                            inst,
                            scale_x=W / (nw * word_scale()),
                            scale_y=H / (nh * word_scale()),
                        )
                        if poly:
                            words.append({"vertices": poly, "text": ""})
                    if words:
                        lines_out.append({"words": words, "text": ""})
                if lines_out:
                    paragraphs.append({"lines": lines_out})

        annotations.append({"image_id": sample["image_id"], "paragraphs": paragraphs})

    return {"annotations": annotations}


def model_mask_size() -> int:
    return 256


def word_scale() -> float:
    """word_hr masks are 384 across the padded 1024 square."""
    return 384.0 / 1024.0


def _nms_indices(masks, scores, threshold):
    if not masks:
        return []
    from Mobile_Hi_SAM.evaluation.pq import bbox, iou as mask_iou
    order = np.argsort(-np.asarray(scores))
    kept, kept_boxes, kept_idx = [], [], []
    for i in order:
        box = bbox(masks[i])
        if box is None:
            continue
        if any(mask_iou(masks[i], k, box, kb) > threshold for k, kb in zip(kept, kept_boxes)):
            continue
        kept.append(masks[i])
        kept_boxes.append(box)
        kept_idx.append(int(i))
    return kept_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="run the official evaluator")
    p_score.add_argument("--gt", required=True, help="HierText ground-truth JSON")
    p_score.add_argument("--result", required=True, help="prediction JSON")
    p_score.add_argument("--output", default=None)
    p_score.add_argument("--no_lines", action="store_true")
    p_score.add_argument("--no_paragraphs", action="store_true")
    p_score.add_argument("--mask_stride", type=int, default=1)

    p_pred = sub.add_parser("predict", help="write a prediction JSON")
    p_pred.add_argument("--run_dir", required=True)
    p_pred.add_argument("--root", required=True)
    p_pred.add_argument("--split", default="validation")
    p_pred.add_argument("--out", required=True)
    p_pred.add_argument("--encoder_ckpt", default=None)
    p_pred.add_argument("--n_side", type=int, default=16)
    p_pred.add_argument("--min_score", type=float, default=0.4)
    p_pred.add_argument("--nms_iou", type=float, default=0.7)
    p_pred.add_argument("--batch_prompts", type=int, default=32)
    p_pred.add_argument("--max_images", type=int, default=None)
    p_pred.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.command == "score":
        score(args.gt, args.result, args.output,
              not args.no_lines, not args.no_paragraphs, args.mask_stride)
        return

    from Mobile_Hi_SAM.models.mobile_hisam_model import pick_device
    from Mobile_Hi_SAM.evaluation.evaluate_hisam_metrics import load_model
    from Mobile_Hi_SAM.train.hisam_hiertext_dataset import HiSAMHierTextDataset

    device = pick_device(args.device)
    model, _ = load_model(args.run_dir, device, args.encoder_ckpt, strict=True)
    dataset = HiSAMHierTextDataset(
        args.root, split=args.split, deterministic=True, augment=False
    )
    preds = predict(model, dataset, device, args.n_side, args.min_score,
                    args.nms_iou, args.batch_prompts, args.max_images)
    Path(args.out).write_text(json.dumps(preds))
    print(f"Wrote {len(preds['annotations'])} predictions to {args.out}")


if __name__ == "__main__":
    main()
