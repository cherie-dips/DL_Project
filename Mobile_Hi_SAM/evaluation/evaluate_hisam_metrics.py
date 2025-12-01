"""
Complete Hi-SAM Metrics Evaluation
Computes: fgIOU, PQ, F-score, Precision, Recall for Word/Line/Layout
Matches Table 16/17 format from Hi-SAM paper
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from pathlib import Path

PROJECT_ROOT = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/MOBILE-HI-SAM"
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM
from Mobile_Hi_SAM.models.hierarchical_decoder import HierarchicalDecoder
from Mobile_Hi_SAM.train.hiertext_hierarchical_dataset import (
    HierTextHierarchicalDataset,
    collate_fn
)


class MobileHiSAMHierarchical(MobileHiSAM):
    """Same wrapper as training"""
    def __init__(self, checkpoint_path=None, img_size=1024, embed_dim=256):
        super().__init__(
            checkpoint_path=checkpoint_path,
            img_size=img_size,
            embed_dim=embed_dim,
            enable_hierarchical=False,
        )
        self.hierarchical_decoder = HierarchicalDecoder(
            transformer_dim=embed_dim,
            transformer=self.mask_decoder.transformer,
            num_multimask_outputs=3,
        )

    def forward_hierarchical(self, batched_input):
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        image_embeddings = self.image_encoder(input_images)
        adapted_embeddings = self.adapter(image_embeddings)

        all_para_masks, all_para_iou = [], []
        all_line_masks, all_line_iou = [], []
        all_word_masks, all_word_iou = [], []

        for img_record, curr_emb in zip(batched_input, adapted_embeddings):
            points = (img_record["point_coords"], img_record["point_labels"]) if "point_coords" in img_record else None
            sparse_emb, dense_emb = self.prompt_encoder(
                points=points, boxes=img_record.get("boxes", None), masks=img_record.get("mask_inputs", None)
            )

            para_masks, para_iou, line_masks, line_iou, word_masks, word_iou = \
                self.hierarchical_decoder(
                    image_embeddings=curr_emb.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )

            all_para_masks.append(self._postprocess_masks(para_masks, img_record))
            all_para_iou.append(para_iou)
            all_line_masks.append(self._postprocess_masks(line_masks, img_record))
            all_line_iou.append(line_iou)
            all_word_masks.append(self._postprocess_masks(word_masks, img_record))
            all_word_iou.append(word_iou)

        return (torch.cat(all_para_masks, dim=0), torch.cat(all_para_iou, dim=0),
                torch.cat(all_line_masks, dim=0), torch.cat(all_line_iou, dim=0),
                torch.cat(all_word_masks, dim=0), torch.cat(all_word_iou, dim=0))


def compute_fg_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute foreground IoU (fgIOU) - intersection over union for foreground pixels only
    """
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    gt_binary = (gt_mask > threshold).float()
    
    # Only compute IoU where GT has foreground
    fg_mask = gt_binary > 0
    if fg_mask.sum() == 0:
        return 1.0 if pred_binary.sum() == 0 else 0.0
    
    pred_fg = pred_binary * fg_mask
    gt_fg = gt_binary * fg_mask
    
    intersection = (pred_fg * gt_fg).sum()
    union = (pred_fg + gt_fg).clamp(0, 1).sum()
    
    iou = intersection / (union + 1e-6)
    return iou.item()


def compute_metrics_per_image(pred_mask, gt_mask, iou_threshold=0.5):
    """
    Compute all metrics for a single image:
    - fgIOU: Foreground IoU
    - PQ: Panoptic Quality (SQ * RQ)
    - Precision, Recall, F-score
    
    Returns dict with all metrics
    """
    pred_binary = (torch.sigmoid(pred_mask) > iou_threshold).float()
    gt_binary = (gt_mask > iou_threshold).float()
    
    # Flatten
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    # Basic metrics
    tp = (pred_flat * gt_flat).sum().item()
    fp = (pred_flat * (1 - gt_flat)).sum().item()
    fn = ((1 - pred_flat) * gt_flat).sum().item()
    
    # Precision, Recall, F-score
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # fgIOU
    fg_iou = compute_fg_iou(pred_mask, gt_mask, threshold=iou_threshold)
    
    # PQ (Panoptic Quality)
    # SQ = segmentation quality (IoU when matched)
    # RQ = recognition quality (TP / (TP + 0.5*FP + 0.5*FN))
    sq = fg_iou
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    pq = sq * rq
    
    return {
        'fgIOU': fg_iou,
        'PQ': pq,
        'P': precision,
        'R': recall,
        'F': f_score,
    }


class HiSAMEvaluator:
    """Evaluates using Hi-SAM metrics"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_dataset(self, dataloader):
        """Evaluate entire dataset"""
        
        # Metrics for each level
        word_metrics = {'fgIOU': [], 'PQ': [], 'P': [], 'R': [], 'F': []}
        line_metrics = {'fgIOU': [], 'PQ': [], 'P': [], 'R': [], 'F': []}
        layout_metrics = {'fgIOU': [], 'PQ': [], 'P': [], 'R': [], 'F': []}
        
        print("\nEvaluating...")
        for batch in tqdm(dataloader):
            # Move to device
            for sample in batch:
                sample["image"] = sample["image"].to(self.device)
                sample["point_coords"] = sample["point_coords"].to(self.device)
                sample["point_labels"] = sample["point_labels"].to(self.device)
                sample["gt_para_mask"] = sample["gt_para_mask"].to(self.device)
                sample["gt_line_mask"] = sample["gt_line_mask"].to(self.device)
                sample["gt_word_mask"] = sample["gt_word_mask"].to(self.device)
            
            # Forward
            para_masks, para_iou, line_masks, line_iou, word_masks, word_iou = \
                self.model.forward_hierarchical(batch)
            
            gt_para = torch.stack([s["gt_para_mask"] for s in batch])
            gt_line = torch.stack([s["gt_line_mask"] for s in batch])
            gt_word = torch.stack([s["gt_word_mask"] for s in batch])
            
            # Compute metrics for each sample
            batch_size = para_masks.shape[0]
            for i in range(batch_size):
                # Word metrics
                word_m = compute_metrics_per_image(word_masks[i, 0], gt_word[i, 0])
                for k, v in word_m.items():
                    word_metrics[k].append(v)
                
                # Line metrics
                line_m = compute_metrics_per_image(line_masks[i, 0], gt_line[i, 0])
                for k, v in line_m.items():
                    line_metrics[k].append(v)
                
                # Layout (paragraph) metrics
                layout_m = compute_metrics_per_image(para_masks[i, 0], gt_para[i, 0])
                for k, v in layout_m.items():
                    layout_metrics[k].append(v)
        
        # Average metrics
        results = {
            'Word': {k: np.mean(v) * 100 for k, v in word_metrics.items()},  # Convert to percentage
            'Text-line': {k: np.mean(v) * 100 for k, v in line_metrics.items()},
            'Layout Analysis': {k: np.mean(v) * 100 for k, v in layout_metrics.items()},
        }
        
        # Add column name (T = IoU threshold, always 0.5 in our case)
        for level in results:
            results[level]['T'] = 0.5
        
        return results


def load_checkpoint(run_dir, device):
    """Load model from training run directory"""
    run_dir = Path(run_dir)
    
    # Find best checkpoint
    checkpoint_path = run_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        # Try alternate locations
        checkpoint_path = run_dir / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        encoder_ckpt = config.get('checkpoint_encoder', 
                                  f"{PROJECT_ROOT}/MobileSAM/weights/mobile_sam.pt")
    else:
        encoder_ckpt = f"{PROJECT_ROOT}/MobileSAM/weights/mobile_sam.pt"
    
    # Create model
    model = MobileHiSAMHierarchical(
        checkpoint_path=encoder_ckpt,
        img_size=1024,
        embed_dim=256,
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print("✓ Checkpoint loaded successfully")
    return model


def print_results_table(results):
    """Print results in Hi-SAM paper format (Table 16/17)"""
    print("\n" + "="*80)
    print("RESULTS (Hi-SAM Format)")
    print("="*80)
    print(f"{'Method':<25} | {'fgIOU':<8} | {'PQ':<8} | {'F-score':<8} | {'P':<8} | {'R':<8} | {'T':<5}")
    print("-"*80)
    
    for level in ['Word', 'Text-line', 'Layout Analysis']:
        m = results[level]
        print(f"{level:<25} | {m['fgIOU']:>7.2f} | {m['PQ']:>7.2f} | {m['F']:>8.2f} | "
              f"{m['P']:>7.2f} | {m['R']:>7.2f} | {m['T']:.1f}")
    
    print("="*80)
    print("\nNOTE: All values are percentages")
    print("Target Hi-SAM-H metrics (validation set):")
    print("  Word:     fgIOU=74.86, PQ=64.63, F=84.08")
    print("  Text-line: fgIOU=74.86, PQ=69.58, F=90.06")
    print("  Layout:   fgIOU=75.45, PQ=60.42, F=78.16")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help='Training run directory')
    parser.add_argument('--root', type=str, 
                       default='/scratch/hpc/visitor/px151.visitor/DL/DL_Project/hiertext',
                       help='HierText dataset root')
    parser.add_argument('--split', default='val', help='Dataset split')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("⚠️  Warning: Running on CPU. Validation will be slow!")
        print("   Consider using GPU or reducing --max_samples")
    
    # Load model
    model = load_checkpoint(args.run_dir, device)
    
    # Load dataset
    print(f"\n📊 Loading {args.split} dataset...")
    dataset = HierTextHierarchicalDataset(
        args.root,
        split=args.split,
        max_items=args.max_samples,
        img_size=1024
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if not torch.cuda.is_available() else 4,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    evaluator = HiSAMEvaluator(model, device)
    results = evaluator.evaluate_dataset(dataloader)
    
    # Print results
    print_results_table(results)
    
    # Save results
    run_dir = Path(args.run_dir)
    output_path = run_dir / "validation_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Summary
    avg_pq = np.mean([results[l]['PQ'] for l in ['Word', 'Text-line', 'Layout Analysis']])
    avg_fgiou = np.mean([results[l]['fgIOU'] for l in ['Word', 'Text-line', 'Layout Analysis']])
    
    print(f"\n📈 SUMMARY:")
    print(f"   Average PQ:    {avg_pq:.2f}%")
    print(f"   Average fgIOU: {avg_fgiou:.2f}%")
    print(f"\n   Target (Hi-SAM-H): PQ~65%, fgIOU~75%")
    
    gap_pq = 65 - avg_pq
    gap_fgiou = 75 - avg_fgiou
    print(f"\n   Gap to Hi-SAM: PQ {gap_pq:+.2f}%, fgIOU {gap_fgiou:+.2f}%")


if __name__ == "__main__":
    main()
