"""
Train Mobile-Hi-SAM with Hierarchical Supervision V2
- Full HierarchicalLoss (Dice + Focal + IoU)
- Organized checkpoint structure
- Auto-validation after training
- Incremental training support
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/MOBILE-HI-SAM"
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM
from Mobile_Hi_SAM.models.hierarchical_decoder import HierarchicalDecoder
from Mobile_Hi_SAM.models.hierarchical_loss import HierarchicalLoss, SimplifiedHierarchicalLoss
from Mobile_Hi_SAM.train.hiertext_hierarchical_dataset import (
    HierTextHierarchicalDataset,
    collate_fn
)


class MobileHiSAMHierarchical(MobileHiSAM):
    """Extended Mobile-Hi-SAM with hierarchical decoder"""

    def __init__(self, checkpoint_path=None, img_size=1024, embed_dim=256):
        super().__init__(
            checkpoint_path=checkpoint_path,
            img_size=img_size,
            embed_dim=embed_dim,
            enable_hierarchical=False,
        )

        print("[MobileHiSAM] Adding HierarchicalDecoder...")
        self.hierarchical_decoder = HierarchicalDecoder(
            transformer_dim=embed_dim,
            transformer=self.mask_decoder.transformer,
            num_multimask_outputs=3,
        )

    def forward_hierarchical(self, batched_input):
        """Forward pass with hierarchical predictions"""
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input],
            dim=0
        )

        image_embeddings = self.image_encoder(input_images)
        adapted_embeddings = self.adapter(image_embeddings)

        all_para_masks = []
        all_para_iou = []
        all_line_masks = []
        all_line_iou = []
        all_word_masks = []
        all_word_iou = []

        for img_record, curr_emb in zip(batched_input, adapted_embeddings):
            if "point_coords" in img_record:
                points = (img_record["point_coords"], img_record["point_labels"])
            else:
                points = None

            sparse_prompt_embs, dense_prompt_embs = self.prompt_encoder(
                points=points,
                boxes=img_record.get("boxes", None),
                masks=img_record.get("mask_inputs", None)
            )

            para_masks, para_iou, line_masks, line_iou, word_masks, word_iou = \
                self.hierarchical_decoder(
                    image_embeddings=curr_emb.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_prompt_embs,
                    dense_prompt_embeddings=dense_prompt_embs,
                    multimask_output=False,
                )

            para_masks = self._postprocess_masks(para_masks, img_record)
            line_masks = self._postprocess_masks(line_masks, img_record)
            word_masks = self._postprocess_masks(word_masks, img_record)

            all_para_masks.append(para_masks)
            all_para_iou.append(para_iou)
            all_line_masks.append(line_masks)
            all_line_iou.append(line_iou)
            all_word_masks.append(word_masks)
            all_word_iou.append(word_iou)

        para_masks = torch.cat(all_para_masks, dim=0)
        para_iou = torch.cat(all_para_iou, dim=0)
        line_masks = torch.cat(all_line_masks, dim=0)
        line_iou = torch.cat(all_line_iou, dim=0)
        word_masks = torch.cat(all_word_masks, dim=0)
        word_iou = torch.cat(all_word_iou, dim=0)

        return para_masks, para_iou, line_masks, line_iou, word_masks, word_iou


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, use_amp=True):
    """Train for one epoch with mixed precision"""
    model.train()

    total_loss = 0
    loss_components = {'para': 0, 'line': 0, 'word': 0}
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move data to device
        for sample in batch:
            sample["image"] = sample["image"].to(device)
            sample["point_coords"] = sample["point_coords"].to(device)
            sample["point_labels"] = sample["point_labels"].to(device)
            sample["gt_para_mask"] = sample["gt_para_mask"].to(device)
            sample["gt_line_mask"] = sample["gt_line_mask"].to(device)
            sample["gt_word_mask"] = sample["gt_word_mask"].to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with autocast(enabled=use_amp):
            para_masks, para_iou, line_masks, line_iou, word_masks, word_iou = \
                model.forward_hierarchical(batch)

            gt_para = torch.stack([s["gt_para_mask"] for s in batch])
            gt_line = torch.stack([s["gt_line_mask"] for s in batch])
            gt_word = torch.stack([s["gt_word_mask"] for s in batch])

            # Compute loss
            if isinstance(criterion, SimplifiedHierarchicalLoss):
                loss, loss_dict = criterion(
                    para_masks, gt_para,
                    line_masks, gt_line,
                    word_masks, gt_word,
                )
            else:
                loss, loss_dict = criterion(
                    para_masks, para_iou, gt_para,
                    line_masks, line_iou, gt_line,
                    word_masks, word_iou, gt_word,
                )

        # Backward with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Logging
        total_loss += loss.item()
        if 'para' in loss_dict:
            loss_components['para'] += loss_dict['para']
            loss_components['line'] += loss_dict['line']
            loss_components['word'] += loss_dict['word']
        elif 'para_total' in loss_dict:
            loss_components['para'] += loss_dict['para_total']
            loss_components['line'] += loss_dict['line_total']
            loss_components['word'] += loss_dict['word_total']
        
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'para': f"{loss_components['para']/num_batches:.4f}",
            'line': f"{loss_components['line']/num_batches:.4f}",
            'word': f"{loss_components['word']/num_batches:.4f}",
        })

    avg_loss = total_loss / max(1, num_batches)
    avg_components = {k: v / max(1, num_batches) for k, v in loss_components.items()}

    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser(description="Train Mobile-Hi-SAM V2")
    parser.add_argument("--root", type=str, required=True, help="HierText dataset root")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (None=all)")
    parser.add_argument("--checkpoint_encoder", type=str, required=True, help="MobileSAM checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save_freq", type=int, default=5, help="Save every N epochs")
    parser.add_argument("--loss_type", type=str, default="full",
                        choices=["simplified", "full"],
                        help="Loss: simplified (Dice) or full (Dice+Focal+IoU)")
    
    # Loss weights
    parser.add_argument("--weight_para", type=float, default=1.0)
    parser.add_argument("--weight_line", type=float, default=1.0)
    parser.add_argument("--weight_word", type=float, default=1.0)
    parser.add_argument("--weight_focal", type=float, default=20.0, help="Focal loss weight")
    
    # Training options
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory with organized structure
    if args.run_name:
        output_dir = Path(f"hierarchical_training_{args.run_name}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"hierarchical_training_{timestamp}")
    
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Checkpoints: {checkpoint_dir}")

    # Save config
    config = vars(args)
    config['output_dir'] = str(output_dir)
    config['checkpoint_dir'] = str(checkpoint_dir)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load dataset
    print("\nLoading dataset...")
    dataset = HierTextHierarchicalDataset(
        args.root,
        split="train",
        max_items=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")

    # Load model
    print("\nLoading model...")
    model = MobileHiSAMHierarchical(
        checkpoint_path=args.checkpoint_encoder,
        img_size=1024,
        embed_dim=256,
    )
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    # Setup loss
    if args.loss_type == "simplified":
        print("\nUsing SimplifiedHierarchicalLoss (Dice only)")
        criterion = SimplifiedHierarchicalLoss(
            weight_para=args.weight_para,
            weight_line=args.weight_line,
            weight_word=args.weight_word,
        )
    else:
        print(f"\nUsing HierarchicalLoss (Dice + Focal[{args.weight_focal}] + IoU)")
        criterion = HierarchicalLoss(
            weight_para=args.weight_para,
            weight_line=args.weight_line,
            weight_word=args.weight_word,
            weight_dice=1.0,
            weight_focal=args.weight_focal,
            weight_iou=1.0,
        )

    # Setup optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Resume from checkpoint
    start_epoch = 1
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"{'='*60}\n")

    training_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        avg_loss, loss_components = train_epoch(
            model, loader, criterion, optimizer, scaler, device, epoch, args.use_amp
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log results
        epoch_results = {
            'epoch': epoch,
            'loss': avg_loss,
            'para_loss': loss_components['para'],
            'line_loss': loss_components['line'],
            'word_loss': loss_components['word'],
            'learning_rate': current_lr,
        }
        training_history.append(epoch_results)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Paragraph: {loss_components['para']:.6f}")
        print(f"  Line: {loss_components['line']:.6f}")
        print(f"  Word: {loss_components['word']:.6f}")
        print(f"  Learning Rate: {current_lr:.6e}")

        # Track best model
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'loss_components': loss_components,
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, best_path)
            print(f"  ✓ New best model saved! (loss: {best_loss:.6f})")

        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Output: {output_dir}")
    print(f"  Best checkpoint: {checkpoint_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")
    
    # Write marker file for validation
    with open(output_dir / ".training_complete", "w") as f:
        json.dump({
            'best_checkpoint': str(checkpoint_dir / 'best_model.pth'),
            'best_loss': best_loss,
            'total_epochs': args.epochs,
            'completed_at': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✓ Ready for validation!")
    print(f"Run: python ../evaluation/evaluate_hisam_metrics.py --run_dir {output_dir}")


if __name__ == "__main__":
    main()
