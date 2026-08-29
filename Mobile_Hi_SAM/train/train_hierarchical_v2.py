"""
Train Mobile-Hi-SAM (MobileSAM encoder + Hi-SAM H-Decoder) on HierText.

The model wrapper lives in models/mobile_hisam_model.py and is shared with the
evaluation script, so training and evaluation cannot drift apart.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Resolve the package root from this file so the script is not tied to one machine.
PROJECT_ROOT = os.environ.get(
    "MOBILE_HISAM_ROOT",
    str(Path(__file__).resolve().parents[2]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import (  # noqa: E402
    MobileHiSAM,
    amp_supported,
    assert_no_dead_parameters,
    pick_device,
)
from Mobile_Hi_SAM.models.hierarchical_loss import HierarchicalLoss  # noqa: E402
from Mobile_Hi_SAM.train.hiertext_hierarchical_dataset import (  # noqa: E402
    HierTextHierarchicalDataset,
    collate_fn,
)

MASK_KEYS = (
    "image",
    "point_coords",
    "point_labels",
    "gt_word_mask",
    "gt_word_mask_lr",
    "gt_line_mask",
    "gt_para_mask",
)
OPTIONAL_KEYS = ("gt_text_mask", "gt_text_mask_lr")


def to_device(batch, device):
    for k in MASK_KEYS:
        batch[k] = batch[k].to(device, non_blocking=True)
    for k in OPTIONAL_KEYS:
        if k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def run_epoch(model, loader, criterion, optimizer, scaler, device, epoch, use_amp,
              train=True, log_every=0, total_epochs=None):
    model.train(train)
    torch.set_grad_enabled(train)

    totals, count = {}, 0
    checked_dead = not train  # only meaningful on a training step

    desc = f"{'Epoch' if train else 'Val  '} {epoch}"
    # A tqdm bar redirected to a file is just carriage-return noise; the periodic
    # lines below are what a background run actually needs.
    interactive = sys.stderr.isatty()
    pbar = tqdm(loader, desc=desc, leave=False, disable=not interactive)
    n_steps = len(loader)
    window, t_window = {}, time.time()

    for batch in pbar:
        batch = to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=False)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model.forward_hierarchical(batch)
            loss, logs = criterion(outputs, batch)

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            # A trainable parameter with no gradient is a module that was built
            # and optimised but never reached by the forward pass.
            if not checked_dead:
                assert_no_dead_parameters(model)
                checked_dead = True

            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + v
            window[k] = window.get(k, 0.0) + v
        count += 1

        # Periodic progress: averaged over the window since the last line, so a
        # long epoch reports movement instead of going quiet.
        if log_every and count % log_every == 0:
            elapsed = time.time() - t_window
            avg = {k: v / log_every for k, v in window.items()}
            tag = "train" if train else "val  "
            suffix = f"/{total_epochs}" if total_epochs else ""
            print(f"  [{tag}] epoch {epoch}{suffix}  step {count}/{n_steps}  "
                  f"loss={avg['total']:.4f}  word={avg['word']:.4f} "
                  f"line={avg['line']:.4f} para={avg['para']:.4f}  "
                  f"({elapsed / log_every:.2f}s/step)", flush=True)
            window, t_window = {}, time.time()

        pbar.set_postfix({
            "loss": f"{logs['total']:.4f}",
            "w": f"{logs['word']:.3f}",
            "l": f"{logs['line']:.3f}",
            "p": f"{logs['para']:.3f}",
        })

    torch.set_grad_enabled(True)
    return {k: v / max(1, count) for k, v in totals.items()}


def build_loader(args, split, shuffle, deterministic):
    stroke_dir = (
        os.path.join(args.stroke_gt_root, f"{split}_gt")
        if args.enable_s_decoder and args.stroke_gt_root
        else None
    )
    dataset = HierTextHierarchicalDataset(
        args.root,
        split=split,
        max_items=args.max_samples if split == "train" else args.max_val_samples,
        img_size=1024,
        samples_per_image=args.samples_per_image if split == "train" else 1,
        deterministic=deterministic,
        include_text_mask=args.enable_s_decoder,
        stroke_gt_dir=stroke_dir,
        gt_dir=args.gt_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" or torch.cuda.is_available()),
        drop_last=shuffle,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def main():
    parser = argparse.ArgumentParser(description="Train Mobile-Hi-SAM")
    parser.add_argument("--root", required=True, help="HierText dataset root")
    parser.add_argument("--gt_dir", default="gt",
                        help="annotation subdirectory under --root (default: gt)")
    parser.add_argument("--checkpoint_encoder", required=True, help="MobileSAM checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=500)
    parser.add_argument("--samples_per_image", type=int, default=4,
                        help="nested instances drawn per image per epoch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=25,
                        help="print a progress line every N steps (0 to disable)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_amp", action="store_true",
                        help="mixed precision; CUDA only, ignored elsewhere")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no_val", action="store_true",
                        help="disable validation (then is_best falls back to train loss)")

    # architecture
    parser.add_argument("--enable_s_decoder", action="store_true",
                        help="also build Hi-SAM's S-Decoder (stroke-level branch); "
                             "requires --stroke_gt_root")
    parser.add_argument("--stroke_gt_root", default=None,
                        help="root holding Hi-SAM's contributed stroke masks; the "
                             "split subdirectory is <root>/<split>_gt")
    parser.add_argument("--transformer_mlp_dim", type=int, default=2048)
    parser.add_argument("--adapter", choices=["linear", "context"], default="linear",
                        help="linear: 1x1 conv (receptive field 1 cell). "
                             "context: dilated 3x3 stack (receptive field 21 cells), "
                             "for testing whether the paragraph level is context-starved.")
    parser.add_argument("--unfreeze_encoder", action="store_true",
                        help="fine-tune TinyViT as well. Hi-SAM trains its whole "
                             "encoder, so freezing ours is a second variable on top "
                             "of the encoder swap.")
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="learning rate for the encoder (default: lr/10)")

    # loss weights
    parser.add_argument("--weight_word", type=float, default=1.0)
    parser.add_argument("--weight_line", type=float, default=1.0)
    parser.add_argument("--weight_para", type=float, default=0.5,
                        help="Hi-SAM weights the paragraph level at 0.5")
    parser.add_argument("--weight_focal", type=float, default=20.0)
    parser.add_argument("--weight_iou", type=float, default=1.0)
    parser.add_argument("--weight_containment", type=float, default=0.0)
    parser.add_argument("--use_tversky", action="store_true")
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)

    args = parser.parse_args()

    if args.enable_s_decoder and not args.stroke_gt_root:
        parser.error(
            "--enable_s_decoder needs --stroke_gt_root. The S-Decoder predicts "
            "stroke-level text, and HierText has no stroke annotations of its "
            "own; Hi-SAM's authors contributed them as a separate download (see "
            "their datasets/data_preparation.md). Filled word polygons are not a "
            "valid substitute - training on them would not reproduce Hi-SAM's "
            "fgIOU. Omit --enable_s_decoder to train the H-Decoder alone."
        )

    device = pick_device(args.device)
    if args.use_amp and not amp_supported(device):
        print(f"[warn] --use_amp ignored: loss scaling is not wired up for {device.type}")
        args.use_amp = False
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("GPU: Apple Metal (MPS)")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"hierarchical_training_{run_name}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    config = vars(args) | {"output_dir": str(output_dir)}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nLoading datasets...")
    train_set, train_loader = build_loader(args, "train", shuffle=True, deterministic=False)
    val_loader = None
    if not args.no_val:
        try:
            _, val_loader = build_loader(args, "validation", shuffle=False, deterministic=True)
        except FileNotFoundError:
            print("[warn] no validation split found; falling back to train loss for is_best")

    print("\nLoading model...")
    model = MobileHiSAM(
        checkpoint_path=args.checkpoint_encoder,
        img_size=1024,
        embed_dim=256,
        enable_hierarchical=True,
        enable_s_decoder=args.enable_s_decoder,
        transformer_mlp_dim=args.transformer_mlp_dim,
        freeze_encoder=not args.unfreeze_encoder,
        adapter=args.adapter,
    ).to(device)
    print(model.parameter_report())

    criterion = HierarchicalLoss(
        weight_word=args.weight_word,
        weight_line=args.weight_line,
        weight_para=args.weight_para,
        weight_focal=args.weight_focal,
        weight_iou=args.weight_iou,
        weight_containment=args.weight_containment,
        use_tversky=args.use_tversky,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
    )

    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.lr, args.encoder_lr),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.use_amp)

    # A rough budget so a laptop run is not started blind.
    per_image = {"cuda": 0.05, "mps": 0.18, "cpu": 1.2}.get(device.type, 0.5)
    n_train = len(train_set)
    epoch_min = n_train * per_image / 60
    print(f"\n{n_train:,} training samples "
          f"({train_set._n_records():,} images x {args.samples_per_image})")
    print(f"~{epoch_min:.0f} min/epoch estimated on {device.type}; "
          f"{args.epochs} epochs ~ {epoch_min * args.epochs / 60:.1f} h")
    if val_loader is not None:
        print(f"validating on {len(val_loader.dataset):,} images each epoch")

    start_epoch, best_metric = 1, float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint does not match the model.\n  missing={list(missing)[:10]}"
                f"\n  unexpected={list(unexpected)[:10]}"
            )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", float("inf"))

    print(f"\n{'='*64}\nTraining from epoch {start_epoch} to {args.epochs}\n{'='*64}")
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_logs = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.use_amp, train=True,
            log_every=args.log_every, total_epochs=args.epochs,
        )

        val_logs = None
        if val_loader is not None:
            with torch.no_grad():
                val_logs = run_epoch(
                    model, val_loader, criterion, optimizer, scaler,
                    device, epoch, args.use_amp, train=False,
                )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Selecting on train loss just picks the last epoch of a monotone curve.
        selection = val_logs["total"] if val_logs else train_logs["total"]
        is_best = selection < best_metric
        if is_best:
            best_metric = selection

        record = {
            "epoch": epoch,
            "lr": lr,
            "train": train_logs,
            "val": val_logs,
            "selected_on": "val" if val_logs else "train",
        }
        history.append(record)

        print(f"\nEpoch {epoch}/{args.epochs}  lr={lr:.3e}")
        print(f"  train  total={train_logs['total']:.4f}  "
              f"word={train_logs['word']:.4f} line={train_logs['line']:.4f} para={train_logs['para']:.4f}")
        if val_logs:
            print(f"  val    total={val_logs['total']:.4f}  "
                  f"word={val_logs['word']:.4f} line={val_logs['line']:.4f} para={val_logs['para']:.4f}")
        if is_best:
            print(f"  * new best ({record['selected_on']} loss {best_metric:.4f})")

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_logs": train_logs,
            "val_logs": val_logs,
            "best_metric": best_metric,
            "config": config,
        }
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(payload, checkpoint_dir / f"epoch_{epoch:02d}.pth")
        if is_best:
            torch.save(payload, checkpoint_dir / "best_model.pth")

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*64}\nDone. Best {history[-1]['selected_on']} loss: {best_metric:.6f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best_model.pth'}\n{'='*64}")


if __name__ == "__main__":
    main()
