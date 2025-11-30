#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from models.maskdino_wrapper import MaskDINOBackbone
from models.tga_adapter import TextGroupingAdapter
from data.hiertext_dataset import HierTextDataset
from metrics import compute_iou_batch, compute_dice_batch, compute_pixel_acc, compute_simple_pq

def train_one_epoch(model, backbone, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="train"):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            feats = backbone(imgs)
        preds = model(feats)
        preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        loss = F.binary_cross_entropy_with_logits(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, backbone, loader, device):
    model.eval()
    total_loss = 0.0
    iou_list, dice_list, acc_list, pq_list = [], [], [], []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="eval"):
            imgs, masks = imgs.to(device), masks.to(device)
            feats = backbone(imgs)
            preds = model(feats)
            preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.binary_cross_entropy_with_logits(preds, masks)
            total_loss += loss.item()

            iou_list.append(compute_iou_batch(preds, masks))
            dice_list.append(compute_dice_batch(preds, masks))
            acc_list.append(compute_pixel_acc(preds, masks))
            pq_list.append(compute_simple_pq(preds, masks))
    n = len(iou_list) if len(iou_list) else 1
    stats = {
        "loss": total_loss / len(loader),
        "iou": sum(iou_list) / n,
        "dice": sum(dice_list) / n,
        "acc": sum(acc_list) / n,
        "pq": sum(pq_list) / n
    }
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", default="hiertext/train_annotations.json")
    parser.add_argument("--val-ann", default="hiertext/val_annotations.json")
    parser.add_argument("--test-ann", default="hiertext/test_annotations.json")
    parser.add_argument("--img-train", default="hiertext/train")
    parser.add_argument("--img-val", default="hiertext/validation")
    parser.add_argument("--img-test", default="hiertext/test")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resize", type=int, default=1024, help="max side; set 0 to disable")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights", default="maskdino_r50.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Path to model checkpoint to resume from")

    args = parser.parse_args()

    device = args.device
    print("Device:", device)
    print("Loading MaskDINO weights from:", args.weights)
    backbone = MaskDINOBackbone(weights_path=args.weights).to(device)
    model = TextGroupingAdapter(in_channels=256).to(device)
    if args.resume:
        print("Resuming from checkpoint:", args.resume)
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)


    train_ds = HierTextDataset(args.img_train, args.train_ann, resize_max_side=(args.resize or None))
    val_ds = HierTextDataset(args.img_val, args.val_ann, resize_max_side=(args.resize or None))
    test_ds = None
    if os.path.exists(args.test_ann):
        test_ds = HierTextDataset(args.img_test, args.test_ann, resize_max_side=(args.resize or None))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if test_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = 1e9

    for epoch in range(args.epochs):
        print(f"\n===== EPOCH {epoch} =====")
        train_loss = train_one_epoch(model, backbone, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.6f}")

        val_stats = evaluate(model, backbone, val_loader, device)
        print(f"VAL | loss={val_stats['loss']:.6f} IoU={val_stats['iou']:.4f} Dice={val_stats['dice']:.4f} Acc={val_stats['acc']:.4f} PQ={val_stats['pq']:.4f}")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Saved best model -> checkpoints/best_model.pth")

        torch.save(model.state_dict(), f"checkpoints/tga_epoch{epoch}.pth")

    # final test evaluation
    if test_loader is not None:
        print("\n===== TEST EVAL =====")
        test_stats = evaluate(model, backbone, test_loader, device)
        print(f"TEST | loss={test_stats['loss']:.6f} IoU={test_stats['iou']:.4f} Dice={test_stats['dice']:.4f} Acc={test_stats['acc']:.4f} PQ={test_stats['pq']:.4f}")
        # save final test stats to file
        with open("checkpoints/test_stats.txt", "w") as fh:
            fh.write(json.dumps(test_stats, indent=2))

if __name__ == "__main__":
    main()

