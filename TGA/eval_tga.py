import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.maskdino_wrapper import MaskDINOBackbone
from models.tga_adapter import TextGroupingAdapter
from data.hiertext_dataset import HierTextDataset
from metrics import compute_iou_batch, compute_dice_batch, compute_pixel_acc, compute_simple_pq
import json

def evaluate(model, backbone, loader, device):
    model.eval()
    total_loss = 0.0
    iou_list, dice_list, acc_list, pq_list = [], [], [], []
    with torch.no_grad():
        for imgs, masks in loader:
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

    n = len(iou_list)
    return {
        "loss": total_loss / n,
        "iou": sum(iou_list) / n,
        "dice": sum(dice_list) / n,
        "acc": sum(acc_list) / n,
        "pq": sum(pq_list) / n,
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading backbone...")
    backbone = MaskDINOBackbone(weights_path="maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth").to(device)

    print("Loading TGA model...")
    model = TextGroupingAdapter(in_channels=256).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))

    print("Loading test dataset...")
    test_ds = HierTextDataset("hiertext/test", "hiertext/test_annotations.json", resize_max_side=1024)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    print("Running evaluation...")
    stats = evaluate(model, backbone, test_loader, device)
    print("RESULTS:\n", stats)

    with open("checkpoints/best_model_test_metrics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved: checkpoints/best_model_test_metrics.json")

