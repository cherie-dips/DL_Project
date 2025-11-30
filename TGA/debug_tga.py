import torch
from models.maskdino_wrapper import MaskDINOBackbone
from models.tga_adapter import TextGroupingAdapter
from data.hiertext_dataset import HierTextDataset
import torch.nn.functional as F

ds = HierTextDataset("hiertext/train", "hiertext/train_annotations.json", resize_max_side=1024)
img, mask = ds[0]
img = img.unsqueeze(0).cuda()

backbone = MaskDINOBackbone(weights_path="maskdino_r50.pth").cuda()
tga = TextGroupingAdapter(in_channels=256).cuda()

with torch.no_grad():
    feats = backbone(img)
    pred = tga(feats)
    pred = F.interpolate(pred, size=mask.shape[-2:], mode='bilinear')
    print("Pred stats:", pred.min().item(), pred.max().item(), pred.mean().item())

