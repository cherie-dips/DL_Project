import torch
from models.maskdino_wrapper import MaskDINOBackbone
from data.hiertext_dataset import HierTextDataset

ds = HierTextDataset("hiertext/train", "hiertext/train_annotations.json", resize_max_side=1024)
img, mask = ds[0]
img = img.unsqueeze(0).cuda()

backbone = MaskDINOBackbone(weights_path="maskdino_r50.pth").cuda()

with torch.no_grad():
    feats = backbone(img)

print(type(feats))
if isinstance(feats, dict):
    for k,v in feats.items():
        print(k, v.shape)
else:
    print(feats.shape)

