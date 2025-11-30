import torch
from models.maskdino_wrapper import MaskDINOBackbone

m = MaskDINOBackbone(weights_path="maskdino_r50.pth")
x = torch.randn(1, 3, 512, 512)
out = m(x)

print("Output shape:", out.shape)
print("SUCCESS")

