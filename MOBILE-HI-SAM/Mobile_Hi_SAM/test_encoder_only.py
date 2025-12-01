import sys
import os
import torch
from PIL import Image
from torchvision import transforms

# ------------------------------------------------------------
# Force Python to treat project root as importable
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

print("Python path:", sys.path)

from Mobile_Hi_SAM.models.mobile_encoder import MobileSAMEncoder
encoder = MobileSAMEncoder()
encoder.cpu()

img = Image.new("RGB", (1024,1024), (128,128,128))
img = transforms.ToTensor()(img).unsqueeze(0)

with torch.no_grad():
    out = encoder(img)
print("output:", out.shape)

