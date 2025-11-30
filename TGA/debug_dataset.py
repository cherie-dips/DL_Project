from data.hiertext_dataset import HierTextDataset
import matplotlib.pyplot as plt
import torch

ds = HierTextDataset("hiertext/train", "hiertext/train_annotations.json", resize_max_side=1024)

img, mask = ds[0]

print("Image shape:", img.shape)
print("Mask sum:", mask.sum())

plt.subplot(1,2,1)
plt.imshow(img.permute(1,2,0))
plt.subplot(1,2,2)
plt.imshow(mask[0])
plt.show()

