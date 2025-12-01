# Mobile-Hi-SAM (integration)

This repo integrates MobileSAM encoder with Hi-SAM decoders to produce a lightweight, promptable segmentation model.

Structure:
- Mobile_Hi_SAM/          <-- integrated package
  - models/               (integration code)
  - adapters/             (adapter layer)
  - configs/
  - train/
  - sample_images/
  - sanity_forward.py

Quick local sanity test (from project root `Mobile-Hi-SAM/`):

1. Place 1-2 test images in `Mobile_Hi_SAM/sample_images/`.
2. Ensure MobileSAM checkpoint exists:
   - recommended: `MobileSAM/weights/mobile_sam.pt`
3. Run sanity forward:

