# Lightweight Model for Hierarchical Text Segmentation on the HierText Dataset
Aryan Daga, Dipti Dhawade, Lipi Singhal, Tisha Bhavsar

## Introduction to Problem Statement

Scene text understanding is a core challenge in computer vision, especially when dealing with complex, real-world layouts. Our objective was to build a lightweight yet accurate model for Hierarchical Text Segmentation on the HierText dataset, which provides detailed annotations at word, line, and paragraph levels.

## The Hiertext dataset

The HierText dataset is the first scene-text dataset that provides hierarchical annotations at the word, line, and paragraph levels. It is also the only dataset that supports both text detection and layout analysis jointly.

It contains 11,639 high-resolution images (long side = 1600px), split into 8,281 train, 1,724 val, and 1,634 test images.

Images in the dataset were sourced from Open Images v6 using Google Cloud OCR to filter text-rich images. Only images with sufficient detected text, high OCR confidence, and mostly English content were kept.

The key characteristics of the dataset include:

- High text density: ~103.8 words per image, making it the densest public scene-text dataset (≈3× denser than TextOCR).
- True hierarchical labels: Words → grouped into lines → grouped into paragraphs using polygon masks.
- Uniform spatial distribution: Text appears across the full image, unlike other datasets where text is mostly centered.
- Rich variety: Natural scenes, documents, posters, signs, and curved/irregular text.

HierText uses Panoptic Quality (PQ) as the unified evaluation metric for:
- Word-level segmentation
- Line-level segmentation
- Paragraph-level (layout) segmentation

PQ is defined as:

$$
PQ = \frac{\sum \text{IoU(TP)}}{|\text{TP}| + 0.5|\text{FP}| + 0.5|\text{FN}|}
$$

Where:

- TP = true positive matches between predicted and ground-truth masks  
- FP = false positives  
- FN = false negatives  

PQ is chosen as the evaluation metric because:

- It jointly evaluates mask quality (IoU) and detection quality (F1).
- Works consistently across word, line, and paragraph masks.
- Captures errors that matter for text: missing/wrong pixels, over/under segmentation.

---

# 4. Text Grouping Adapter (TGA)

## Introduction

The Text Grouping Adapter (TGA), proposed by Ding et al. in *Text Grouping Adapter: Adapting Pre-Trained Text Detector for Layout Analysis (CVPR 2024)*, introduces a lightweight, plug-in module that converts any text detector into a full layout analysis system. It has about 6M additional parameters. TGA uses a pre-trained text detector as its backbone (like MaskDINO) and adds grouping logic.

## Architecture

TGA consists of three main components:
a. Text Instance Feature Assembling (TIFA)  
b. Group Mask Prediction (GMP)  
c. Affinity Matrix Prediction  

These components convert instance masks from the detector into paragraph-level grouping predictions.

### Backbone: MaskDINO-R50

MaskDINO is a transformer-based instance segmentation model. Given an image, the MaskDINO backbone outputs 1/4 to 1/32 of the original resolution to produce multiscale FPN features, and the instance masks become the input to the TGA.

## Text Instance Feature Assembling (TIFA)

### Step 1: Pixel Embedding Map
All FPN maps are resized to 1/8 scale and merged into a single feature map of size:



256 × (H/8) × (W/8)



### Step 2: Mask-Guided Feature Pooling
Each instance mask is resized to 1/8 scale and multiplied with the pixel embedding map.  
The masked features are summed to form a **256-dimensional embedding** for each instance.

This creates one vector per text instance.

## Group Mask Prediction (GMP)

GMP performs global grouping and predicts paragraph-level masks.

- Three transformer layers refine the instance embeddings  
- Embedding dimension: 256  
- Feedforward dimension: 512  
- Attention heads: 4  
- Number of layers: 3  

Each instance predicts a group mask at 1/4 resolution.  
All instances belonging to the same paragraph share the same ground-truth mask.  
This teaches the model the full paragraph structure, not just pairwise links.

## Affinity Matrix

The final step is to compute pairwise similarity between instances.

Each entry \( (i, j) \) is computed as:

$$
A(i, j) = F_i \cdot F_j
$$

Where \(F_i\) and \(F_j\) are the final 256-dimensional instance embeddings.

High affinity means the two instances belong to the same paragraph.  
This affinity matrix is thresholded and clustered to form paragraph groups.

## Loss Functions

TGA uses three losses:

### a. Dice Loss (Group Mask Loss)

Dice loss compares the predicted group mask with the ground-truth paragraph region.

$$
Dice = 1 - \frac{2|X \cap Y|}{|X| + |Y|}
$$

### b. Affinity Loss (Binary Cross Entropy)

Each pair of instances is labeled as “same paragraph” or “different paragraph.”

$$
L = - \left[ A \log(\hat{A}) + (1 - A) \log(1 - \hat{A}) \right]
$$

### c. Detection Loss

This is the original MaskDINO detection loss:

$$
L_{det} = L_{class} + L_{box} + L_{mask}
$$

### Final Loss Function

$$
L_{TGA} = L_{dice} + L_{affinity} + L_{det}
$$

---

## Running the Model on IITD HPC Server

### Environment Setup
1. Created a clean Conda environment  
2. Installed PyTorch and Detectron2 compatible with CUDA on the HPC  
3. Installed TGA dependencies  
4. Ensured version alignment  
A mismatch between PyTorch, Detectron2, and TGA configs caused errors until fixed.

### Preprocessing

All images were padded so that both height and width are divisible by 32.

Examples:

- 703 × 900 → 704 × 928  
- 768 × 1023 → 768 × 1024  

This ensures proper feature alignment.

### Main Scripts

#### train_net.py  
Runs training.



python train_net.py 
--num-gpus 1 
--config-file configs/TGA_MaskDINO_R50.yaml 
--resume



#### build_tga_model()  
Assembles MaskDINO + TIFA + GMP + Affinity head.

#### tga_loss.py  
Implements Dice loss, BCE affinity loss, and combined TGA loss.

#### evaluator_tga.py  
Evaluates PQ, F1, Dice, and IoU of our model on the HierText dataset.

## Backbone Issue

Training failed due to:

- Wrong MaskDINO checkpoint  
- Hundreds of missing or unexpected weight keys  
- All instance masks = zero  

TGA could not learn (PQ = 0)

### Fix Attempts and Debugging

We attempted:

a. Testing four different versions of MaskDINO-R50  
b. Rechecking all input–output size constraints  
c. Verifying mask resolution consistency  
d. Checking dataset annotations  

Despite these efforts, the model continued to produce empty instance masks.

## Conclusion

TGA represents a powerful and lightweight approach for converting a pre-trained text detector into a full layout analysis system. Its design is modular, its computational overhead is small, and the original paper demonstrates strong performance on hierarchical text datasets such as HierText. Under ideal conditions, TGA should have provided a reliable baseline for our paragraph-level segmentation task.

However, despite extensive attempts to reproduce the method, TGA did not train successfully in our setup. The backbone initialization failure repeatedly resulted in empty instance masks, which prevented TGA from learning meaningful embeddings. Even after correcting the checkpoint mismatch, training remained unstable and inconsistent across runs.

---

# OUR PROPOSED SOLUTION: MOBILE-HI-SAM

## 1. Logic Behind the Architecture

The objective of Mobile-Hi-SAM is to combine the efficiency of MobileSAM’s TinyViT encoder with the hierarchical text segmentation capabilities of Hi-SAM. Hi-SAM relies on a large ViT-H encoder and requires significant compute, making it impractical for deployment. Mobile-Hi-SAM addresses this by introducing:

- A lightweight encoder that reduces parameters and computational demands.  
- An adapter module to refine MobileSAM embeddings into text-aware features suitable for hierarchical decoding.  
- A modal aligner that produces learned sparse prompt tokens, enabling the network to infer hierarchical structures without explicit prompts.  
- A hierarchical decoder with independent heads for paragraph, line, and word segmentation to compensate for the reduced representational power of the lightweight encoder.  
- A multi-component loss (Dice, Focal, IoU) to ensure balanced optimization across all hierarchy levels.

This design maintains the hierarchical reasoning introduced by Hi-SAM while significantly lowering computational requirements.

## 2. Components of the Solution

### 2.1 Image Encoder (Frozen MobileSAM TinyViT)

Input:



[B, 3, 1024, 1024]



Output:



[B, 256, 64, 64]



### 2.2 Adapter Module



Conv2d(256→256, 1×1) + LayerNorm2d + GELU



Output:



[B, 256, 64, 64]



### 2.3 Modal Aligner (Implicit Prompt Generator)

a) Spatial Attention Generator → `[B, 12, 64, 64]`  
b) Weighted Feature Pooling → `[B, 12, 256]`  
c) Transformer Refinement → `[B, 12, 256]`  

### 2.4 Prompt Encoder

Produces:



[B, 256, 64, 64]



### 2.5 S-Decoder (SAM-Style Mask Decoder)

Performs two-way attention between prompt tokens and image embeddings.

### 2.6 Hierarchical Decoder (Three Independent Heads)

Each head:

- IoU token: `[1,256]`  
- Three mask tokens: `[3,256]`  
- Upscaling  
- Hypernetwork MLP  
- IoU prediction MLP  

Outputs:

- `[B, 3, 256, 256]` masks  
- `[B, 3]` IoU scores  

## 3. Novelty and Modifications Introduced

- Replacing ViT-H with TinyViT reduces model size by ~98%.  
- External adapter module bridges MobileSAM to hierarchical decoding.  
- Fully independent heads allow level-specific specialization.  
- Entire architecture redesigned to work with 256-dim features instead of 768–1024.

## 4. Hierarchical Loss Function

Each level uses:

| Component | Purpose | Weight |
|----------|---------|--------|
| Dice Loss | Region overlap | 1.0 |
| Focal Loss (α=0.25, γ=2.0) | Hard example focus | 20.0 |
| IoU Prediction Loss | Confidence calibration | 1.0 |

Total loss:

$$
L_{total} = L_{paragraph} + L_{line} + L_{word}
$$

## 5. Training Configuration

| Parameter | Value |
|-----------|--------|
| Epochs | 50 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Batch size | 4 |
| Resolution | 1024×1024 |
| Precision | Mixed (AMP) |
| Encoder | Frozen |
| Hardware | Single GPU |

## 6. Possible Modifications to Improve PQ and fgIoU

- Fine-tune part of the encoder  
- Increase focal loss impact  
- Cross-level consistency constraints  
- Larger upscaling modules  
- Boundary-aware losses  
- Hybrid transformer-CNN blocks  

## 7. Performance Comparison with Hi-SAM

| Model | Avg PQ | Avg fgIoU | Params | Deployability |
|--------|---------|-----------|----------|----------------|
| Hi-SAM-H | ~65% | ~75% | ~650M | TPU/GPU only |
| Mobile-Hi-SAM | 40.31% | 58.35% | 12.6M | Mobile / Edge |

Level-wise PQ:
- Word PQ: 40.41  
- Line PQ: 40.81  
- Layout PQ: 39.69  

## Interpretation

Mobile-Hi-SAM demonstrates that hierarchical text segmentation can be achieved efficiently without heavy compute requirements, making it suitable for practical deployments outside high-performance clusters.

---
