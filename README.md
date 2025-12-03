# Lightweight Model for Hierarchical Text Segmentation on the HierText Dataset

**Authors:** Aryan Daga, Dipti Dhawade, Lipi Singhal, Tisha Bhavsar

-----

## 1\. Introduction to Problem Statement

Scene text understanding is a core challenge in computer vision, especially when dealing with complex, real-world layouts. Our objective was to build a lightweight yet accurate model for Hierarchical Text Segmentation on the HierText dataset, which provides detailed annotations at word, line, and paragraph levels.

## 2\. The Hiertext Dataset

The HierText dataset is the first scene-text dataset that provides hierarchical annotations at the word, line, and paragraph levels. It is also the only dataset that supports both text detection and layout analysis jointly.

It contains 11,639 high-resolution images (long side = 1600px), split into 8,281 train, 1,724 val, and 1,634 test images.

Images in the dataset were sourced from Open Images v6 using Google Cloud OCR to filter text-rich images. Only images with sufficient detected text, high OCR confidence, and mostly English content were kept.

**The key characteristics of the dataset include:**

  * **High text density:** \~103.8 words per image, making it the densest public scene-text dataset (≈3× denser than TextOCR).
  * **True hierarchical labels:** Words → grouped into lines → grouped into paragraphs using polygon masks.
  * **Uniform spatial distribution:** Text appears across the full image, unlike other datasets where text is mostly centered.
  * **Rich variety:** Natural scenes, documents, posters, signs, and curved/irregular text.

### Evaluation Metric: Panoptic Quality (PQ)

HierText uses Panoptic Quality (PQ) as the unified evaluation metric for:

1.  Word-level segmentation
2.  Line-level segmentation
3.  Paragraph-level (layout) segmentation

PQ is defined as:

$$PQ = \frac{\sum IoU(TP)}{|TP| + 0.5|FP| + 0.5|FN|}$$

Where:

  * **TP** = true positive matches between predicted and ground-truth masks
  * **FP** = false positives
  * **FN** = false negatives

PQ is chosen as the evaluation metric because:

  * It jointly evaluates mask quality (IoU) and detection quality (F1).
  * Works consistently across word, line, and paragraph masks.
  * Captures errors that matter for text: missing/wrong pixels, over/under segmentation.

-----

## 3\. Existing Solutions

### 3.1 Unified Detector

The Unified Detector is an end-to-end model designed to perform scene text detection and layout analysis simultaneously, unlike traditional systems that treat these as separate tasks. It is built on top of MaX-DeepLab, a transformer-based panoptic segmentation framework.

The model predicts:

  * Text detection masks (words or lines)
  * An affinity matrix that groups detections into paragraphs

<img width="559" height="677" alt="image" src="https://github.com/user-attachments/assets/d074ed35-72db-4fb5-91a1-aeb77c349da2" />


This allows the system to localize text and recover layout structure in one forward pass. The unified detector works as follows:

1.  **Object Queries + Pixel Features:** The model uses learnable object queries (N of them) that interact with pixel features through a dual-path transformer inside MaX-DeepLab. This enables the model to localize arbitrarily shaped text regions as segmentation masks.
2.  **Text Detection Branch:**
      * Produces N soft masks (each representing a potential text instance).
      * Produces a textness score for each query to filter out non-text.
3.  **Layout Branch (Affinity Matrix):**
      * Uses a 3-layer multi-head self-attention module to produce cluster embeddings.
      * Computes pairwise similarity between embeddings to create an N×N affinity matrix, where High values ⇒ same paragraph, Low values ⇒ different paragraphs.
      * A union-find algorithm merges instances into paragraph groups.
4.  **End-to-End Optimization:** Both detection and layout branches are optimized jointly using a combination of PQ-based detection loss, Affinity (binary cross-entropy) loss, and Segmentation + instance discrimination losses.

**Performance on HierText:**

  * Text Detection PQ: 62.23
  * Layout Analysis PQ: 53.60

### 3.2 Text Grouping Adapter (TGA)

#### Introduction

The Text Grouping Adapter (TGA), proposed by Ding et al. in *Text Grouping Adapter: Adapting Pre-Trained Text Detector for Layout Analysis (CVPR 2024)*, introduces a lightweight, plug-in module that converts any text detector into a full layout analysis system. It has about 6M additional parameters. TGA uses a pre-trained text detector as its backbone (like MaskDINO) and adds grouping logic.

#### Architecture

<img width="1297" height="596" alt="image" src="https://github.com/user-attachments/assets/22d60010-dd2d-444b-ae05-bf359f0c048e" />

TGA consists of three main components which convert instance masks from the detector into paragraph-level grouping predictions.

[Image of Text Grouping Adapter architecture diagram]

**1. Backbone: MaskDINO-R50**
MaskDINO is a transformer-based instance segmentation model. Given an image, the MaskDINO backbone outputs 1/4 to 1/32 of the original resolution to produce multiscale FPN features and the instance masks become the input to the TGA.

**2. Text Instance Feature Assembling (TIFA)**

  * **Step 1 (Pixel Embedding Map):** All FPN maps are resized to 1/8 scale and merged into a single feature map of size 256 x (H/8) x (W/8).
  * **Step 2 (Mask-Guided Feature Pooling):** Each instance mask is resized to 1/8 scale and multiplied with the pixel embedding map. The masked features are summed to form a 256-dimensional embedding for each instance. This creates one vector per text instance.

**3. Group Mask Prediction (GMP)**
GMP performs global grouping and predicts paragraph-level masks.

  * Three transformer layers refine the instance embeddings (Embedding dim: 256, Feedforward dim: 512, Attention heads: 4).
  * Each instance predicts a group mask at 1/4 resolution.
  * All instances belonging to the same paragraph share the same ground-truth mask. This teaches the model the full paragraph structure, not just pairwise links.

**4. Affinity Matrix**
The final step is to compute pairwise similarity between instances. Each entry $(i,j)$ is computed as:
$$A(i,j) = dot(F_i, F_j)$$
Where $F_i$ and $F_j$ are the final 256-dimensional instance embeddings. High affinity means the two instances belong to the same paragraph. This affinity matrix is thresholded and clustered to form paragraph groups.

#### Loss Functions

TGA uses three losses:

1.  **Dice Loss (Group Mask Loss):** Compares predicted group mask with GT paragraph region.
    $$Dice = 1 - \frac{2 \times intersection}{sum(pred) + sum(gt)}$$
2.  **Affinity Loss (Binary Cross Entropy):** Trains the affinity matrix.
    $$L = - [ A \times log(\hat{A}) + (1-A) \times log(1-\hat{A}) ]$$
3.  **Detection Loss:** The original MaskDINO detection loss ($L_{class} + L_{box} + L_{mask}$).

**Final Loss Function:**
$$L_{TGA} = \text{dice loss} + \text{affinity loss} + \text{detection loss}$$

#### Running the Model on IITD HPC Server

**Environment Setup**

1.  Created a clean Conda environment.
2.  Installed PyTorch and Detectron2 compatible with CUDA on the HPC.
3.  Installed TGA dependencies.
4.  Ensured version alignment (a mismatch between PyTorch, Detectron2, and TGA configs caused errors until fixed).

**Preprocessing**
All images were padded so that both height and width are divisible by 32. This is required because MaskDINO’s FPN downsamples by powers of 2.

  * Ex: 703 x 900 -\> 704 x 928
  * Ex: 768 x 1023 -\> 768 x 1024

**Main Scripts**

  * `train_net.py`: Runs training.
    ```bash
    python train_net.py \
      --num-gpus 1 \
      --config-file configs/TGA_MaskDINO_R50.yaml \
      --resume
    ```
  * `build_tga_model()`: Assembles MaskDINO + TIFA + GMP + Affinity head.
  * `tga_loss.py`: Implements Dice loss, BCE affinity loss, and combined TGA loss.
  * `evaluator_tga.py`: Evaluates PQ, F1, Dice, and IoU of our model on the HierText dataset.

#### Backbone Issue & Debugging

Training failed due to a wrong MaskDINO checkpoint and hundreds of missing/unexpected weight keys. The result was that all instance masks were zero, and TGA could not learn (PQ = 0).

**Fix Attempts:**

1.  **Testing four different versions of MaskDINO-R50:** Including the 50-epoch pretraining model, 100-epoch version, and TGA-recommended checkpoint.
2.  **Rechecking input–output size constraints:** Ensuring image padding to multiples of 32 and FPN pyramid alignment.
3.  **Verifying mask resolution consistency:** MaskDINO produces masks at 1/4 resolution; TGA resizes to 1/8 for TIFA. All interpolation/alignment rechecked.
4.  **Checking dataset annotations:** Ensured dataset masks, bounding boxes, and categories were correctly formatted (COCO-style).

### Conclusion

TGA represents a powerful and lightweight approach. However, despite extensive attempts to reproduce the method, TGA did not train successfully in our setup. The backbone initialization failure repeatedly resulted in empty instance masks.

### 3.3 Hi-SAM

#### Introduction

Hi-SAM does unified text segmentation and Layout Analysis. Hi-SAM excels in segmentation across four hierarchies:

1.  **Pixel-Level Text:** Identifies every pixel belonging to text strokes (foreground) vs. background.
2.  **Word-Level Text:** Identifies text instances (words).
3.  **Text Line:** Identifies groups of words (lines).
4.  **Paragraph/Layout Analysis:** Recognizes higher-level text structures (paragraphs).

#### Architecture Overview

<img width="1260" height="496" alt="image" src="https://github.com/user-attachments/assets/9e8b8fc0-d9a9-44df-b93a-cf7e297a8cda" />


  * **Feature Extraction (SAM’s Image Encoder + Adapter Tuning):** Uses the frozen image encoder of Segment Anything Model (SAM). Adapters are inserted into the ViT blocks (Down-projection → ReLU → Up-projection). Only these adapters are trained to learn fine text details.
  * **Pixel Level Masks (Self Prompting Module + S-Decoder):** Generates accurate pixel-level masks. Image embeddings are converted into implicit prompt tokens and fed into S-Decoder to generate Low-Resolution and High-Resolution masks.
  * **Hierarchical Text Segmentation (Prompt Encoder):** Generates masks at word, text-line, and paragraph levels from a single prompt. Outputs three distinct tokens for every point prompt: Token 1 (Word), Token 2 (Text-line), Token 3 (Paragraph).
  * **Layout Analysis (H-Decoder):** Performs layout reasoning. Takes initial paragraph masks, computes pairwise IoU, and merges them using Union-Find if IoU \> 0.5.

#### Results

Hi-SAM demonstrates strong performance. Reported Panoptic Quality (PQ):

  * Word PQ: 64.30
  * Text-Line PQ: 66.96
  * Paragraph PQ: 59.09

#### Bottlenecks & Limitations

  * **Heavy Encoder (\~632 M):** Relies on a large SAM encoder, making it difficult to achieve real-time inference.
  * **Resource Intensive:** Training requires significant resources (8 × NVIDIA Tesla V100 GPUs, 150 epochs).
  * **Parameters:** 62.2 M trainable & 699.2 M total.

### Conclusion
 Hi-SAM is SOTA but impractical for resource-constrained use cases, highlighting the need for a Lightweight Model.

-----

## 4\. MobileSAM + Hierarchical Decoder

### Introduction

The model is designed by combining the MobileSAM encoder with a custom hierarchical decoder. We hypothesized that the MobileSAM encoder can provide three spatial feature maps: Shallow features (high res), Mid-level features, and Deep features (low res/high semantic).

### Model Architecture

<img width="758" height="1098" alt="image" src="https://github.com/user-attachments/assets/a17765be-602a-43ee-b55b-b41f229cb825" />


  * **Encoder (MobileSAM):** Uses pretrained Tiny ViT Encoder. Takes 1024×1024 images and outputs Shallow (C=256), Mid-level (C=256), and Deep (C=320) feature maps.
  * **Decoder (Hierarchical Decoder):** Uses a three-stage upsampling path (Deep → Mid → Shallow fusion). Each stage contains Convolution + normalization + activation and Bilinear upsampling. The final segmentation head outputs 3 binary masks (word, line, paragraph).

### Loss Functions

  * **Binary Cross-Entropy (BCE) Loss:** For pixel-wise classification.
  * **Dice Loss:** For boundary accuracy.
  * **Final Loss:** BCE Loss + Dice Loss.

### Results

  * Word PQ: 0.4969
  * Text-Line PQ: 0.5254
  * Paragraph PQ: 0.4300

### Conclusion
The PQ scores are approximately 75% of Hi-SAM’s scores, showing that our assumption enabled a workable multi-scale fusion approach. The model is small (\~6–7M parameters).

-----

## 5\. MOBILE Hi-SAM (Our Novelty\!)

### 5.1 Logic Behind the Architecture

The objective of Mobile-Hi-SAM is to combine the efficiency of MobileSAM’s TinyViT encoder with the hierarchical text segmentation capabilities of Hi-SAM. We address the heavy compute of Hi-SAM by introducing:

  * A lightweight encoder.
  * An adapter module to refine MobileSAM embeddings.
  * A modal aligner for implicit prompt tokens.
  * A hierarchical decoder with independent heads.
  * A multi-component loss (Dice, Focal, IoU).

### 5.2 Components of the Solution

<img width="1356" height="772" alt="image" src="https://github.com/user-attachments/assets/613f0a17-f280-4053-8b0f-227e6ae9fb5b" />

**5.2.1 Image Encoder (Frozen MobileSAM TinyViT)**
A pretrained TinyViT-based MobileSAM encoder is used without modification to reduce training cost.

  * Input: `[B, 3, 1024, 1024]`
  * Output: `[B, 256, 64, 64]`

**5.2.2 Adapter Module**
A lightweight refinement block that transforms MobileSAM features into embeddings compatible with the hierarchical decoding pipeline.

  * **Architecture:** Conv2d(256→256, 1×1) + LayerNorm2d + GELU
  * **Output:** `[B, 256, 64, 64]`

**5.2.3 Modal Aligner (Implicit Prompt Generator)**
Produces sparse prompt tokens representing hierarchical cues.

  * **a) Spatial Attention Generator:** Stack of 3×3 convolutions generating 12 attention maps.
  * **b) Weighted Feature Pooling:** Produces 12 sparse tokens via attention-weighted pooling.
  * **c) Transformer Refinement:** One transformer layer (8 heads) performs self-attention and cross-attention.

**5.2.4 Prompt Encoder**
Follows the SAM/Hi-SAM design. Generates dense prompt embeddings with positional encodings.

**5.2.5 S-Decoder (SAM-Style Mask Decoder)**
Performs two-way attention between sparse prompts, dense embeddings, and image embeddings.

**5.2.6 Hierarchical Decoder (Three Independent Heads)**
Three separate heads for paragraph, line, and word segmentation. Independent heads were chosen to allow better level-specific specialization given the reduced encoder capacity. Each head includes:

  * IoU token `[1, 256]`
  * Three mask tokens `[3, 256]`
  * Two-way transformer
  * Level-specific feature upscaling (ConvTranspose2d + Conv2d)
  * Level-specific hypernetwork MLP
  * IoU prediction MLP

**Outputs (per head):** Masks `[B, 3, 256, 256]`, IoU scores `[B, 3]`.

### 5.3 Novelty and Modifications Introduced

1.  **Replacement of ViT-H Encoder with TinyViT:** Replaced ≈650M-parameter ViT-H with ≈5M-parameter TinyViT. Reduces model size by \~98%.
2.  **Adapter Module:** Uses an external adapter to bridge MobileSAM embeddings to Hi-SAM-compatible processing (unlike Hi-SAM's internal adapters).
3.  **Fully Independent Hierarchical Heads:** Separates all three heads (Word, Line, Paragraph) to improve specialization, unlike Hi-SAM which shares components.
4.  **Dimension Redesign:** All modules adapted for 256-dimensional embeddings (down from 768–1024 in Hi-SAM).

### 5.4 Hierarchical Loss Function

Each hierarchy level uses a combined loss function:

| Component | Purpose | Weight |
| :--- | :--- | :--- |
| **Dice Loss** | Region overlap, class imbalance robustness | 1.0 |
| **Focal Loss** ($\alpha=0.25, \gamma=2.0$) | Focus on hard examples and fine structures | 20.0 |
| **IoU Prediction Loss (MSE)** | Confidence calibration for mask quality | 1.0 |

**Total Loss:**
$$L_{total} = L_{paragraph} + L_{line} + L_{word}$$

### 5.5 Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Epochs** | 50 |
| **Learning rate** | 1e-4 |
| **Optimizer** | AdamW |
| **Batch size** | 4 |
| **Resolution** | 1024×1024 |
| **Precision** | Mixed (AMP) |
| **Encoder** | Frozen |
| **Hardware** | Single GPU |

Training incorporates sparse sampling of text regions to reduce memory and improve stability.

### 5.6 Possible Modifications to Improve PQ and fgIoU

Potential improvements include:

  * Fine-tuning a portion of the MobileSAM encoder.
  * Increasing the weight of the focal loss in the hierarchical loss.
  * Adding cross-level consistency constraints (e.g., enforcing word ⊆ line ⊆ paragraph).
  * Using larger upscaling modules or hybrid CNN-transformer blocks.
  * Incorporating boundary-aware or Lovász losses.

### 5.7 Performance Comparison with Hi-SAM

| Model | Avg PQ | Avg fgIoU | Params | Deployability |
| :--- | :--- | :--- | :--- | :--- |
| **Hi-SAM-H** | \~65% | \~75% | \~650M | TPU/GPU only |
| **Mobile-Hi-SAM** | **40.31%** | **58.35%** | **12.6M** | **Mobile / edge suitable** |

**Observations:**

  * Mobile-Hi-SAM retains approximately 62% of Hi-SAM’s PQ and 78% of fgIoU.
  * Achieves this with a model size ≈2% of Hi-SAM’s.
  * Suitable for real-time or mobile applications where Hi-SAM is infeasible.

**Independent Hierarchical Heads Performance:**

  * Word PQ: 40.41
  * Line PQ: 40.81
  * Layout PQ: 39.69

### Conclusion

Mobile-Hi-SAM demonstrates that hierarchical text segmentation can be achieved efficiently without heavy compute requirements, making it suitable for practical deployments outside of high-performance compute clusters.


### Video: https://drive.google.com/file/d/1KRr0IiJe4BI6k3QTmSATROZW1ll48nXP/view?usp=share_link




