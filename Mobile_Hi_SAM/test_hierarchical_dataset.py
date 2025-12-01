"""
Test script for hierarchical dataset loader
Run from: ~/scratch/DL/DL_Project/MOBILE-HI-SAM/Mobile_Hi_SAM
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/MOBILE-HI-SAM"
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.train.hiertext_hierarchical_dataset import HierTextHierarchicalDataset

print("=" * 60)
print("Testing Hierarchical Dataset Loader")
print("=" * 60)

# Path to your HierText dataset
HIERTEXT_ROOT = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/hiertext"
print(f"\nLoading dataset from: {HIERTEXT_ROOT}")
print("Loading 10 samples for testing...\n")

try:
    # Create dataset
    dataset = HierTextHierarchicalDataset(
        root=HIERTEXT_ROOT,
        split="train",
        max_items=10,
    )
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}\n")
    
    # Test first sample
    print("Testing first sample...")
    sample = dataset[0]
    
    print("\n" + "=" * 60)
    print("Sample 0 Details:")
    print("=" * 60)
    
    print("\nAvailable keys:", list(sample.keys()))
    
    print("\nShapes:")
    print(f"  Image: {sample['image'].shape}")
    print(f"  Paragraph mask: {sample['gt_para_mask'].shape}")
    print(f"  Line mask: {sample['gt_line_mask'].shape}")
    print(f"  Word mask: {sample['gt_word_mask'].shape}")
    
    print("\nHierarchy counts:")
    print(f"  Paragraphs: {sample['num_paragraphs']}")
    print(f"  Lines: {sample['num_lines']}")
    print(f"  Words: {sample['num_words']}")
    
    print("\nPrompt:")
    print(f"  Point coords: {sample['point_coords'].shape}")
    print(f"  Point labels: {sample['point_labels'].shape}")
    
    print("\nMetadata:")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  Original size: {sample['original_size']}")
    
    # Check a few more samples
    print("\n" + "=" * 60)
    print("Testing additional samples:")
    print("=" * 60)
    
    for i in range(1, min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image: {sample['image_id']}")
        print(f"  Hierarchy: {sample['num_paragraphs']} paras, "
              f"{sample['num_lines']} lines, {sample['num_words']} words")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nDataset is ready for training! 🚀")
    
except Exception as e:
    print(f"\n❌ Error occurred:")
    print(f"   {type(e).__name__}: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
