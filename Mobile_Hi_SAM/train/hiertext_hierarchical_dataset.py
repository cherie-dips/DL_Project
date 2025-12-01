"""
HierText Dataset Loader with 3-Level Hierarchy
Extracts paragraph, line, and word annotations simultaneously.
"""

import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HierTextHierarchicalDataset(Dataset):
    """
    HierText dataset with hierarchical annotations.
    Returns paragraph, line, and word masks for each sample.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        max_items: int = None,
        img_size: int = 1024,
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        
        self.jsonl_path = os.path.join(root, "gt", f"{split}.jsonl")
        self.img_folder = os.path.join(root, split)
        
        print(f"[HierText] Loading annotations from: {self.jsonl_path}")
        print(f"[HierText] Images folder: {self.img_folder}")
        
        # Load JSON
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract annotations
        if isinstance(data, dict):
            if "annotations" in data:
                self.records = data["annotations"]
            elif "images" in data:
                self.records = data["images"]
            else:
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        self.records = value
                        break
        elif isinstance(data, list):
            self.records = data
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")
        
        print(f"[HierText] Found {len(self.records)} total annotations")
        
        # Filter records that have hierarchical annotations
        self.records = [r for r in self.records if self._has_hierarchy(r)]
        print(f"[HierText] Filtered to {len(self.records)} with hierarchy")
        
        # Take subset if specified
        if max_items and len(self.records) > max_items:
            self.records = random.sample(self.records, max_items)
        
        print(f"[HierText] Using {len(self.records)} samples")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def _has_hierarchy(self, record):
        """Check if record has paragraph/line/word hierarchy"""
        if "paragraphs" not in record:
            return False
        
        for para in record["paragraphs"]:
            if "lines" in para and len(para["lines"]) > 0:
                for line in para["lines"]:
                    if "words" in line and len(line["words"]) > 0:
                        return True
        return False
    
    def polygon_to_mask(self, size, vertices):
        """Convert polygon vertices to binary mask"""
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        
        try:
            if isinstance(vertices[0], list):
                pts = [(v[0], v[1]) for v in vertices]
            else:
                pts = [(vertices[i], vertices[i+1]) for i in range(0, len(vertices), 2)]
            
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
        except:
            pass
        
        return torch.tensor(np.array(mask), dtype=torch.float32)
    
    def extract_hierarchy(self, record, img_size):
        """
        Extract paragraph, line, and word annotations.
        
        Returns:
            para_vertices: List of paragraph polygons
            line_vertices: List of line polygons
            word_vertices: List of word polygons
        """
        para_vertices = []
        line_vertices = []
        word_vertices = []
        
        if "paragraphs" not in record:
            return para_vertices, line_vertices, word_vertices
        
        for para in record["paragraphs"]:
            # Paragraph vertices
            if "vertices" in para and len(para["vertices"]) >= 3:
                para_vertices.append(para["vertices"])
            
            # Line vertices
            if "lines" in para:
                for line in para["lines"]:
                    if "vertices" in line and len(line["vertices"]) >= 3:
                        line_vertices.append(line["vertices"])
                    
                    # Word vertices
                    if "words" in line:
                        for word in line["words"]:
                            if "vertices" in word and len(word["vertices"]) >= 3:
                                word_vertices.append(word["vertices"])
        
        return para_vertices, line_vertices, word_vertices
    
    def create_hierarchical_masks(self, para_verts, line_verts, word_verts, img_size):
        """
        Create masks for all hierarchy levels.
        Combines multiple instances into single mask per level.
        
        Returns:
            para_mask: (H, W) - Union of all paragraph masks
            line_mask: (H, W) - Union of all line masks
            word_mask: (H, W) - Union of all word masks
        """
        W, H = img_size
        
        # Create combined masks
        para_mask = torch.zeros((H, W), dtype=torch.float32)
        line_mask = torch.zeros((H, W), dtype=torch.float32)
        word_mask = torch.zeros((H, W), dtype=torch.float32)
        
        # Add all paragraph masks
        for verts in para_verts:
            mask = self.polygon_to_mask((W, H), verts)
            para_mask = torch.maximum(para_mask, mask)
        
        # Add all line masks
        for verts in line_verts:
            mask = self.polygon_to_mask((W, H), verts)
            line_mask = torch.maximum(line_mask, mask)
        
        # Add all word masks
        for verts in word_verts:
            mask = self.polygon_to_mask((W, H), verts)
            word_mask = torch.maximum(word_mask, mask)
        
        return para_mask, line_mask, word_mask
    
    def create_prompt_from_hierarchy(self, para_verts, img_size):
        """
        Create a prompt point from paragraph center.
        Could be extended to use more sophisticated hierarchical prompts.
        """
        W, H = img_size
        
        if len(para_verts) == 0:
            # Fallback to center
            return 0.5, 0.5
        
        # Use first paragraph center as prompt
        try:
            verts = para_verts[0]
            if isinstance(verts[0], list):
                verts_np = np.array(verts)
            else:
                verts_np = np.array(verts).reshape(-1, 2)
            
            cx = np.clip(verts_np[:, 0].mean() / W, 0, 1)
            cy = np.clip(verts_np[:, 1].mean() / H, 0, 1)
        except:
            cx, cy = 0.5, 0.5
        
        return cx, cy
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        # Get image ID
        img_id = None
        if "image_id" in rec:
            img_id = rec["image_id"]
        elif "info" in rec and "image_id" in rec["info"]:
            img_id = rec["info"]["image_id"]
        elif "image_path" in rec:
            img_id = os.path.splitext(os.path.basename(rec["image_path"]))[0]
        
        if img_id is None:
            img_id = f"img_{idx}"
        
        # Find image file
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            test_path = os.path.join(self.img_folder, f"{img_id}{ext}")
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None or not os.path.exists(img_path):
            # Dummy image
            img = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
            W, H = self.img_size, self.img_size
        else:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
        
        # Extract hierarchical annotations
        para_verts, line_verts, word_verts = self.extract_hierarchy(rec, (W, H))
        
        # Create masks for all levels
        para_mask, line_mask, word_mask = self.create_hierarchical_masks(
            para_verts, line_verts, word_verts, (W, H)
        )
        
        # Resize masks
        resize_transform = transforms.Resize((self.img_size, self.img_size))
        para_mask = resize_transform(para_mask.unsqueeze(0)).squeeze(0)
        line_mask = resize_transform(line_mask.unsqueeze(0)).squeeze(0)
        word_mask = resize_transform(word_mask.unsqueeze(0)).squeeze(0)
        
        # Create prompt
        cx, cy = self.create_prompt_from_hierarchy(para_verts, (W, H))
        point_coords = torch.tensor([[[cx * self.img_size, cy * self.img_size]]], dtype=torch.float32)
        point_labels = torch.tensor([[1]], dtype=torch.long)
        
        # Transform image
        img_t = self.transform(img)
        
        return {
            "image": img_t,
            "original_size": (self.img_size, self.img_size),
            "point_coords": point_coords,
            "point_labels": point_labels,
            # Hierarchical ground truth masks
            "gt_para_mask": para_mask.unsqueeze(0),  # (1, H, W)
            "gt_line_mask": line_mask.unsqueeze(0),  # (1, H, W)
            "gt_word_mask": word_mask.unsqueeze(0),  # (1, H, W)
            # Metadata
            "image_id": img_id,
            "num_paragraphs": len(para_verts),
            "num_lines": len(line_verts),
            "num_words": len(word_verts),
        }
    
    def __len__(self):
        return len(self.records)


def collate_fn(batch):
    """Custom collate function (returns list of dicts)"""
    return batch
