"""
HierText dataset with per-instance hierarchical supervision.

Each sample is one (word, line, paragraph) triple drawn from the annotation tree,
prompted at the word's centroid. This is what makes the three levels genuinely
different targets and makes the prompt determine the answer.

Geometry follows SAM's protocol: the long side is resized to img_size and the
image is padded bottom-right. Masks are rasterised directly at the decoder's
output resolution from scaled vertices, so no mask is ever interpolated.
"""

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def index_jsonl(path: str, usable) -> Optional[List[int]]:
    """Byte offsets of usable records in a line-delimited JSON file.

    Returns None if the file is not line-delimited, so the caller can fall back
    to a full parse. Indexing keeps only one record in memory at a time, which
    matters because the reconstructed train split parses to 2.55 GB - and every
    DataLoader worker would otherwise carry its own copy of it.
    """
    offsets = []
    with open(path, "rb") as f:
        first = f.readline()
        if not first:
            return []
        try:
            record = json.loads(first)
        except json.JSONDecodeError:
            return None                      # single large JSON object
        if not isinstance(record, dict) or "paragraphs" not in record:
            return None
        if usable(record):
            offsets.append(0)
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return None
            if usable(record):
                offsets.append(offset)
    return offsets


def load_annotations(path: str) -> List[dict]:
    """HierText ships a single JSON object despite the .jsonl extension.

    Falls back to line-delimited parsing so both layouts work.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("annotations", "images"):
            if key in data and isinstance(data[key], list):
                return data[key]
        for value in data.values():
            if isinstance(value, list) and value:
                return value
    raise ValueError(f"Unrecognised annotation layout in {path}: {type(data)}")


def _as_xy(vertices: Sequence) -> np.ndarray:
    """Accept [[x,y],...] or [x,y,x,y,...] and return an (N, 2) float array."""
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr


class HierTextHierarchicalDataset(Dataset):
    """Yields one nested (word, line, paragraph) instance per sample."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_items: Optional[int] = None,
        img_size: int = 1024,
        mask_size: int = 256,
        word_mask_size: int = 384,
        samples_per_image: int = 1,
        deterministic: bool = False,
        seed: int = 0,
        include_text_mask: bool = False,
        text_mask_size: int = 1024,
        stroke_gt_dir: Optional[str] = None,
        records: Optional[List[dict]] = None,
    ):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.mask_size = mask_size
        self.word_mask_size = word_mask_size
        self.samples_per_image = max(1, samples_per_image)
        # Validation must be reproducible: the same index has to yield the same
        # instance on every pass, or the val curve measures sampling noise.
        self.deterministic = deterministic
        self.seed = seed
        # Stroke-level text foreground for the S-Decoder's pixel branch.
        #
        # HierText itself has no stroke annotations. Hi-SAM's authors contributed
        # them as a separate download (binary PNGs, 0/255, in train_gt/
        # validation_gt/test_gt) - see Hi-SAM's datasets/data_preparation.md. A
        # union of filled word polygons is NOT a substitute: it is a box-ish blob
        # where the target is letter strokes, so training on it would not
        # reproduce Hi-SAM's fgIOU and the number would not be comparable.
        self.include_text_mask = include_text_mask
        self.text_mask_size = text_mask_size
        self.stroke_gt_dir = stroke_gt_dir
        if include_text_mask and not stroke_gt_dir:
            raise ValueError(
                "include_text_mask=True requires stroke_gt_dir pointing at "
                "Hi-SAM's contributed stroke-level PNG masks (e.g. "
                f"<root>/{split}_gt). Download them per Hi-SAM's "
                "datasets/data_preparation.md. Filled polygons are not a valid "
                "stand-in for stroke ground truth."
            )
        if stroke_gt_dir and not os.path.isdir(stroke_gt_dir):
            raise FileNotFoundError(f"stroke_gt_dir does not exist: {stroke_gt_dir}")

        self.jsonl_path = os.path.join(root, "gt", f"{split}.jsonl")
        self.img_folder = os.path.join(root, split)

        self.offsets = None
        if records is None:
            print(f"[HierText] Loading annotations from: {self.jsonl_path}")
            self.offsets = index_jsonl(self.jsonl_path, self._usable)
            if self.offsets is not None:
                self.records = None
                print(f"[HierText] Indexed {len(self.offsets)} usable records "
                      f"(lazy; records parsed on demand)")
            else:
                records = load_annotations(self.jsonl_path)
                print(f"[HierText] Found {len(records)} total annotations "
                      "(single JSON object - parsed into memory)")

        if self.offsets is None:
            self.records = [r for r in records if self._usable(r)]
            print(f"[HierText] Filtered to {len(self.records)} with a complete tree")

        if max_items and self._n_records() > max_items:
            rng = random.Random(seed)
            if self.offsets is not None:
                self.offsets = rng.sample(self.offsets, max_items)
            else:
                self.records = rng.sample(self.records, max_items)

        if not self._n_records():
            raise ValueError(
                f"No usable records for split '{split}'. Every record needs at "
                "least one paragraph -> line -> word chain."
            )

        print(
            f"[HierText] Using {self._n_records()} images "
            f"x {self.samples_per_image} instance(s) = "
            f"{self._n_records() * self.samples_per_image} samples"
        )

    def _n_records(self) -> int:
        return len(self.offsets) if self.offsets is not None else len(self.records or [])

    def get_record(self, idx: int) -> dict:
        """One annotation record, parsed on demand when the file is line-delimited."""
        if self.offsets is None:
            return self.records[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(self.offsets[idx])
            return json.loads(f.readline())

    # ------------------------------------------------------------------
    # Annotation tree
    # ------------------------------------------------------------------
    @staticmethod
    def extract_nested(record: dict) -> List[dict]:
        """Keep the hierarchy instead of flattening it into per-level unions."""
        tree = []
        for para in record.get("paragraphs", []) or []:
            if len(para.get("vertices", []) or []) < 3:
                continue
            lines = []
            for line in para.get("lines", []) or []:
                if len(line.get("vertices", []) or []) < 3:
                    continue
                words = [
                    w["vertices"]
                    for w in (line.get("words", []) or [])
                    if len(w.get("vertices", []) or []) >= 3
                ]
                if words:
                    lines.append({"verts": line["vertices"], "words": words})
            if lines:
                tree.append({"verts": para["vertices"], "lines": lines})
        return tree

    def _usable(self, record: dict) -> bool:
        return len(self.extract_nested(record)) > 0

    # ------------------------------------------------------------------
    # Rasterisation
    # ------------------------------------------------------------------
    @staticmethod
    def polygon_to_mask(vertices, scale: float, size: int) -> torch.Tensor:
        """Rasterise a polygon directly at `size`, scaling the vertices.

        Drawing at the target resolution avoids resizing a binary mask, which
        would produce fractional targets that break focal loss.
        """
        mask = Image.new("L", (size, size), 0)
        pts = _as_xy(vertices) * scale
        if len(pts) >= 3:
            ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in pts], fill=1)
        return torch.from_numpy(np.array(mask, dtype=np.float32))

    @staticmethod
    def centroid(vertices, scale: float) -> Tuple[float, float]:
        pts = _as_xy(vertices) * scale
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _find_image(self, record: dict, idx: int) -> Tuple[Optional[str], str]:
        img_id = (
            record.get("image_id")
            or (record.get("info") or {}).get("image_id")
            or (
                os.path.splitext(os.path.basename(record["image_path"]))[0]
                if "image_path" in record
                else None
            )
            or f"img_{idx}"
        )
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
            candidate = os.path.join(self.img_folder, f"{img_id}{ext}")
            if os.path.exists(candidate):
                return candidate, img_id
        return None, img_id

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n_records() * self.samples_per_image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record_idx = idx // self.samples_per_image
        record = self.get_record(record_idx)

        rng = random.Random(self.seed * 1_000_003 + idx) if self.deterministic else random

        img_path, img_id = self._find_image(record, record_idx)
        if img_path is None:
            img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        else:
            img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # --- SAM geometry: long side to img_size, pad bottom-right ---
        scale = self.img_size / max(W, H)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        img = img.resize((nw, nh), Image.BILINEAR)

        canvas = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        canvas.paste(img, (0, 0))
        image = torch.from_numpy(
            np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )

        # --- sample one nested instance ---
        tree = self.extract_nested(record)
        para = rng.choice(tree)
        line = rng.choice(para["lines"])
        word = rng.choice(line["words"])

        # Masks rasterise at their own decoder resolution; the vertex scale
        # composes the image resize with the decoder's downsampling factor.
        mask_scale = scale * (self.mask_size / self.img_size)
        word_scale = scale * (self.word_mask_size / self.img_size)

        cx, cy = self.centroid(word, scale)
        cx = min(max(cx, 0.0), self.img_size - 1)
        cy = min(max(cy, 0.0), self.img_size - 1)

        return {
            "image": image,
            "point_coords": torch.tensor([[cx, cy]], dtype=torch.float32),
            "point_labels": torch.tensor([1], dtype=torch.int64),
            # Ground truth, each at the resolution it is supervised at
            "gt_word_mask": self.polygon_to_mask(word, word_scale, self.word_mask_size).unsqueeze(0),
            "gt_word_mask_lr": self.polygon_to_mask(word, mask_scale, self.mask_size).unsqueeze(0),
            "gt_line_mask": self.polygon_to_mask(line["verts"], mask_scale, self.mask_size).unsqueeze(0),
            "gt_para_mask": self.polygon_to_mask(para["verts"], mask_scale, self.mask_size).unsqueeze(0),
            # Geometry needed to undo the resize-and-pad at inference
            "input_size": (nh, nw),
            "original_size": (H, W),
            "image_id": img_id,
            **self._text_mask(img_id, scale, nh, nw),
        }

    @classmethod
    def from_records(cls, records: List[dict], img_folder: str = "", **kwargs):
        """Build from in-memory annotations, bypassing disk. Used by the tests."""
        ds = cls(root="", split="", records=records, **kwargs)
        ds.img_folder = img_folder
        return ds

    def _text_mask(self, img_id: str, scale: float, nh: int, nw: int) -> Dict[str, Any]:
        """Load the stroke-level mask and put it through the image's geometry."""
        if not self.include_text_mask:
            return {}

        path = None
        for ext in (".png", ".PNG", ".jpg"):
            candidate = os.path.join(self.stroke_gt_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                f"No stroke-level mask for image '{img_id}' in {self.stroke_gt_dir}"
            )

        stroke = Image.open(path).convert("L").resize((nw, nh), Image.NEAREST)
        canvas = Image.new("L", (self.img_size, self.img_size), 0)
        canvas.paste(stroke, (0, 0))

        full = torch.from_numpy(
            (np.array(canvas, dtype=np.uint8) > 127).astype(np.float32)
        ).unsqueeze(0)

        hr = torch.nn.functional.interpolate(
            full.unsqueeze(0), (self.text_mask_size, self.text_mask_size), mode="nearest"
        )[0]
        lr = torch.nn.functional.interpolate(
            full.unsqueeze(0), (self.mask_size, self.mask_size), mode="nearest"
        )[0]
        return {"gt_text_mask": hr, "gt_text_mask_lr": lr}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack into real batch tensors so the decoder runs once, not once per image."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "point_coords": torch.stack([b["point_coords"] for b in batch]),
        "point_labels": torch.stack([b["point_labels"] for b in batch]),
        "gt_word_mask": torch.stack([b["gt_word_mask"] for b in batch]),
        "gt_word_mask_lr": torch.stack([b["gt_word_mask_lr"] for b in batch]),
        "gt_line_mask": torch.stack([b["gt_line_mask"] for b in batch]),
        "gt_para_mask": torch.stack([b["gt_para_mask"] for b in batch]),
        "input_size": [b["input_size"] for b in batch],
        "original_size": [b["original_size"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        **({k: torch.stack([b[k] for b in batch])
            for k in ("gt_text_mask", "gt_text_mask_lr")}
           if "gt_text_mask" in batch[0] else {}),
    }


class HierTextEvalDataset(HierTextHierarchicalDataset):
    """Whole-image evaluation: every instance at every level, not one sampled triple.

    Instance-matched PQ needs the full set of ground-truth polygons per image.
    Polygons are returned as scaled vertex arrays rather than rasterised masks so
    the loader stays light - an image with ~126 words would otherwise ship ~8 MB
    of boolean masks per sample.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.get_record(idx)

        img_path, img_id = self._find_image(record, idx)
        if img_path is None:
            img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        else:
            img = Image.open(img_path).convert("RGB")
        W, H = img.size

        scale = self.img_size / max(W, H)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        canvas.paste(img, (0, 0))
        image = torch.from_numpy(
            np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )

        mask_scale = scale * (self.mask_size / self.img_size)
        word_scale = scale * (self.word_mask_size / self.img_size)

        tree = self.extract_nested(record)
        word_polys, line_polys, para_polys = [], [], []
        for para in tree:
            para_polys.append(_as_xy(para["verts"]) * mask_scale)
            for line in para["lines"]:
                line_polys.append(_as_xy(line["verts"]) * mask_scale)
                for word in line["words"]:
                    word_polys.append(_as_xy(word) * word_scale)

        return {
            "image": image,
            "word_polys": word_polys,   # at word_mask_size
            "line_polys": line_polys,   # at mask_size
            "para_polys": para_polys,   # at mask_size
            "input_size": (nh, nw),
            "original_size": (H, W),
            "scale": scale,
            "image_id": img_id,
        }


def eval_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Images stack; per-image instance lists stay as lists of varying length."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "word_polys": [b["word_polys"] for b in batch],
        "line_polys": [b["line_polys"] for b in batch],
        "para_polys": [b["para_polys"] for b in batch],
        "input_size": [b["input_size"] for b in batch],
        "original_size": [b["original_size"] for b in batch],
        "scale": [b["scale"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


def rasterize_polys(polys, size: int) -> List[np.ndarray]:
    """Rasterise already-scaled polygons into boolean instance masks."""
    out = []
    for pts in polys:
        mask = Image.new("L", (size, size), 0)
        if len(pts) >= 3:
            ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in pts], fill=1)
        arr = np.array(mask, dtype=bool)
        if arr.any():
            out.append(arr)
    return out
