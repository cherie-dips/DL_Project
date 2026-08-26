"""
End-to-end smoke test for the Mobile-Hi-SAM training pipeline.

Runs without the MobileSAM checkpoint by substituting a stand-in encoder with
the same output contract (B, 256, 64, 64). Everything downstream - prompt
encoder, HiDecoder, loss, backward, the dead-gradient assertion - is the real
code path.

    python -m Mobile_Hi_SAM.tests.test_pipeline

Each check corresponds to a gate in REMEDIATION_PLAN.md.
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ----------------------------------------------------------------------
# Stand in for MobileSAM's TinyViT so the test runs without the checkpoint.
# Installed into sys.modules before mobile_hisam_model imports it.
# ----------------------------------------------------------------------
class _StubEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, img_size=1024, out_chans=256):
        super().__init__()
        self.img_size = img_size
        self.stem = nn.Conv2d(3, 32, 3, stride=4, padding=1)
        self.bn = nn.BatchNorm2d(32)          # TinyViT has BatchNorm; A2 must freeze it
        self.proj = nn.Conv2d(32, out_chans, 3, stride=4, padding=1)

    def forward(self, x):
        return self.proj(torch.relu(self.bn(self.stem(x))))


_stub_module = types.ModuleType("Mobile_Hi_SAM.models.mobile_encoder")
_stub_module.MobileSAMEncoder = _StubEncoder
sys.modules["Mobile_Hi_SAM.models.mobile_encoder"] = _stub_module

from Mobile_Hi_SAM.models.mobile_hisam_model import (  # noqa: E402
    MobileHiSAM,
    assert_no_dead_parameters,
)
from Mobile_Hi_SAM.models.hierarchical_loss import HierarchicalLoss  # noqa: E402
from Mobile_Hi_SAM.train.hiertext_hierarchical_dataset import (  # noqa: E402
    HierTextHierarchicalDataset,
    collate_fn,
)

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"{PASS if condition else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(condition)


def build_model():
    return MobileHiSAM(
        checkpoint_path=None, img_size=1024, embed_dim=256,
        enable_hierarchical=True, enable_s_decoder=False,
    )


def fake_batch(batch_size=2, include_text_mask=False):
    """One synthetic nested record, put through the real dataset code path."""
    W, H = 1600, 1067
    rec = {
        "image_id": "synthetic",
        "paragraphs": [{
            "vertices": [[0.05 * W, 0.05 * H], [0.95 * W, 0.05 * H],
                         [0.95 * W, 0.55 * H], [0.05 * W, 0.55 * H]],
            "lines": [{
                "vertices": [[0.1 * W, 0.1 * H], [0.9 * W, 0.1 * H],
                             [0.9 * W, 0.2 * H], [0.1 * W, 0.2 * H]],
                "words": [
                    {"vertices": [[0.10 * W, 0.10 * H], [0.30 * W, 0.10 * H],
                                  [0.30 * W, 0.20 * H], [0.10 * W, 0.20 * H]]},
                    {"vertices": [[0.35 * W, 0.10 * H], [0.55 * W, 0.10 * H],
                                  [0.55 * W, 0.20 * H], [0.35 * W, 0.20 * H]]},
                ],
            }],
        }],
    }
    ds = HierTextHierarchicalDataset.from_records(
        [rec], img_folder="/nonexistent", deterministic=True,
        include_text_mask=include_text_mask, text_mask_size=1024,
    )
    return collate_fn([ds[0] for _ in range(batch_size)])


def main():
    torch.manual_seed(0)
    print("\n=== A1: input scaling ===")
    model = build_model()
    x = torch.rand(1, 3, 1024, 1024)          # ToTensor range
    pre = model.preprocess(x)
    lo, hi = float(pre.min()), float(pre.max())
    check("preprocess spans SAM's intended range",
          lo < -2.0 and hi > 2.0, f"[{lo:.3f}, {hi:.3f}] want ~[-2.12, +2.25]")

    print("\n=== A2: frozen encoder stays frozen ===")
    model.train()
    check("image_encoder in eval mode after model.train()", not model.image_encoder.training)
    check("prompt_encoder in eval mode after model.train()", not model.prompt_encoder.training)
    before = model.image_encoder.bn.running_mean.clone()
    for _ in range(5):
        model.encode(torch.rand(2, 3, 1024, 1024))
    check("encoder BatchNorm buffers unchanged after 5 forwards",
          torch.equal(before, model.image_encoder.bn.running_mean))

    print("\n=== B1/B2/B3: targets ===")
    batch = fake_batch(2)
    w, l, p = batch["gt_word_mask_lr"], batch["gt_line_mask"], batch["gt_para_mask"]
    check("targets are strictly binary",
          set(torch.unique(batch["gt_word_mask"]).tolist()) <= {0.0, 1.0})
    check("word / line / para are different shapes",
          float(w.mean()) < float(l.mean()) < float(p.mean()),
          f"{w.mean():.4f} < {l.mean():.4f} < {p.mean():.4f}")
    check("containment word<=line<=para holds exactly",
          float((w * (1 - l)).sum()) == 0 and float((l * (1 - p)).sum()) == 0)

    print("\n=== A5: batched forward ===")
    model.eval()
    with torch.no_grad():
        out = model.forward_hierarchical(batch)
        looped = [
            model.forward_hierarchical({
                "image": batch["image"][i:i + 1],
                "point_coords": batch["point_coords"][i:i + 1],
                "point_labels": batch["point_labels"][i:i + 1],
            })
            for i in range(batch["image"].shape[0])
        ]
    delta = max(
        float((out[k] - torch.cat([o[k] for o in looped])).abs().max())
        for k in ("word", "line", "para", "word_hr")
    )
    check("batched output equals per-image loop", delta < 1e-4, f"max delta {delta:.2e}")
    check("output shapes match Hi-SAM's H-Decoder",
          out["word_hr"].shape[-1] == 384 and out["line"].shape[-1] == 256,
          f"word_hr {tuple(out['word_hr'].shape)}, line {tuple(out['line'].shape)}")

    print("\n=== A4: no dead parameters ===")
    model.train()
    criterion = HierarchicalLoss()
    out = model.forward_hierarchical(batch)
    loss, logs = criterion(out, batch)
    loss.backward()
    try:
        assert_no_dead_parameters(model)
        check("every trainable parameter receives a gradient", True)
    except AssertionError as exc:
        check("every trainable parameter receives a gradient", False, str(exc)[:200])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_trainable = sum(
        p.numel() for p in model.image_encoder.parameters() if p.requires_grad
    )
    check("encoder contributes no trainable parameters", encoder_trainable == 0)
    print(f"        trainable (excl. stub encoder): {trainable:,}")

    print("\n=== loss ===")
    check("loss is finite", bool(np.isfinite(logs["total"])), f"total={logs['total']:.4f}")
    check("per-level losses are reported separately",
          all(k in logs for k in ("word", "line", "para")))

    print("\n=== C2/C3 available behind flags ===")
    strict = HierarchicalLoss(use_tversky=True, weight_containment=0.5)
    _, logs2 = strict(out, batch)
    check("Tversky + containment run and log", "containment" in logs2,
          f"containment={logs2.get('containment', float('nan')):.4f}")

    print("\n=== S-Decoder config (opt-in) ===")
    s_model = MobileHiSAM(
        checkpoint_path=None, img_size=1024, embed_dim=256,
        enable_hierarchical=True, enable_s_decoder=True,
    )
    check("ModalAligner built only with the S-Decoder",
          s_model.modal_aligner is not None and model.modal_aligner is None)
    s_batch = fake_batch(1, include_text_mask=True)   # HR branch is 1024^2; keep it small
    check("dataset emits the pixel-level union target", "gt_text_mask" in s_batch)

    s_model.train()
    s_out = s_model.forward_hierarchical(s_batch)
    check("S-Decoder emits a 1024^2 pixel-level mask",
          s_out["pixel_hr"].shape[-2:] == s_batch["gt_text_mask"].shape[-2:],
          f"pred {tuple(s_out['pixel_hr'].shape)} vs gt {tuple(s_batch['gt_text_mask'].shape)}")
    s_loss, s_logs = HierarchicalLoss()(s_out, s_batch)
    s_loss.backward()
    aligner_grads = [
        p.grad is not None for p in s_model.modal_aligner.parameters() if p.requires_grad
    ]
    check("ModalAligner receives gradient through the S-Decoder",
          all(aligner_grads) and len(aligner_grads) > 0,
          f"{sum(aligner_grads)}/{len(aligner_grads)} tensors")
    try:
        assert_no_dead_parameters(s_model)
        check("S-Decoder config has no dead parameters", True)
    except AssertionError as exc:
        check("S-Decoder config has no dead parameters", False, str(exc)[:160])

    print("\n" + "=" * 64)
    failed = [n for n, ok in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
