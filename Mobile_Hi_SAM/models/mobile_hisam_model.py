"""
Mobile-Hi-SAM: MobileSAM's TinyViT encoder driving Hi-SAM's decoders.

The encoder is the only component that differs from Hi-SAM, which is what makes
the comparison to the published Hi-SAM numbers attributable to the encoder swap.

Pipeline:
    image -> MobileSAMEncoder (frozen) -> Adapter -> HiDecoder
                                       -> ModalAligner (self-generated prompts)
                                       -> PromptEncoder (frozen, SAM weights)
    optional: -> MaskDecoder (S-Decoder) for the 1024^2 pixel-level branch
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..adapters.adapter import build_adapter
from .mask_decoder import HiDecoder, MaskDecoder
from .mobile_encoder import MobileSAMEncoder
from .modal_aligner import ModalAligner
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

# Native output resolutions of Hi-SAM's H-Decoder. Losses are computed at these
# resolutions against ground truth rasterised there directly, so that neither the
# prediction is upsampled nor the target downsampled before they are compared.
HI_MASK_SIZE = 256   # word / line / paragraph masks from the hypernetwork path
HI_WORD_SIZE = 384   # refined word branch (word_mask_dc -> interpolate -> refine)
HR_MASK_SIZE = 1024  # S-Decoder pixel-level branch, when enabled


class MobileHiSAM(nn.Module):
    """Hi-SAM with MobileSAM's TinyViT encoder in place of SAM's ViT-H."""

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        img_size: int = 1024,
        embed_dim: int = 256,
        prompt_embed_dim: int = 256,
        enable_hierarchical: bool = True,
        enable_s_decoder: bool = False,
        init_decoder_from_sam: bool = True,
        transformer_mlp_dim: int = 2048,
        freeze_encoder: bool = True,
        adapter: str = "linear",
    ):
        super().__init__()

        self.img_size = img_size
        self.enable_hierarchical = enable_hierarchical
        self.enable_s_decoder = enable_s_decoder
        self.freeze_encoder = freeze_encoder

        # ------------------------------------------------------------------
        # 1. MobileSAM encoder (frozen; see train() for the buffer freeze too)
        # ------------------------------------------------------------------
        print("[MobileHiSAM] Loading MobileSAM encoder...")
        self.image_encoder = MobileSAMEncoder(
            checkpoint_path=checkpoint_path,
            img_size=img_size,
            out_chans=embed_dim,
        )
        # Hi-SAM trains its whole ViT-H encoder plus in-block adapters. Freezing
        # TinyViT entirely is a second variable on top of the encoder swap, so it
        # is a flag rather than a fixed choice - see REMEDIATION_PLAN.md.
        for p in self.image_encoder.parameters():
            p.requires_grad = not freeze_encoder
        if not freeze_encoder:
            # TinyViT is constructed with num_classes=1000 and so carries an
            # ImageNet classification head. Segmentation never reaches it, so
            # leaving it trainable would put 320,640 dead parameters in the
            # optimiser - exactly the bug this rewrite removed elsewhere.
            for name, param in self.image_encoder.named_parameters():
                if name.startswith(("encoder.head.", "encoder.norm_head.")):
                    param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. Adapter (trainable)
        # ------------------------------------------------------------------
        print(f"[MobileHiSAM] Building adapter ({adapter})...")
        self.adapter_kind = adapter
        self.adapter = build_adapter(adapter, in_dim=embed_dim, out_dim=embed_dim)

        # ------------------------------------------------------------------
        # 3. Modal Aligner (trainable) - self-generated prompts
        #
        # In Hi-SAM the aligner exists to prompt the S-Decoder; the H-Decoder
        # takes point prompts. Without an S-Decoder it has no consumer, and
        # building it anyway recreates the dead-parameter bug this rewrite
        # removes. It is therefore tied to enable_s_decoder.
        # ------------------------------------------------------------------
        if enable_s_decoder:
            print("[MobileHiSAM] Building ModalAligner...")
            self.modal_aligner = ModalAligner(
                transformer_dim=embed_dim,
                prompt_len=12,
                nhead=8,
                dropout=0.1,
                attn_layers=1,
            )
        else:
            self.modal_aligner = None

        # ------------------------------------------------------------------
        # 4. Prompt Encoder (frozen, initialised from the SAM checkpoint)
        # ------------------------------------------------------------------
        print("[MobileHiSAM] Loading PromptEncoder...")
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(img_size // 16, img_size // 16),  # 64x64
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # 5. Hi-SAM decoders (trainable)
        #
        # Each decoder gets its OWN TwoWayTransformer. Sharing one instance
        # between the S- and H-Decoder is a ~3.29M parameter deviation in the
        # component we are trying to hold fixed against Hi-SAM.
        # ------------------------------------------------------------------
        def _make_transformer():
            return TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                num_heads=8,
                mlp_dim=transformer_mlp_dim,
            )

        if enable_hierarchical:
            print("[MobileHiSAM] Building HiDecoder (H-Decoder)...")
            self.hi_decoder = HiDecoder(
                transformer_dim=embed_dim,
                transformer=_make_transformer(),
                num_multimask_outputs=3,
            )
        else:
            self.hi_decoder = None

        if enable_s_decoder:
            print("[MobileHiSAM] Building MaskDecoder (S-Decoder)...")
            self.mask_decoder = MaskDecoder(
                transformer_dim=embed_dim,
                transformer=_make_transformer(),
                num_multimask_outputs=3,
            )
        else:
            # Not constructed at all - an unused decoder in the optimiser is
            # exactly the dead-parameter bug this rewrite removes.
            self.mask_decoder = None

        # ------------------------------------------------------------------
        # 6. Pixel normalization
        #
        # ToTensor() produces [0, 1]; SAM's published constants are for
        # [0, 255]. Scale the constants, not the tensor.
        # ------------------------------------------------------------------
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1) / 255.0,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) / 255.0,
        )

        # ------------------------------------------------------------------
        # 7. Initialise the frozen prompt encoder (and optionally the decoders)
        #    from the SAM checkpoint instead of leaving them random.
        # ------------------------------------------------------------------
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_sam_components(
                checkpoint_path, init_decoder=init_decoder_from_sam
            )
        else:
            print(
                "[MobileHiSAM] WARNING: no checkpoint given - the frozen prompt "
                "encoder keeps random weights it can never learn away from."
            )

    # ----------------------------------------------------------------------
    # Checkpoint loading
    # ----------------------------------------------------------------------
    def load_sam_components(self, checkpoint_path: str, init_decoder: bool = True):
        """Load prompt_encoder (and optionally decoder) weights from a SAM checkpoint.

        The prompt encoder is frozen, so whatever it is initialised with is what
        it uses forever. Leaving it random means the point prompt carries fixed
        but arbitrary information and ``get_dense_pe()`` feeds a random basis to
        the transformer.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and not any(
            k.startswith("image_encoder.") for k in ckpt
        ):
            ckpt = ckpt["model"]

        def _subtree(prefix: str) -> Dict[str, torch.Tensor]:
            return {
                k[len(prefix):]: v
                for k, v in ckpt.items()
                if k.startswith(prefix)
            }

        # --- prompt encoder: must be complete, it can never be corrected ---
        pe_state = _subtree("prompt_encoder.")
        if pe_state:
            missing, unexpected = self.prompt_encoder.load_state_dict(
                pe_state, strict=False
            )
            if missing or unexpected:
                raise RuntimeError(
                    "[MobileHiSAM] Incomplete prompt_encoder load from "
                    f"{checkpoint_path}. missing={list(missing)} "
                    f"unexpected={list(unexpected)}. The prompt encoder is "
                    "frozen, so a partial load is permanent."
                )
            print(
                f"[MobileHiSAM] PromptEncoder initialised from checkpoint "
                f"({len(pe_state)} tensors)"
            )
        else:
            print(
                "[MobileHiSAM] WARNING: checkpoint has no prompt_encoder.* keys; "
                "the frozen prompt encoder stays random."
            )

        # --- decoders: free pretrained signal, partial overlap is expected ---
        if not init_decoder:
            return
        md_state = _subtree("mask_decoder.")
        if not md_state:
            print("[MobileHiSAM] Checkpoint has no mask_decoder.* keys; decoders start from scratch.")
            return

        for name, module in (("hi_decoder", self.hi_decoder),
                             ("mask_decoder", self.mask_decoder)):
            if module is None:
                continue
            own = module.state_dict()
            usable = {
                k: v for k, v in md_state.items()
                if k in own and own[k].shape == v.shape
            }
            module.load_state_dict(usable, strict=False)
            print(
                f"[MobileHiSAM] {name}: initialised {len(usable)}/{len(own)} "
                f"tensors from SAM's mask_decoder"
            )

    # ----------------------------------------------------------------------
    # Frozen means frozen: requires_grad=False stops weight updates but not
    # BatchNorm running statistics, and TinyViT is full of them.
    # ----------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.image_encoder.eval()
        else:
            # Fine-tuning needs the encoder in train mode: TinyViT's attention
            # caches its relative-position bias table in eval() and only indexes
            # the parameter in train(), so an eval-mode encoder silently gives
            # attention_biases no gradient at all.
            self.image_encoder.train(mode)
            # But BatchNorm statistics still stay fixed - a batch of 2 is far too
            # small to re-estimate running statistics from.
            for module in self.image_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        self.prompt_encoder.eval()
        return self

    def parameter_groups(self, base_lr: float, encoder_lr: Optional[float] = None):
        """Optimiser groups, so a fine-tuned encoder can use a smaller step."""
        encoder = [p for p in self.image_encoder.parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder}
        rest = [p for p in self.parameters()
                if p.requires_grad and id(p) not in encoder_ids]
        groups = [{"params": rest, "lr": base_lr}]
        if encoder:
            groups.append({"params": encoder, "lr": encoder_lr or base_lr * 0.1})
        return groups

    @property
    def device(self):
        return self.pixel_mean.device

    # ----------------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------------
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise and pad a (B, 3, H, W) batch to the encoder's square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return x

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def _empty_dense(self, batch_size: int) -> torch.Tensor:
        """The no-mask dense embedding, broadcast to a batch."""
        return self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, *self.prompt_encoder.image_embedding_size
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) in [0, 1] -> adapted embeddings (B, C, 64, 64)."""
        return self.adapter(self.image_encoder(self.preprocess(images)))

    def decode_prompts(
        self,
        embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the H-Decoder for N prompts against ONE image's embeddings.

        At inference many prompts hit the same image, so the encoder must run
        once and only the decoder per prompt. Passing an expanded image batch
        instead re-runs the encoder N times, which dominates the cost.

        Args:
            embeddings: (1, C, H, W) from ``encode``.
            point_coords: (N, 1, 2) in padded-image coordinates.
            point_labels: (N, 1).
        """
        if self.hi_decoder is None:
            raise RuntimeError("decode_prompts requires enable_hierarchical=True")

        # Hi-SAM discards the prompt encoder's dense output and passes no
        # dense_prompt_embeddings to either decoder, so no_mask_embed is never
        # added to src. Passing it changes what the transformer sees.
        sparse, _ = self.prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        masks, iou_pred, word_masks = self.hi_decoder(
            image_embeddings=embeddings,       # (1,C,H,W) -> repeated to N inside
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            multimask_output=True,
        )
        return {
            "word": masks[:, 0:1],
            "line": masks[:, 1:2],
            "para": masks[:, 2:3],
            "word_hr": word_masks,
            "iou": iou_pred,
        }

    def forward_hierarchical(
        self,
        batch: Dict[str, Any],
        use_modal_aligner: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Batched hierarchical forward.

        Args:
            batch: dict with ``image`` (B,3,H,W) in [0,1] and, unless
                ``use_modal_aligner``, ``point_coords`` (B,N,2) and
                ``point_labels`` (B,N).
            use_modal_aligner: take prompts from the ModalAligner instead of
                supplied points, so inference needs no ground truth.

        Returns:
            dict of logits. ``word``/``line``/``para`` are (B,1,256,256);
            ``word_hr`` is (B,1,384,384); ``iou`` is (B,3).
        """
        if self.hi_decoder is None:
            raise RuntimeError("forward_hierarchical requires enable_hierarchical=True")

        embeddings = self.encode(batch["image"])

        if use_modal_aligner:
            if self.modal_aligner is None:
                raise RuntimeError(
                    "use_modal_aligner=True requires enable_s_decoder=True; the "
                    "aligner is not built otherwise."
                )
            sparse = self.modal_aligner(embeddings)
        else:
            points = (batch["point_coords"], batch["point_labels"])
            sparse, _ = self.prompt_encoder(points=points, boxes=None, masks=None)

        # No dense_prompt_embeddings: Hi-SAM passes none to either decoder.
        masks, iou_pred, word_masks = self.hi_decoder(
            image_embeddings=embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            multimask_output=True,
        )

        # multimask_output=True slices [1:], so the three surviving tokens are
        # word / line / paragraph in that order. word_masks comes from token 1,
        # the same token that produces masks[:, 0].
        out = {
            "word": masks[:, 0:1],
            "line": masks[:, 1:2],
            "para": masks[:, 2:3],
            "word_hr": word_masks,
            "iou": iou_pred,
        }

        if self.enable_s_decoder:
            # Hi-SAM prompts the S-Decoder from the ModalAligner, not from the
            # point. This is also what puts the aligner in the gradient path.
            aligner_sparse = self.modal_aligner(embeddings)
            s_masks, hr_masks, s_iou, s_iou_hr = self.mask_decoder(
                image_embeddings=embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=aligner_sparse,
                multimask_output=False,
            )
            # Text foreground is unambiguous, so single-mask mode: token 0 feeds
            # both the coarse 256^2 mask and the refined 1024^2 one, and both are
            # supervised.
            out["pixel"] = s_masks
            out["pixel_hr"] = hr_masks
            out["pixel_iou_lr"] = s_iou      # quality of the 256^2 mask
            out["pixel_iou"] = s_iou_hr      # quality of the 1024^2 mask

        return out

    def forward_grouped(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Hi-SAM's batching: each image carries its own group of prompts.

        The encoder runs once per image; the H-Decoder then decodes that image's
        whole prompt group against the single embedding. Hierarchical outputs are
        concatenated across the batch and line up with the concatenated targets.
        The S-Decoder runs once per image, prompted by the ModalAligner.
        """
        if self.hi_decoder is None:
            raise RuntimeError("forward_grouped requires enable_hierarchical=True")

        embeddings = self.encode(batch["image"])          # (B, C, 64, 64)
        counts = batch["prompt_counts"]
        coords, labels = batch["point_coords"], batch["point_labels"]

        groups, offset = [], 0
        for i, n in enumerate(counts):
            groups.append(
                self.decode_prompts(
                    embeddings[i:i + 1], coords[offset:offset + n], labels[offset:offset + n]
                )
            )
            offset += n

        out = {
            key: torch.cat([g[key] for g in groups], dim=0)
            for key in ("word", "line", "para", "word_hr", "iou")
        }

        if self.enable_s_decoder:
            # One call returns both the coarse and the high-res branch.
            aligner_sparse = self.modal_aligner(embeddings)
            s_masks, hr_masks, s_iou, s_iou_hr = self.mask_decoder(
                image_embeddings=embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=aligner_sparse,
                multimask_output=False,
            )
            out["pixel"] = s_masks
            out["pixel_hr"] = hr_masks
            out["pixel_iou_lr"] = s_iou
            out["pixel_iou"] = s_iou_hr

        return out

    def forward(self, batch: Dict[str, Any], **kwargs):
        return self.forward_hierarchical(batch, **kwargs)

    # ----------------------------------------------------------------------
    # Postprocessing (inference only)
    # ----------------------------------------------------------------------
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Undo the resize-and-pad: crop to the unpadded region, then rescale.

        Args:
            masks: (B, C, h, w) logits at decoder resolution.
            input_size: (h, w) the image actually occupies inside the padded
                square, i.e. before padding but after the aspect-preserving resize.
            original_size: (H, W) of the source image.
        """
        masks = nn.functional.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = nn.functional.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    # ----------------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------------
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_report(self) -> str:
        lines = []
        total = trainable = 0
        for name, module in [
            (f"image_encoder ({'frozen' if self.freeze_encoder else 'fine-tuned'})",
             self.image_encoder),
            ("adapter", self.adapter),
            ("modal_aligner", self.modal_aligner),
            ("prompt_encoder (frozen)", self.prompt_encoder),
            ("hi_decoder", self.hi_decoder),
            ("mask_decoder", self.mask_decoder),
        ]:
            if module is None:
                lines.append(f"  {name:<26} not built")
                continue
            n = sum(p.numel() for p in module.parameters())
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total += n
            trainable += t
            lines.append(f"  {name:<26} {n:>12,}  trainable {t:>12,}")
        lines.append(f"  {'TOTAL':<26} {total:>12,}  trainable {trainable:>12,}")
        return "\n".join(lines)


def pick_device(preference: Optional[str] = None) -> torch.device:
    """Choose the best available accelerator.

    Order: explicit preference, then CUDA, then Apple Metal (MPS), then CPU.
    """
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_supported(device: torch.device) -> bool:
    """Mixed precision with loss scaling is only wired up for CUDA here.

    torch.amp.GradScaler supports CUDA and CPU, not MPS. On Apple silicon the
    frozen encoder means activation memory is already small (~0.2 GB at batch 4),
    so AMP would buy little and risks silent dtype problems.
    """
    return device.type == "cuda"


def structurally_unused_prefixes(model: nn.Module) -> Tuple[str, ...]:
    """Parameters SAM's token layout strands, given how each decoder is called.

    Both decoders keep SAM's 4-token layout (num_mask_tokens =
    num_multimask_outputs + 1) but use different slices of it:

      * H-Decoder, multimask_output=True  -> slice [1:], so token 0's
        hypernetwork is unused.
      * S-Decoder, multimask_output=False -> slice [0:1], so tokens 1-3's
        hypernetworks are unused. Token 0 also feeds the HR branch.

    This is Hi-SAM's own behaviour, kept for parity. It is derived from the
    model's configuration rather than hard-coded so that the allow-list cannot
    quietly absorb a genuine dead-parameter bug.
    """
    prefixes = []
    if getattr(model, "hi_decoder", None) is not None:
        prefixes.append("hi_decoder.output_hypernetworks_mlps.0.")
    if getattr(model, "mask_decoder", None) is not None:
        prefixes += [f"mask_decoder.output_hypernetworks_mlps.{i}." for i in (1, 2, 3)]
    return tuple(prefixes)


def assert_no_dead_parameters(model: nn.Module, allow: Optional[Tuple[str, ...]] = None):
    """Fail loudly if a trainable parameter received no gradient.

    Call once after the first ``loss.backward()``. Catches modules that are
    constructed and optimised but never reached by the forward pass.
    """
    if allow is None:
        allow = structurally_unused_prefixes(model)
    dead = [
        (n, p.numel())
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is None and not n.startswith(allow)
    ]
    if dead:
        total = sum(n for _, n in dead)
        detail = "\n".join(f"    {n} ({c:,})" for n, c in dead[:20])
        raise AssertionError(
            f"{total:,} trainable params across {len(dead)} tensors receive no "
            f"gradient:\n{detail}"
        )
    return True
