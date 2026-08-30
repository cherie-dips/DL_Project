"""
Uncertainty-based point sampling, as Hi-SAM's ``loss_hi_masks`` uses.

Hi-SAM does not compute its hierarchical mask losses densely. It follows
Mask2Former/PointRend: sample ``num_points`` locations biased toward uncertain
regions (logits near zero, i.e. the decision boundary), and evaluate focal and
dice only there. Their constants are ``num_points=128*128``,
``oversample_ratio=3.0``, ``importance_sample_ratio=0.75``.

This matters beyond efficiency. Boundary-weighted sampling changes what the loss
optimises relative to a dense average, so a dense loss and this one do not
produce the same model even with identical weights.
"""

from typing import Callable

import torch
import torch.nn.functional as F


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Uncertainty as negative distance from the decision boundary.

    ``logits`` is (N, 1, P); returns (N, 1, P). A logit near 0 is maximally
    uncertain, so -|logit| is largest there.
    """
    assert logits.shape[1] == 1
    return -torch.abs(logits)


def point_sample(inputs: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Sample a feature map at continuous coordinates in [0, 1].

    ``inputs`` is (N, C, H, W), ``point_coords`` is (N, P, 2) with x, y in [0, 1].
    Returns (N, C, P).

    Bilinear interpolation is written out rather than delegated to
    ``F.grid_sample`` because MPS has no ``grid_sampler_2d_backward``, which
    makes the backward pass fail outright on Apple silicon. This matches
    grid_sample's ``align_corners=False`` convention with zero padding, and is
    numerically equivalent - see the equivalence check in the tests.
    """
    if point_coords.dim() == 4:
        point_coords = point_coords.squeeze(2)

    N, C, H, W = inputs.shape
    x = point_coords[..., 0] * W - 0.5
    y = point_coords[..., 1] * H - 0.5

    x0, y0 = torch.floor(x), torch.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0

    def gather(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        # Out-of-range samples contribute zero, as grid_sample's zero padding does.
        valid = ((xi >= 0) & (xi <= W - 1) & (yi >= 0) & (yi <= H - 1)).unsqueeze(1)
        xc = xi.clamp(0, W - 1).long()
        yc = yi.clamp(0, H - 1).long()
        flat = (yc * W + xc).unsqueeze(1).expand(-1, C, -1)      # (N, C, P)
        return inputs.reshape(N, C, H * W).gather(2, flat) * valid

    top = gather(x0, y0) * (1 - wx).unsqueeze(1) + gather(x1, y0) * wx.unsqueeze(1)
    bot = gather(x0, y1) * (1 - wx).unsqueeze(1) + gather(x1, y1) * wx.unsqueeze(1)
    return top * (1 - wy).unsqueeze(1) + bot * wy.unsqueeze(1)


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func: Callable[[torch.Tensor], torch.Tensor],
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """Sample points, most of them where the prediction is uncertain.

    Oversamples ``num_points * oversample_ratio`` uniformly, keeps the most
    uncertain ``importance_sample_ratio`` fraction, and fills the remainder
    uniformly at random. Returns (N, num_points, 2) coordinates in [0, 1].
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)

    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    # Uncertainty is computed on the sampled logits, not interpolated from a
    # precomputed uncertainty map - interpolating uncertainty and taking
    # uncertainty of an interpolation are not the same thing.
    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )

    if num_random_points > 0:
        point_coords = torch.cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)],
            dim=1,
        )
    return point_coords
