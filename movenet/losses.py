from __future__ import annotations
import torch
from torch.nn import functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # FP16 rounds 1-1e-4 to 1, making log(1-p) become log(0). Heatmap
    # focal loss is inexpensive, so always evaluate it in FP32 under AMP.
    targets = targets.float()
    pred = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = targets.eq(1).float()
    neg = targets.lt(1).float()
    neg_weights = (1 - targets).pow(4)
    loss = -(torch.log(pred) * (1 - pred).pow(2) * pos +
             torch.log(1 - pred) * pred.pow(2) * neg_weights * neg)
    return loss.sum() / pos.sum().clamp(min=1)


def _gather(output: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = output.shape
    flat = output.view(b, c, -1).permute(0, 2, 1)
    return flat.gather(1, indices.unsqueeze(-1).expand(-1, -1, c))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < pred.ndim: mask = mask.unsqueeze(-1)
    return (F.l1_loss(pred, target, reduction="none") * mask).sum() / mask.sum().clamp(min=1)


def multipose_loss(outputs, batch, weights=None):
    weights = weights or {"center":1., "kpt_hm":1., "kpt_reg":1., "kpt_off":1.}
    center = focal_loss(outputs["center_heatmap"], batch["center_heatmap"])
    kpt_hm = focal_loss(outputs["keypoint_heatmap"], batch["keypoint_heatmap"])
    idx, pmask = batch["indices"], batch["person_mask"]
    num_keypoints = outputs["keypoint_heatmap"].shape[1]
    kreg = _gather(outputs["keypoint_regression"], idx).view(*idx.shape, num_keypoints, 2)
    kpt_reg = masked_l1(kreg, batch["keypoint_regression"].view(*idx.shape,num_keypoints,2), batch["keypoint_mask"])
    off = outputs["keypoint_offset"].view(outputs["keypoint_offset"].shape[0],num_keypoints,2,*outputs["keypoint_offset"].shape[-2:])
    kpt_off = masked_l1(off, batch["keypoint_offset"], batch["keypoint_offset_mask"].unsqueeze(2))
    parts = {"center":center, "kpt_hm":kpt_hm, "kpt_reg":kpt_reg,
             "kpt_off":kpt_off}
    return sum(weights[k] * v for k, v in parts.items()), parts
