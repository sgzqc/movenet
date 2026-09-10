from __future__ import annotations
import torch
from torch.nn import functional as F


@torch.no_grad()
def decode_multipose(outputs: dict[str, torch.Tensor], max_people: int = 6,
                     center_threshold: float = .15, search_radius: int = 20,
                     stride: int = 4, distance_sigma: float = 6.) -> list[list[dict]]:
    """Decode heads into pixel-space boxes and COCO keypoints."""
    centers = outputs["center_heatmap"].sigmoid()
    centers = centers * (centers == F.max_pool2d(centers, 3, 1, 1))
    khm = outputs["keypoint_heatmap"].sigmoid()
    bsz, _, h, w = centers.shape
    num_keypoints = khm.shape[1]
    results = []
    for b in range(bsz):
        scores, inds = centers[b, 0].flatten().topk(min(max_people, h*w))
        persons = []
        for score, ind in zip(scores, inds):
            if score < center_threshold: continue
            iy, ix = int(ind // w), int(ind % w)
            # Original MoveNet four-head design: the center is only an
            # integer heatmap index used to sample the regression field.
            cy, cx = float(iy), float(ix)
            reg = outputs["keypoint_regression"][b, :, iy, ix].view(num_keypoints, 2)
            offsets = outputs["keypoint_offset"][b].view(num_keypoints, 2, h, w)
            points = []
            for k in range(num_keypoints):
                py, px = cy + reg[k,0], cx + reg[k,1]
                base_y = max(0, min(h - 1, int(py)))
                base_x = max(0, min(w - 1, int(px)))
                y0, y1 = max(0, base_y-search_radius), min(h, base_y+search_radius+1)
                x0, x1 = max(0, base_x-search_radius), min(w, base_x+search_radius+1)
                patch = khm[b,k,y0:y1,x0:x1]
                # MoveNet-style instance association: a heatmap peak is useful
                # only when it agrees with this center's regressed keypoint.
                # Pure argmax can steal a high-confidence corner from an
                # adjacent parking space, especially at shared vertices.
                yy = torch.arange(y0, y1, device=patch.device, dtype=patch.dtype)[:,None]
                xx = torch.arange(x0, x1, device=patch.device, dtype=patch.dtype)[None,:]
                distance2 = (yy - py).square() + (xx - px).square()
                association = torch.exp(-distance2 / (2 * distance_sigma ** 2))
                _, qi = (patch * association).flatten().max(0)
                qy, qx = y0 + int(qi // patch.shape[1]), x0 + int(qi % patch.shape[1])
                ks = khm[b,k,qy,qx]
                oy, ox = offsets[k,:,qy,qx]
                points.append([(qx + float(ox))*stride, (qy + float(oy))*stride, float(ks)])
            xy = torch.tensor([[p[0], p[1]] for p in points])
            box = [float(xy[:,0].min()), float(xy[:,1].min()),
                   float(xy[:,0].max()), float(xy[:,1].max())]
            persons.append({"score":float(score), "box":box, "keypoints":points})
        results.append(persons)
    return results
