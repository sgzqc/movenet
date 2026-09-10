from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


def _gaussian_radius(height: float, width: float, overlap: float = .7) -> int:
    a1, b1 = 1, height + width
    c1 = width * height * (1 - overlap) / (1 + overlap)
    r1 = (b1 + math.sqrt(max(0., b1 * b1 - 4 * a1 * c1))) / 2
    return max(1, int(r1))


def _draw_gaussian(hm: np.ndarray, center: tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    x = np.arange(diameter, dtype=np.float32) - radius
    g = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * (diameter / 6) ** 2))
    cx, cy = center
    left, right = min(cx, radius), min(hm.shape[1] - cx - 1, radius)
    top, bottom = min(cy, radius), min(hm.shape[0] - cy - 1, radius)
    if min(left, right, top, bottom) < 0:
        return
    patch = hm[cy-top:cy+bottom+1, cx-left:cx+right+1]
    np.maximum(patch, g[radius-top:radius+bottom+1, radius-left:radius+right+1], out=patch)


class CocoKeypoints(Dataset):
    """COCO person-keypoint dataset with stride-4 dense targets."""

    def __init__(self, images: str, annotations: str, input_size: int = 256,
                 max_people: int = 30, train: bool = True):
        self.root = Path(images)
        self.coco = COCO(annotations)
        self.ids = sorted(self.coco.getImgIds(catIds=[1]))
        self.input_size, self.max_people, self.train = input_size, max_people, train
        self.mean = np.array([.485, .456, .406], np.float32)
        self.std = np.array([.229, .224, .225], np.float32)

    def __len__(self): return len(self.ids)

    def _letterbox(self, image: np.ndarray, anns: list[dict]):
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dx, dy = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        canvas[dy:dy+nh, dx:dx+nw] = image
        return canvas, scale, dx, dy

    def __getitem__(self, index: int):
        info = self.coco.loadImgs(self.ids[index])[0]
        image = cv2.imread(str(self.root / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.root / info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = self.coco.getAnnIds(imgIds=[info["id"]], catIds=[1], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        image, scale, dx, dy = self._letterbox(image, anns)
        flip = self.train and np.random.random() < .5
        if flip:
            image = image[:, ::-1].copy()
        target = self._targets(anns, scale, dx, dy, flip)
        image = (image.astype(np.float32) / 255. - self.mean) / self.std
        target["image"] = torch.from_numpy(image.transpose(2, 0, 1))
        target["image_id"] = torch.tensor(info["id"], dtype=torch.int64)
        return target

    def _targets(self, anns, scale: float, dx: int, dy: int, flip: bool):
        s, out = 4, self.input_size // 4
        result = {
            "center_heatmap": np.zeros((1, out, out), np.float32),
            "keypoint_heatmap": np.zeros((17, out, out), np.float32),
            "keypoint_regression": np.zeros((self.max_people, 34), np.float32),
            "center_offset": np.zeros((self.max_people, 2), np.float32),
            "box_size": np.zeros((self.max_people, 2), np.float32),
            "indices": np.zeros(self.max_people, np.int64),
            "person_mask": np.zeros(self.max_people, np.float32),
            "keypoint_mask": np.zeros((self.max_people, 17), np.float32),
            "keypoint_offset": np.zeros((17, 2, out, out), np.float32),
            "keypoint_offset_mask": np.zeros((17, out, out), np.float32),
        }
        people = [a for a in anns if a.get("num_keypoints", 0) > 0 and a["area"] > 16]
        people.sort(key=lambda a: a["area"], reverse=True)
        for n, ann in enumerate(people[:self.max_people]):
            kp = np.asarray(ann["keypoints"], np.float32).reshape(17, 3)
            kp[:, 0] = kp[:, 0] * scale + dx
            kp[:, 1] = kp[:, 1] * scale + dy
            x, y, w, h = ann["bbox"]
            x, y, w, h = x * scale + dx, y * scale + dy, w * scale, h * scale
            if flip:
                kp[:, 0] = self.input_size - 1 - kp[:, 0]
                # COCO left/right pairs.
                order = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
                kp = kp[order]
                x = self.input_size - x - w
            visible = kp[:, 2] > 0
            if not visible.any(): continue
            # MoveNet center: mean of annotated keypoints, clipped to the box.
            center = kp[visible, :2].mean(0) / s
            center[0] = np.clip(center[0], x / s, (x + w) / s)
            center[1] = np.clip(center[1], y / s, (y + h) / s)
            ci = np.floor(center).astype(np.int64)
            if not (0 <= ci[0] < out and 0 <= ci[1] < out): continue
            radius = _gaussian_radius(h / s, w / s)
            _draw_gaussian(result["center_heatmap"][0], tuple(ci), radius)
            result["indices"][n] = ci[1] * out + ci[0]
            result["person_mask"][n] = 1
            result["center_offset"][n] = center[::-1] - ci[::-1]  # y,x
            result["box_size"][n] = [h / s, w / s]
            for k in range(17):
                if not visible[k]: continue
                p = kp[k, :2] / s
                pi = np.floor(p).astype(np.int64)
                if not (0 <= pi[0] < out and 0 <= pi[1] < out): continue
                result["keypoint_regression"][n, 2*k:2*k+2] = p[::-1] - center[::-1]
                result["keypoint_mask"][n, k] = 1
                _draw_gaussian(result["keypoint_heatmap"][k], tuple(pi), max(1, radius // 3))
                result["keypoint_offset"][k, :, pi[1], pi[0]] = p[::-1] - pi[::-1]
                result["keypoint_offset_mask"][k, pi[1], pi[0]] = 1
        for key, value in list(result.items()):
            if isinstance(value, np.ndarray): result[key] = torch.from_numpy(value)
        return result
