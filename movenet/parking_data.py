"""Dataset for IPM parking-space quadrilaterals.

Label format, one instance per line:
parking_space x1 y1 x2 y2 x3 y3 x4 y4
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import _draw_gaussian, _gaussian_radius


class ParkingSpaces(Dataset):
    def __init__(self, root: str, input_size: int = 512, max_spaces: int = 64):
        self.root = Path(root)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples = []
        for image in sorted(p for p in self.root.iterdir() if p.suffix.lower() in extensions):
            label = image.with_suffix(".txt")
            if label.exists(): self.samples.append((image, label))
        if not self.samples:
            raise RuntimeError(f"No image/txt pairs found in {self.root}")
        self.input_size, self.max_spaces = input_size, max_spaces
        self.mean = np.array([.485, .456, .406], np.float32)
        self.std = np.array([.229, .224, .225], np.float32)

    def __len__(self): return len(self.samples)

    @staticmethod
    def read_label(path: Path) -> np.ndarray:
        spaces = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields: continue
            if fields[0] != "parking_space" or len(fields) != 9:
                raise ValueError(f"{path}:{line_no}: expected 'parking_space' plus 8 coordinates")
            spaces.append(np.asarray(fields[1:], np.float32).reshape(4, 2))
        return np.stack(spaces) if spaces else np.empty((0, 4, 2), np.float32)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        spaces = self.read_label(label_path)
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dx, dy = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        image = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        image[dy:dy+nh, dx:dx+nw] = resized
        spaces *= scale
        spaces[..., 0] += dx; spaces[..., 1] += dy
        target = self._targets(spaces)
        image = (image.astype(np.float32) / 255. - self.mean) / self.std
        target["image"] = torch.from_numpy(image.transpose(2,0,1))
        target["image_id"] = torch.tensor(index, dtype=torch.int64)
        return target

    def _targets(self, spaces):
        stride, nk = 4, 4
        out = self.input_size // stride
        r = {
            "center_heatmap":np.zeros((1,out,out),np.float32),
            "keypoint_heatmap":np.zeros((nk,out,out),np.float32),
            "keypoint_regression":np.zeros((self.max_spaces,nk*2),np.float32),
            "indices":np.zeros(self.max_spaces,np.int64),
            "person_mask":np.zeros(self.max_spaces,np.float32),
            "keypoint_mask":np.zeros((self.max_spaces,nk),np.float32),
            "keypoint_offset":np.zeros((nk,2,out,out),np.float32),
            "keypoint_offset_mask":np.zeros((nk,out,out),np.float32),
        }
        # Largest first gives deterministic ownership if two centers share a cell.
        spaces = sorted(spaces, key=lambda q: -float(np.ptp(q[:,0])*np.ptp(q[:,1])))
        occupied = set()
        slot = 0
        for q in spaces:
            if slot >= self.max_spaces: break
            p = q / stride
            center = p.mean(0)
            ci = np.floor(center).astype(np.int64)
            if not (0 <= ci[0] < out and 0 <= ci[1] < out) or tuple(ci) in occupied: continue
            occupied.add(tuple(ci))
            bw, bh = np.ptp(p[:,0]), np.ptp(p[:,1])
            radius = _gaussian_radius(bh, bw)
            _draw_gaussian(r["center_heatmap"][0], tuple(ci), radius)
            r["indices"][slot] = ci[1]*out + ci[0]
            r["person_mask"][slot] = 1
            for k in range(nk):
                pi = np.floor(p[k]).astype(np.int64)
                if not (0 <= pi[0] < out and 0 <= pi[1] < out): continue
                r["keypoint_regression"][slot,2*k:2*k+2] = p[k,::-1] - center[::-1]
                r["keypoint_mask"][slot,k] = 1
                _draw_gaussian(r["keypoint_heatmap"][k], tuple(pi), max(1,radius//3))
                r["keypoint_offset"][k,:,pi[1],pi[0]] = p[k,::-1] - pi[::-1]
                r["keypoint_offset_mask"][k,pi[1],pi[0]] = 1
            slot += 1
        return {k:torch.from_numpy(v) for k,v in r.items()}
