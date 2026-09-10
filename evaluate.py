from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from infer import image_paths, preprocess, restore_point
from movenet.decode import decode_multipose
from movenet.model import MoveNetMultiPose
from movenet.parking_data import ParkingSpaces


def greedy_match(pred: np.ndarray, target: np.ndarray):
    """One-to-one center matching, sufficient for this small evaluation set."""
    pc, tc = pred.mean(1), target.mean(1)
    distances = np.linalg.norm(pc[:, None] - tc[None], axis=2)
    pairs = []
    for flat in np.argsort(distances, axis=None):
        p, t = np.unravel_index(flat, distances.shape)
        if any(a == p or b == t for a, b in pairs):
            continue
        pairs.append((p, t))
        if len(pairs) == min(len(pred), len(target)):
            break
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Evaluate parking-space corner accuracy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--threshold", type=float, default=.8)
    ap.add_argument("--max-spaces", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = int(ckpt.get("args", {}).get("input_size", 512))
    model = MoveNetMultiPose(num_keypoints=4)
    model.load_state_dict(ckpt["model"]); model.to(device).eval()
    all_errors, all_normalized = [], []
    matched = predicted = annotated = 0
    for path in image_paths(Path(args.data)):
        label = path.with_suffix(".txt")
        if not label.exists(): continue
        image = cv2.imread(str(path)); h, w = image.shape[:2]
        tensor, scale, dx, dy = preprocess(image, size)
        with torch.inference_mode():
            persons = decode_multipose(model(tensor.to(device)), args.max_spaces,
                                       args.threshold)[0]
        pred = np.asarray([[restore_point(q, scale, dx, dy, w, h)
                            for q in p["keypoints"]] for p in persons], np.float32)
        target = ParkingSpaces.read_label(label)
        pairs = greedy_match(pred, target) if len(pred) and len(target) else []
        image_errors = []
        for p, t in pairs:
            errors = np.linalg.norm(pred[p] - target[t], axis=1)
            diagonal = np.linalg.norm(np.ptp(target[t], axis=0)).clip(1e-6)
            all_errors.extend(errors); all_normalized.extend(errors / diagonal)
            image_errors.extend(errors)
        predicted += len(pred); annotated += len(target); matched += len(pairs)
        value = float(np.mean(image_errors)) if image_errors else float("nan")
        print(f"{path.name}: pred={len(pred)} gt={len(target)} matched={len(pairs)} mean_px={value:.3f}")
    errors = np.asarray(all_errors)
    if not len(errors): raise RuntimeError("No matched parking-space corners")
    print(f"instances: predicted={predicted} annotated={annotated} matched={matched}")
    print(f"corner mean error: {errors.mean():.3f} px")
    print(f"corner median error: {np.median(errors):.3f} px")
    print(f"corner max error: {errors.max():.3f} px")
    for threshold in (2, 5, 10, 20):
        print(f"PCK@{threshold}px: {(errors <= threshold).mean() * 100:.2f}%")
    print(f"NME (box diagonal): {np.mean(all_normalized) * 100:.4f}%")


if __name__ == "__main__": main()
