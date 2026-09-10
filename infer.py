from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from movenet.decode import decode_multipose
from movenet.model import MoveNetMultiPose


COLORS = [(255, 70, 70), (60, 220, 60), (70, 160, 255), (230, 90, 230)]


def parse_args():
    p = argparse.ArgumentParser(description="Infer and visualize parking-space quadrilaterals")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="input image or image directory")
    p.add_argument("--output", default="runs/inference")
    p.add_argument("--input-size", type=int, help="override size stored in checkpoint")
    p.add_argument("--max-spaces", type=int, default=20)
    p.add_argument("--threshold", type=float, default=.15)
    p.add_argument("--search-radius", type=int, default=20,
                   help="keypoint heatmap search radius in stride-4 cells")
    p.add_argument("--distance-sigma", type=float, default=6.,
                   help="strength of regression/heatmap association in output cells")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def preprocess(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[dy:dy+nh, dx:dx+nw] = resized
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    tensor = ((canvas.astype(np.float32) / 255. - mean) / std).transpose(2, 0, 1)
    return torch.from_numpy(tensor).unsqueeze(0), scale, dx, dy


def restore_point(point, scale, dx, dy, width, height):
    x = np.clip((point[0] - dx) / scale, 0, width - 1)
    y = np.clip((point[1] - dy) / scale, 0, height - 1)
    return int(round(x)), int(round(y))


def draw_predictions(image, persons, scale, dx, dy):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for instance, person in enumerate(persons, 1):
        points = [restore_point(p, scale, dx, dy, w, h) for p in person["keypoints"]]
        polygon = np.asarray(points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, (0, 255, 255), 3, cv2.LINE_AA)
        center = tuple(np.asarray(points).mean(0).astype(int))
        cv2.circle(canvas, center, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"P{instance} {person['score']:.2f}",
                    (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 255, 255), 2, cv2.LINE_AA)
        for k, ((x, y), raw) in enumerate(zip(points, person["keypoints"]), 1):
            color = COLORS[k - 1]
            cv2.circle(canvas, (x, y), 7, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{k}:{raw[2]:.2f}", (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)
    return canvas


def image_paths(path: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    if path.is_file(): return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    size = args.input_size or int(saved_args.get("input_size", 512))
    model = MoveNetMultiPose(num_keypoints=4)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    paths = image_paths(Path(args.input))
    if not paths: raise RuntimeError(f"No input images found at {args.input}")
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable image: {path}"); continue
        tensor, scale, dx, dy = preprocess(image, size)
        with torch.inference_mode():
            heads = model(tensor.to(device))
            persons = decode_multipose(heads, args.max_spaces, args.threshold,
                                       args.search_radius, distance_sigma=args.distance_sigma)[0]
        rendered = draw_predictions(image, persons, scale, dx, dy)
        destination = output / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(destination), rendered)
        print(f"{path.name}: {len(persons)} spaces -> {destination}")


if __name__ == "__main__":
    main()
