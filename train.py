from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from movenet.data import CocoKeypoints
from movenet.parking_data import ParkingSpaces
from movenet.losses import multipose_loss
from movenet.model import MoveNetMultiPose


def parse_args():
    p = argparse.ArgumentParser(description="Train a MoveNet-inspired MultiPose Lightning model")
    p.add_argument("--dataset", choices=["parking", "coco"], default="parking")
    p.add_argument("--data", default="data/training", help="parking image/txt directory")
    p.add_argument("--images", help="COCO train2017 image directory")
    p.add_argument("--annotations", help="person_keypoints_train2017.json")
    p.add_argument("--output", default="runs/multipose")
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--repeat", type=int, default=1,
                   help="logically repeat the dataset without copying files")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrained-backbone", action="store_true")
    p.add_argument("--pretrained-path",
                   help="local MobileNetV2 .pth file or directory; avoids downloading")
    p.add_argument("--amp", action="store_true",
                   help="enable CUDA mixed precision; default is full FP32")
    p.add_argument("--resume")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.input_size % 32: raise ValueError("--input-size must be divisible by 32")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.dataset == "parking":
        dataset = ParkingSpaces(args.data, args.input_size)
        num_keypoints = 4
    else:
        if not args.images or not args.annotations:
            raise ValueError("COCO requires --images and --annotations")
        dataset = CocoKeypoints(args.images, args.annotations, args.input_size, train=True)
        num_keypoints = 17
    if args.repeat < 1: raise ValueError("--repeat must be at least 1")
    if args.repeat > 1: dataset = ConcatDataset([dataset] * args.repeat)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    model = MoveNetMultiPose(pretrained=args.pretrained_backbone,
                             num_keypoints=num_keypoints,
                             pretrained_path=args.pretrained_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    start, best_loss = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"]); start = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event":"start", "args":vars(args),
                              "samples":len(dataset), "device":str(device),
                              "precision":"amp" if use_amp else "fp32"},
                             ensure_ascii=False) + "\n")
    for epoch in range(start, args.epochs):
        model.train(); sums = {}
        bar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in bar:
            image = batch.pop("image").to(device, non_blocking=True)
            batch.pop("image_id")
            batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=use_amp):
                loss, parts = multipose_loss(model(image), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; checkpoint was not updated")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            scaler.step(optimizer); scaler.update()
            for k,v in {"loss":loss, **parts}.items(): sums[k] = sums.get(k,0.) + float(v.detach())
            bar.set_postfix(loss=f"{float(loss):.4f}")
        scheduler.step()
        metrics = {k:v/len(loader) for k,v in sums.items()}
        record = {"epoch":epoch+1, "lr":scheduler.get_last_lr()[0], **metrics}
        print(record)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
        improved = metrics["loss"] < best_loss
        best_loss = min(best_loss, metrics["loss"])
        state = {"epoch":epoch, "model":model.state_dict(), "optimizer":optimizer.state_dict(),
                 "scheduler":scheduler.state_dict(), "args":vars(args), "metrics":metrics}
        state["best_loss"] = best_loss
        torch.save(state, out / "last.pt")
        if improved:
            torch.save(state, out / "best.pt")


if __name__ == "__main__": main()
