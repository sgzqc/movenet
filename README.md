# PyTorch MoveNet MultiPose Lightning reproduction

An independently trainable reproduction of the publicly documented MoveNet
design. It uses MobileNetV2, a stride-4 FPN, center/keypoint heatmaps,
center-to-keypoint regression and per-keypoint offsets. Center offset and box
size heads are added for multi-person boxes.

This is **not an exact reimplementation of Google's private training code**.
Google publishes the exported model and a high-level SinglePose architecture,
but not the complete MultiPose Lightning training graph, loss recipe or data.

## Parking-space data (default)

Place matching image/text pairs in one directory. Each label line contains a
class name followed by four ordered corner coordinates:

```text
data/training/result_down_2300.jpg
data/training/result_down_2300.txt

parking_space x1 y1 x2 y2 x3 y3 x4 y4
```

Train with:

```powershell
C:\Users\zhaoq\miniconda3\envs\py312\python.exe train.py `
  --dataset parking --data data/training --input-size 512 `
  --batch-size 1 --epochs 200 --pretrained-backbone
```

Use `--repeat 100` to repeat the samples logically without duplicating files.
Training uses FP32 by default. Pass `--amp` explicitly to enable CUDA mixed precision.

Load an offline TorchVision MobileNetV2 checkpoint without network access:

```powershell
C:\Users\zhaoq\miniconda3\envs\py312\python.exe train.py `
  --dataset parking --data data/training `
  --pretrained-path C:\Code\movenet\pretrained
```

`--pretrained-path` accepts either the directory or the full `.pth` filename
and takes precedence over `--pretrained-backbone`.

## Inference visualization

```powershell
C:\Users\zhaoq\miniconda3\envs\py312\python.exe infer.py `
  --checkpoint runs\parking_pretrained_repeat100_e30_fp32\best.pt `
  --input data\training\result_down_2300.jpg `
  --output runs\parking_visualization
```

The output shows yellow parking polygons, numbered color-coded corners, center
points and confidence scores. `--input` can also be an image directory.
Keypoint heatmap peaks are weighted by their distance from each instance's
regressed corners. Tune `--distance-sigma` (default 6 output cells) when
adjacent parking spaces steal one another's corners.

The four corner indices are learned exactly in the order stored in each label.
Use one consistent clockwise/counter-clockwise convention and the same starting
corner throughout the dataset.

## Optional COCO data

Download COCO 2017 `train2017` and `person_keypoints_train2017.json`:

```text
datasets/coco/
  train2017/
  annotations/person_keypoints_train2017.json
```

## Train

```powershell
C:\Users\zhaoq\miniconda3\envs\py312\python.exe train.py `
  --dataset coco `
  --images datasets/coco/train2017 `
  --annotations datasets/coco/annotations/person_keypoints_train2017.json `
  --batch-size 32 --epochs 100 --pretrained-backbone
```

Use `--input-size 256` for the Lightning-like default. Larger multiples of 32
trade speed for accuracy. Checkpoints are saved under `runs/multipose`.

## Heads

| output | channels | supervision/use |
|---|---:|---|
| center_heatmap | 1 | person keypoint-mean centers |
| keypoint_regression | 34 | center-to-joint y/x vectors |
| keypoint_heatmap | 17 | COCO joint heatmaps |
| keypoint_offset | 34 | sub-grid joint refinement |
| center_offset | 2 | sub-grid center refinement |
| box_size | 2 | person box height/width |

Run `python smoke_test.py` for a forward, loss, backward and decode check.
