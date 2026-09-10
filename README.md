# MoveNet 风格的 IPM 泊车位四角点检测

这是一个基于 PyTorch 的多泊车位四角点检测项目。网络参考 Google MoveNet 公开设计，使用 MobileNetV2、stride-4 FPN、多分支密集预测以及 regression-guided heatmap 解码，在 IPM 鸟瞰图中检测多个泊车位。

> 本项目是面向泊车位任务的独立实现，并非 Google 未公开的 MultiPose Lightning 训练代码或权重的逐层复现。

## 网络结构

```text
输入图像
└─ MobileNetV2 backbone
   └─ FPN（输出步长 4）
      ├─ 车位中心热图       1 channel
      ├─ 中心到四角点回归   8 channels
      ├─ 四类角点热图       4 channels
      ├─ 四角点局部偏移     8 channels
      ├─ 中心局部偏移       2 channels
      └─ 车位框高宽         2 channels
```

前四个分支对应 MoveNet 的中心、关键点回归、关键点热图和局部偏移思想；中心偏移与车位框分支用于多实例定位。

## 安装

推荐 Python 3.10 或更高版本：

```powershell
pip install -r requirements.txt
```

## 数据格式

图像与标注放在同一目录，并使用相同文件主名：

```text
data/training/
├── result_down_2300.jpg
└── result_down_2300.txt
```

每行包含类别名和四个有序角点：

```text
parking_space x1 y1 x2 y2 x3 y3 x4 y4
```

所有图片必须采用一致的角点起点和顺/逆时针方向。相邻车位可以共享物理角点，但共享点在各车位中的角点编号可能不同。

## 离线预训练权重

下载 TorchVision MobileNetV2 ImageNet 权重：

[mobilenet_v2-7ebf99e0.pth](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth)

放置为 `pretrained/mobilenet_v2-7ebf99e0.pth`。`--pretrained-path` 可以传目录或完整文件路径，本地权重存在时不会联网。

## 训练

使用本地预训练权重、逻辑重复数据 100 次并训练 30 个 epoch：

```powershell
python train.py `
  --dataset parking `
  --data data\training `
  --input-size 512 `
  --repeat 100 `
  --batch-size 10 `
  --workers 0 `
  --epochs 30 `
  --pretrained-path pretrained `
  --output runs\parking_pretrained_repeat100_e30_fp32
```

`--repeat` 只做逻辑重复，不复制文件。训练默认使用 FP32；显式传入 `--amp` 才会启用 CUDA 混合精度。

输出目录包含：

```text
best.pt    # 训练 loss 最低的 checkpoint
last.pt    # 最后一轮 checkpoint
log.txt    # 配置、学习率、总 loss 与各分支 loss
```

断点续训：

```powershell
python train.py [其他参数] --resume runs\parking_pretrained_repeat100_e30_fp32\last.pt
```

## 推理与可视化

```powershell
python infer.py `
  --checkpoint runs\parking_pretrained_repeat100_e30_fp32\best.pt `
  --input data\training\result_down_2300.jpg `
  --output runs\parking_visualization `
  --threshold 0.8 `
  --max-spaces 6
```

`--input` 支持单张图片或图片目录。输出包含泊车位四边形、四个编号角点、车位中心及置信度。

## MoveNet Step 3 解码

直接取角点热图最大值容易关联到相邻车位。本项目使用中心回归得到的粗角点对相应热图做距离加权：

```text
weighted_heatmap(p) = heatmap(p) × exp(-distance(p, regressed_point)² / (2σ²))
```

再从加权热图取峰值并叠加局部 offset。默认参数为 `--search-radius 20` 和 `--distance-sigma 6`，单位是 stride-4 输出特征图网格。正式使用时应根据独立验证集调整。

## COCO 人体关键点模式

代码保留 COCO 17 点数据管线：

```powershell
python train.py `
  --dataset coco `
  --images datasets\coco\train2017 `
  --annotations datasets\coco\annotations\person_keypoints_train2017.json `
  --input-size 256 `
  --batch-size 32 `
  --epochs 100
```

## 快速测试

```powershell
python smoke_test.py
```

该测试覆盖网络前向传播、loss、反向传播和多人解码。

## 注意事项

- 同一张图片重复多次只能验证模型过拟合，不能增加数据多样性。
- 正式训练应增加不同泊车场、视角、光照和车位形状的样本。
- 当前 `best.pt` 按训练 loss 选择；正式项目应增加独立验证集。
- `data/`、`pretrained/`、`runs/`、`.pt` 与 `.pth` 均被 `.gitignore` 排除。
