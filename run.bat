@echo off
python train.py ^
  --dataset parking ^
  --data data\training ^
  --input-size 512 ^
  --repeat 100 ^
  --batch-size 10 ^
  --epochs 30 ^
  --pretrained-path pretrained ^
  --output runs\parking_four_heads_pretrained_e30_fp32
