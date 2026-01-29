# exp2_alpha_constraint 并行实验脚本

本目录包含用于并行运行exp2_alpha_constraint实验的脚本，可以在多个GPU上同时运行不同的ViT架构。

## 文件说明

- `run_exp2_alpha_constraint_parallel.sh`: 主要的并行实验脚本，在4个GPU上同时运行4种不同的ViT架构
- `test_exp2_parallel.sh`: 测试脚本，只运行一个模型用于验证脚本是否正常工作
- `classifier_ablation/experiments/exp2_alpha_constraint_parallel.py`: 支持命令行参数的实验脚本

## GPU分配

- GPU 0: vit-b-p16
- GPU 1: vit-b-p16-clip
- GPU 2: vit-b-p16-mocov3
- GPU 4: vit-b-p16-dino

## 使用方法

### 1. 测试运行（推荐先运行测试）

```bash
cd /home/raoxuan/projects/low_rank_rda
./sh/test_exp2_parallel.sh
```

这将只运行vit-b-p16模型在GPU 0上，用于验证脚本是否正常工作。

### 2. 完整并行运行

```bash
cd /home/raoxuan/projects/low_rank_rda
./sh/run_exp2_alpha_constraint_parallel.sh
```

这将在4个GPU上同时运行4种不同的ViT架构。

### 3. 单独运行特定模型

```bash
cd /home/raoxuan/projects/low_rank_rda
python classifier_ablation/experiments/exp2_alpha_constraint_parallel.py --model vit-b-p16 --gpu 0
```

可用参数：
- `--model`: 模型名称 (vit-b-p16, vit-b-p16-clip, vit-b-p16-mocov3, vit-b-p16-dino)
- `--gpu`: GPU编号
- `--iterations`: 迭代次数 (默认为0)
- `--num_shots`: 样本数量 (默认为128)

## 监控实验进度

### 查看日志文件

```bash
# 查看特定模型的日志
tail -f 实验结果保存/分类器消融实验/logs/vit-b-p16_gpu0.log

# 查看所有日志文件
ls -la 实验结果保存/分类器消融实验/logs/
```

### 检查进程状态

```bash
# 如果知道进程ID
ps -p <PID1> <PID2> <PID3> <PID4>

# 查看所有Python进程
ps aux | grep exp2_alpha_constraint
```

## 结果输出

实验结果将保存在以下目录：

```
实验结果保存/分类器消融实验/
├── vit-b-p16_iter0/
│   ├── vit-b-p16_constraint_performance.png
│   ├── vit-b-p16_constraint_results.npz
│   └── vit-b-p16_constraint_results.csv
├── vit-b-p16-clip_iter0/
├── vit-b-p16-mocov3_iter0/
└── vit-b-p16-dino_iter0/
```

## 注意事项

1. 确保所有指定的GPU都可用且没有被其他进程占用
2. 实验可能需要较长时间完成，建议使用nohup或screen运行
3. 如果某个实验失败，可以单独运行该模型的实验
4. 日志文件可以帮助诊断问题

## 故障排除

### 检查GPU可用性

```bash
nvidia-smi
```

### 检查CUDA环境

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 检查数据路径

确保数据路径正确，特别是`balanced_datasets`目录是否存在且包含所需数据。