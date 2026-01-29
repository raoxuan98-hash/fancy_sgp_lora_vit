#!/bin/bash

# 并行运行exp2_alpha_constraint.py实验，使用不同的ViT架构在不同的GPU上
# GPU分配：
# GPU 0: vit-b-p16
# GPU 1: vit-b-p16-clip
# GPU 2: vit-b-p16-mocov3
# GPU 4: vit-b-p16-dino

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 设置PYTHONPATH
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# 创建日志目录
LOG_DIR="实验结果保存/分类器消融实验/logs"
mkdir -p $LOG_DIR

# 定义模型名称和对应的GPU
MODELS=("vit-b-p16" "vit-b-p16-clip" "vit-b-p16-mocov3" "vit-b-p16-dino")
GPUS=("0" "1" "2" "4")

# 启动并行进程
echo "开始并行运行exp2_alpha_constraint实验..."
echo "模型和GPU分配:"
for i in "${!MODELS[@]}"; do
    echo "  GPU ${GPUS[$i]}: ${MODELS[$i]}"
done
echo ""

# 启动所有实验进程
PIDS=()
for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    gpu=${GPUS[$i]}
    log_file="${LOG_DIR}/${model}_gpu${gpu}.log"
    
    echo "在GPU $gpu 上启动 $model 实验，日志保存到 $log_file"
    nohup python "$PROJECT_DIR/classifier_ablation/experiments/exp2_alpha_constraint_parallel.py" --model $model --gpu $gpu > $log_file 2>&1 &
    PIDS+=($!)
done

echo ""
echo "所有实验已启动，进程ID:"
for i in "${!PIDS[@]}"; do
    echo "  ${MODELS[$i]} (GPU ${GPUS[$i]}): PID ${PIDS[$i]}"
done

echo ""
echo "使用以下命令监控实验进度:"
echo "  tail -f $LOG_DIR/<model>_gpu<gpu>.log"
echo ""
echo "使用以下命令检查进程状态:"
echo "  ps -p ${PIDS[*]}"
echo ""
echo "等待所有实验完成..."
wait ${PIDS[*]}

echo ""
echo "所有实验完成!"