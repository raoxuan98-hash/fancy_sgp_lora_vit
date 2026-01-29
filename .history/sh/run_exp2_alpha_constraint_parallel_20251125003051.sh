#!/bin/bash

# 设置GPU和模型架构
GPUS=(0 1 2 4)
MODELS=("vit-b-p16" "vit-b-p16-clip" "vit-b-p16-dino" "vit-b-p16-mocov3")

# 设置实验参数
ITERATIONS=0
NUM_SHOTS=128
BASE_OUTPUT_DIR="实验结果保存/分类器消融实验"

# 创建日志目录
LOG_DIR="logs/exp2_alpha_constraint"
mkdir -p $LOG_DIR

# 打印实验信息
echo "=========================================="
echo "运行Alpha约束实验 - 并行执行"
echo "=========================================="
echo "GPU列表: ${GPUS[*]}"
echo "模型列表: ${MODELS[*]}"
echo "迭代次数: $ITERATIONS"
echo "每类样本数: $NUM_SHOTS"
echo "输出目录: $BASE_OUTPUT_DIR"
echo "=========================================="

# 并行运行实验
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    MODEL=${MODELS[$i]}
    
    echo "在GPU $GPU上启动模型 $MODEL 的实验..."
    
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