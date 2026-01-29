
#!/bin/bash

# 修复版的性能曲面等高线图混合并行训练脚本
# 不同GPU之间并行运行，同个GPU上的不同秩实验串行运行

# 设置日志目录
LOG_DIR="./logs/exp1_performance_surface"
mkdir -p $LOG_DIR

# 按GPU分组实验配置 - 使用更合理的GPU分配
declare -A GPU_EXPERIMENTS
# vit-b-p16 使用 GPU 0
GPU_EXPERIMENTS[0]="vit-b-p16:1:0:vit-b-p16-rank1 vit-b-p16:8:0:vit-b-p16-rank8 vit-b-p16:32:0:vit-b-p16-rank32"
# vit-b-p16-clip 使用 GPU 1  
GPU_EXPERIMENTS[1]="vit-b-p16-clip:1:1:vit-b-p16-clip-rank1 vit-b-p16-clip:8:1:vit-b-p16-clip-rank8 vit-b-p16-clip:32:1:vit-b-p16-clip-rank32"
# vit-b-p16-dino 使用 GPU 2
GPU_EXPERIMENTS[2]="vit-b-p16-dino:1:2:vit-b-p16-dino-rank1 vit-b-p16-dino:8:2:vit-b-p16-dino-rank8 vit-b-p16-dino:32:2:vit-b-p16-dino-rank32"
# vit-b-p16-mocov3 使用 GPU 3 (修改为GPU 3而不是GPU 4)
GPU_EXPERIMENTS[3]="vit-b-p16-mocov3:1:3:vit-b-p16-mocov3-rank1 vit-b-p16-mocov3:8:3:vit-b-p16-mocov3-rank8 vit-b-p16-mocov3:32:3:vit-b-p16-mocov3-rank32"

# 获取所有GPU ID
GPUS=(${!GPU_EXPERIMENTS[@]})

# 创建后台进程数组
declare -a PIDS=()
declare -a TEMP_SCRIPTS=()

echo "启动修复版性能曲面等高线图实验（混合并行：不同GPU并行，同GPU串行）..."
echo "使用GPU: ${GPUS[@]}"
echo "架构分配:"
for gpu_id in "${GPUS[@]}"; do
    echo "  GPU $gpu_id: ${GPU_EXPERIMENTS[$gpu_id]}"
done
echo "=================================="

# 记录开始时间
start_time=$(date)
echo "实验开始时间: $start_time"
echo ""

# 清理函数
cleanup() {
    echo "清理临时文件和进程..."
    for temp_script in "${TEMP_SCRIPTS[@]}"; do
        if [ -f "$temp_script" ]; then
            rm -f "$temp_script"
        fi
    done
    
    # 终止所有子进程
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "终止进程 $pid"
            kill -TERM "$pid" 2>/dev/null
            sleep 2
            kill -KILL "$pid" 2>/dev/null
        fi
    done
}

# 设置信号处理
trap cleanup EXIT INT TERM

# 为每个GPU启动串行实验进程
for gpu_id in "${GPUS[@]}"; do
    experiments="${GPU_EXPERIMENTS[$gpu_id]}"
    
    echo "启动GPU $gpu_id 上的串行实验进程..."
    
    # 创建临时脚本文件来运行串行实验
    temp_script="/tmp/gpu_${gpu_id}_serial_experiments.sh"
    TEMP_SCRIPTS+=("$temp_script")
    
    cat > "$temp_script" << EOF
#!/bin/bash

# GPU $gpu_id 上的串行实验脚本
experiments=($experiments)

echo "GPU $gpu_id 开始运行 \${#experiments[@]} 个实验..."

# 设置GPU环境变量
export CUDA_VISIBLE_DEVICES=$gpu_id
echo "CUDA_VISIBLE_DEVICES 设置为: \$CUDA_VISIBLE_DEVICES"

# 清理GPU内存
echo "清理GPU $gpu_id 内存..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --gpu-reset-remaining-utilization-thresholds -i $gpu_id 2>/dev/null || true
fi

# 清理PyTorch缓存
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_device($gpu_id)
    print(f'GPU \$CUDA_VISIBLE_DEVICES memory cleared')
else:
    print('CUDA not available')
" 2>/dev/null || echo "PyTorch缓存清理跳过"

for exp in "\${experiments[@]}"; do
    # 解析实验配置
    IFS=':' read -ra CONFIG <<< "\$exp"
    MODEL_NAME="\${CONFIG[0]}"
    RANK="\${CONFIG[1]}"
    GPU_ID="\${CONFIG[2]}"
    EXP_NAME="\${CONFIG[3]}"
    
    echo "=================================="
    echo "GPU \$GPU_ID 启动实验: \$EXP_NAME"
    echo "  模型: \$MODEL_NAME"
    echo "  Rank: \$RANK"
    echo "  开始时间: \$(date)"
    echo "  CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
    echo "=================================="
    
    # 运行实验 - 确保GPU环境变量正确传递
    LOG_FILE="$LOG_DIR/\${EXP_NAME}.log"
    
    # 设置内存管理环境变量
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # 启动实验并正确传递GPU参数
    CUDA_VISIBLE_DEVICES=\$GPU_ID python classifier_ablation/experiments/exp1_performance_surface.py \\
        --model "\$MODEL_NAME" \\
        --rank "\$RANK" \\
        --gpu "\$GPU_ID" \\
        --iterations 0 \\
        --num_shots 128 \\
        > "\$LOG_FILE" 2>&1
    
    # 检查实验是否成功完成
    exit_code=\$?
    if [ \$exit_code -eq 0 ]; then
        echo "✅ GPU \$GPU_ID 实验 \$EXP_NAME 成功完成"
        # 提取最佳性能
        if grep -q "最佳参数:" "\$LOG_FILE"; then
            echo "最佳性能: \$(grep "最佳参数:" "\$LOG_FILE" | tail -1)"
        fi
        if grep -q "实验完成" "\$LOG_FILE"; then
            echo "实验状态: 完成"
        fi
    else
        echo "❌ GPU \$GPU_ID 实验 \$EXP_NAME 失败，退出码: \$exit_code"
        echo "查看日志文件: \$LOG_FILE"
        # 检查GPU内存问题
        if grep -q "CUDA out of memory" "\$LOG_FILE"; then
            echo "检测到GPU内存不足，清理内存后重试..."
            python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU memory cleared')
" 2>/dev/null
            sleep 5
        fi
    fi
    
    echo "  结束时间: \$(date)"
    echo "  GPU内存使用情况:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $gpu_id
    fi
    echo ""
    
    # 实验间清理
    echo "清理GPU内存..."
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU memory cleared between experiments')
" 2>/dev/null || true
    
    sleep 2  # 实验间隔
done

echo "GPU $gpu_id 上的所有实验完成!"
EOF

    # 使临时脚本可执行
    chmod +x "$temp_script"
    
    # 在后台启动GPU的串行实验进程
    nohup "$temp_script" > "$LOG_DIR/gpu_${gpu_id}_serial.log" 2>&1 &
    
    # 记录进程ID
    PID=$!
    PIDS+=($PID)
    
    echo "GPU $gpu_id 串行实验进程已启动，进程ID: $PID"
    echo ""
    
    # 给系统一点时间启动进程
    sleep 3
done

echo "所有GPU串行实验进程已启动..."
echo "进程ID列表: ${PIDS[@]}"
echo "=================================="

# 监控进程状态
monitor_experiments() {
    local active_pids=()
    local finished_count=0
    local total_count=${#PIDS[@]}
    
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            active_pids+=($pid)
        else
            echo "GPU进程 $pid 已完成"
            finished_count=$((finished_count + 1))
        fi
    done
    
    if [ ${#active_pids[@]} -eq 0 ]; then
        echo "所有GPU实验进程已完成! ($finished_count/$total_count)"
        return 1
    else
        echo "运行中的GPU进程: ${#active_pids[@]}/$total_count, 已完成: $finished_count"
        
        # 显示每个活跃进程的GPU使用情况
        for pid in "${active_pids[@]}"; do
            gpu_id=$(ps -o pid,args -p $pid | tail -1 | awk '{print $NF}' | grep -o '[0-9]' || echo "unknown")
            if [ "$gpu_id" != "unknown" ] && command -v nvidia-smi >/dev/null 2>&1; then
                memory_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $gpu_id 2>/dev/null || echo "N/A")
                echo "  GPU $gpu_id (PID $pid): $memory_info"
            fi
        done
        
        PIDS=("${active_pids[@]}")
        return 0
    fi
}

# 定期检查进程状态
check_interval=60  # 60秒检查一次
max_wait_time=3600  # 最大等待时间1小时
elapsed_time=0

while true; do
    if ! monitor_experiments; then
        break
    fi
    
    if [ $elapsed_time -ge $max_wait_time ]; then
        echo "达到最大等待时间，强制终止所有进程..."
        break
    fi
    
    echo "等待 ${check_interval} 秒后再次检查... (已运行 ${elapsed_time}s)"
    sleep $check_interval
    elapsed_time=$((elapsed_time + check_interval))
done

# 记录结束时间
end_time=$(date)
echo "=================================="
echo "实验监控完成!"
echo "实验开始时间: $start_time"
echo "实验结束时间: $end_time"
echo "总运行时间: $((elapsed_time)) 秒"
echo "=================================="

# 汇总结果
echo "实验结果汇总:"
echo "=================================="
for gpu_id in "${GPUS[@]}"; do
    experiments="${GPU_EXPERIMENTS[$gpu_id]}"
    echo "--- GPU $gpu_id 结果 ---"
    for exp in $experiments; do
        IFS=':' read -ra CONFIG <<< "$exp"
        EXP_NAME="${CONFIG[3]}"
        LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
        
        if [ -f "$LOG_FILE" ]; then
            echo "  $EXP_NAME:"
            # 检查是否包含实验完成的标记
            if grep -q "实验完成" "$LOG_FILE"; then
                echo "    状态: ✅ 完成"
                # 提取最佳性能
                if grep -q "最佳参数:" "$LOG_FILE"; then
                    grep "最佳参数:" "$LOG_FILE" | tail -1 | sed 's/^/    /'
                fi
            elif grep -q "构建QDA分类器失败" "$LOG_FILE"; then
                echo "    状态: ❌ 失败 (QDA构建错误)"
            elif grep -q "CUDA out of memory" "$LOG_FILE"; then
                echo "    状态: ❌ 失败 (GPU内存不足)"
            else
                echo "    状态: ❓ 未完成"
            fi
            
            # 显示实验耗时
            if grep -q "开始时间:" "$LOG_FILE" && grep -q "结束时间:" "$LOG_FILE"; then
                start_line=$(grep -n "开始时间:" "$LOG_FILE" | tail -1 | cut -d: -f1)
                end_line=$(grep -n "结束时间:" "$LOG_FILE" | tail -1 | cut -d: -f1)
                if [ "$start_line" != "" ] && [ "$end_line" != "" ]; then
                    echo "    耗时: $(($end_line - $start_line)) 行"
                fi
            fi
        else
            echo "  $EXP_NAME: 日志文件不存在"
        fi
    done
    echo ""
done

