#!/bin/bash

# 并行运行exp2_alpha_constraint.py实验，使用不同的ViT架构在不同的GPU上
# GPU分配：
# GPU 0: vit-b-p16
# GPU 1: vit-b-p16-clip
# GPU 2: vit-b-p16-mocov3
# GPU 4: vit-b-p16-dino

# 设置工作目录
cd /home/raoxuan/projects/low_rank_rda

# 创建日志目录
LOG_DIR="实验结果保存/分类器消融实验/logs"
mkdir -p $LOG_DIR

# 定义模型名称和对应的GPU
MODELS=("vit-b-p16" "vit-b-p16-clip" "vit-b-p16-mocov3" "vit-b-p16-dino")
GPUS=("0" "1" "2" "4")

# 创建临时Python脚本来运行实验
cat > /tmp/run_exp2_model.py << 'EOF'
import os
import sys
import argparse

# 添加当前目录到Python路径
sys.path.insert(0, '/home/raoxuan/projects/low_rank_rda')

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

# 导入实验脚本
from classifier_ablation.experiments.exp2_alpha_constraint import *

if __name__ == '__main__':
    # 获取模型名称
    model_name = sys.argv[2]
    
    # 实验参数设置
    iterations = 0
    num_shots = 128
    base_output_dir = "实验结果保存/分类器消融实验"
    
    print(f"\n处理架构: {model_name}, iterations: {iterations}")
    print("="*60)
    
    # 创建输出目录
    model_output_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    
    # 获取和适配模型
    print("获取和适配Vision Transformer模型...")
    vit = get_vit(vit_name=model_name)
    adapt_loader = create_adapt_loader(train_subsets)
    vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)
    
    # 提取特征
    print("提取特征...")
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=iterations)
    
    train_dataset_ids = torch.tensor(train_dataset_ids)
    test_dataset_ids = torch.tensor(test_dataset_ids)
    
    # 构建高斯统计量
    print("\n构建高斯统计量...")
    train_stats = build_gaussian_statistics(train_features, train_labels)
    
    print("\n" + "="*60)
    print("运行约束条件下的α1-α2性能曲线对比实验")
    print("约束条件: α1 + α2 = 1.0")
    print("="*60)
    
    # 生成α1采样点
    alpha1_values = np.linspace(0, 1, 21)
    print(f"α1采样点: {alpha1_values}")
    print(f"模型: {model_name}")
    device = "cuda"
    
    qda_accuracies, sgd_linear_accuracies, sgd_nonlinear_accuracies, ncm_accuracies, lda_accuracies = evaluate_classifiers_under_constraint(
        alpha1_values, train_stats, test_features, test_labels, test_dataset_ids,
        alpha3=0.5, device=device)
    
    save_dir = model_output_dir

    plot_path = os.path.join(save_dir, f"{model_name}_constraint_performance.png")
    best_alpha1_qda, best_alpha1_sgd_linear, best_alpha1_sgd_nonlinear, best_acc_qda, best_acc_sgd_linear, best_acc_sgd_nonlinear, best_acc_ncm, best_acc_lda = plot_alpha_constraint_performance(
        alpha1_values, qda_accuracies, sgd_linear_accuracies, sgd_nonlinear_accuracies, ncm_accuracies, lda_accuracies, plot_path
    )
        
    # 保存实验结果
    save_constraint_results(alpha1_values, qda_accuracies, sgd_linear_accuracies, sgd_nonlinear_accuracies, ncm_accuracies, lda_accuracies,
                        model_name, save_dir)
        
    print("\n" + "="*50)
    print("实验总结")
    print("="*50)
    print(f"QDA最佳性能: α1={best_alpha1_qda:.3f}, 准确度={best_acc_qda:.2f}%")
    print(f"线性SGD最佳性能: α1={best_alpha1_sgd_linear:.3f}, 准确度={best_acc_sgd_linear:.2f}%")
    print(f"非线性SGD最佳性能: α1={best_alpha1_sgd_nonlinear:.3f}, 准确度={best_acc_sgd_nonlinear:.2f}%")
    print(f"NCM性能: 准确度={best_acc_ncm:.2f}%")
    print(f"LDA性能: 准确度={best_acc_lda:.2f}%")
    
    # 计算平均性能
    avg_qda = np.mean(qda_accuracies) * 100
    avg_sgd_linear = np.mean(sgd_linear_accuracies) * 100
    avg_sgd_nonlinear = np.mean(sgd_nonlinear_accuracies) * 100
    avg_ncm = np.mean(ncm_accuracies) * 100
    avg_lda = np.mean(lda_accuracies) * 100
    print(f"QDA平均性能: {avg_qda:.2f}%")
    print(f"线性SGD平均性能: {avg_sgd_linear:.2f}%")
    print(f"非线性SGD平均性能: {avg_sgd_nonlinear:.2f}%")
    print(f"NCM平均性能: {avg_ncm:.2f}%")
    print(f"LDA平均性能: {avg_lda:.2f}%")
    
    # 找到性能差异最大的点
    acc_diff_qda_linear = np.abs(np.array(qda_accuracies) - np.array(sgd_linear_accuracies))
    acc_diff_qda_nonlinear = np.abs(np.array(qda_accuracies) - np.array(sgd_nonlinear_accuracies))
    acc_diff_linear_nonlinear = np.abs(np.array(sgd_linear_accuracies) - np.array(sgd_nonlinear_accuracies))
    
    max_diff_idx_qda_linear = np.argmax(acc_diff_qda_linear)
    max_diff_idx_qda_nonlinear = np.argmax(acc_diff_qda_nonlinear)
    max_diff_idx_linear_nonlinear = np.argmax(acc_diff_linear_nonlinear)
    
    print(f"QDA与线性SGD最大性能差异: {acc_diff_qda_linear[max_diff_idx_qda_linear]*100:.2f}% at α1={alpha1_values[max_diff_idx_qda_linear]:.3f}")
    print(f"QDA与非线性SGD最大性能差异: {acc_diff_qda_nonlinear[max_diff_idx_qda_nonlinear]*100:.2f}% at α1={alpha1_values[max_diff_idx_qda_nonlinear]:.3f}")
    print(f"线性SGD与非线性SGD最大性能差异: {acc_diff_linear_nonlinear[max_diff_idx_linear_nonlinear]*100:.2f}% at α1={alpha1_values[max_diff_idx_linear_nonlinear]:.3f}")
    
    print(f"\n{model_name} 实验完成!")
EOF

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
    nohup python /tmp/run_exp2_model.py $gpu $model > $log_file 2>&1 &
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