import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_manager1 import IncrementalDataManager

def check_dataset_size(dataset_name):
    """检查指定数据集的训练集数量"""
    print(f"\n=== 检查数据集: {dataset_name} ===")
    
    # 创建数据管理器，使用默认参数
    dm = IncrementalDataManager(
        dataset_name=dataset_name,
        initial_classes=10,  # 初始类别数
        increment_classes=10,  # 每次增加的类别数
        shuffle=False,
        seed=0
    )
    
    # 获取训练集
    train_dataset = dm.get_subset(task_id=0, source="train", cumulative=True)
    
    print(f"总类别数: {dm.num_classes}")
    print(f"训练集总样本数: {len(train_dataset)}")
    
    # 检查每个类别的样本数
    class_counts = {}
    for _, label, _ in train_dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"每个类别的样本数分布:")
    for class_id in sorted(class_counts.keys()):
        print(f"  类别 {class_id}: {class_counts[class_id]} 个样本")
    
    avg_samples_per_class = len(train_dataset) / len(class_counts)
    print(f"平均每类样本数: {avg_samples_per_class:.2f}")
    
    return len(train_dataset), dm.num_classes

if __name__ == "__main__":
    # 检查 cars196_224 数据集
    cars_train_size, cars_num_classes = check_dataset_size("cars196_224")
    
    # 检查 cub200_224 数据集
    cub_train_size, cub_num_classes = check_dataset_size("cub200_224")
    
    print("\n=== 总结 ===")
    print(f"CARS196_224: {cars_train_size} 个训练样本, {cars_num_classes} 个类别")
    print(f"CUB200_224: {cub_train_size} 个训练样本, {cub_num_classes} 个类别")