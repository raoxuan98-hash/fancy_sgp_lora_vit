import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试直接从数据集类中获取信息，而不实际加载数据
from utils.data1 import DATASET_NAME_TO_CLASS

def check_dataset_info(dataset_name):
    """检查指定数据集的基本信息，不实际加载数据"""
    print(f"\n=== 检查数据集: {dataset_name} ===")
    
    if dataset_name not in DATASET_NAME_TO_CLASS:
        print(f"错误: 数据集 {dataset_name} 不在支持列表中")
        return None, None
    
    # 获取数据集类
    dataset_class = DATASET_NAME_TO_CLASS[dataset_name]
    print(f"数据集类: {dataset_class}")
    
    # 尝试获取数据集的基本信息
    try:
        # 创建一个临时实例来获取信息
        temp_instance = dataset_class() if callable(dataset_class) else dataset_class
        
        # 获取类别信息
        if hasattr(temp_instance, 'class_names') and temp_instance.class_names:
            num_classes = len(temp_instance.class_names)
            print(f"类别数: {num_classes}")
        elif hasattr(temp_instance, 'num_classes'):
            num_classes = temp_instance.num_classes
            print(f"类别数: {num_classes}")
        else:
            print("无法获取类别数")
            num_classes = None
        
        # 尝试获取训练集大小
        if hasattr(temp_instance, 'train_data') and temp_instance.train_data is not None:
            train_size = len(temp_instance.train_data)
            print(f"训练集大小: {train_size}")
            
            if num_classes:
                avg_samples_per_class = train_size / num_classes
                print(f"平均每类样本数: {avg_samples_per_class:.2f}")
        else:
            print("无法获取训练集大小")
            train_size = None
            
        return train_size, num_classes
        
    except Exception as e:
        print(f"获取数据集信息时出错: {str(e)}")
        return None, None

if __name__ == "__main__":
    # 检查 cars196_224 数据集
    cars_train_size, cars_num_classes = check_dataset_info("cars196_224")
    
    # 检查 cub200_224 数据集
    cub_train_size, cub_num_classes = check_dataset_info("cub200_224")
    
    print("\n=== 总结 ===")
    if cars_train_size is not None and cars_num_classes is not None:
        print(f"CARS196_224: {cars_train_size} 个训练样本, {cars_num_classes} 个类别")
    else:
        print("CARS196_224: 无法获取信息")
        
    if cub_train_size is not None and cub_num_classes is not None:
        print(f"CUB200_224: {cub_train_size} 个训练样本, {cub_num_classes} 个类别")
    else:
        print("CUB200_224: 无法获取信息")