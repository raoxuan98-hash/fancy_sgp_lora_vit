import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from pathlib import Path

from utils.data1 import get_dataset, SimpleDataset, pil_loader
from utils.cross_domain_data_manager import CrossDomainDataManagerCore, CrossDomainSimpleDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BalancedCrossDomainDataManagerCore(CrossDomainDataManagerCore):
    """
    平衡后的Cross-domain class-incremental data manager.
    继承自CrossDomainDataManagerCore，支持加载平衡后的数据集
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        balanced_datasets_root: str = "balanced_datasets",
        shuffle: bool = False,
        seed: int = 0,
        num_shots: int = 0,
        log_level: int = logging.INFO,
        use_balanced_datasets: bool = True
    ) -> None:
        
        self.balanced_datasets_root = balanced_datasets_root
        self.use_balanced_datasets = use_balanced_datasets
        
        if use_balanced_datasets:
            # 使用平衡后的数据集
            self._init_balanced_datasets(dataset_names, shuffle, seed, num_shots, log_level)
        else:
            # 使用原始数据集
            super().__init__(dataset_names, shuffle, seed, num_shots, log_level)
    
    def _init_balanced_datasets(self, dataset_names: List[str], shuffle: bool, seed: int, num_shots: int, log_level: int):
        """初始化平衡后的数据集"""
        
        logging.basicConfig(level=log_level)
        self.dataset_names = dataset_names
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_shots = int(num_shots)
        
        # Load all balanced datasets
        self.datasets = []
        self.global_label_offset = []
        self.total_classes = 0
        self.global_class_names = []
        
        for i, dataset_name in enumerate(dataset_names):
            logging.info(f"[BCDM] Loading balanced dataset {i+1}/{len(dataset_names)}: {dataset_name}")
            dataset_info = self._load_balanced_dataset(dataset_name)
            
            if dataset_info is None:
                logging.warning(f"[BCDM] Failed to load balanced dataset {dataset_name}, falling back to original")
                # 回退到原始数据集
                original_dataset = get_dataset(dataset_name)
                dataset_info = {
                    'name': dataset_name,
                    'train_data': np.asarray(getattr(original_dataset, 'train_data', [])),
                    'test_data': np.asarray(getattr(original_dataset, 'test_data', [])),
                    'train_targets': np.asarray(getattr(original_dataset, 'train_targets', []), dtype=np.int64),
                    'test_targets': np.asarray(getattr(original_dataset, 'test_targets', []), dtype=np.int64),
                    'num_classes': len(getattr(original_dataset, 'class_names', [])),
                    'use_path': bool(getattr(original_dataset, "use_path", False)),
                    'class_names': list(getattr(original_dataset, "class_names", []) or []),
                    'templates': list(getattr(original_dataset, "templates", []) or [])
                }
            
            # Apply few-shot sampling if num_shots > 0
            if self.num_shots > 0:
                logging.info(f"[BCDM] Applying few-shot sampling: {self.num_shots} shots per class")
                dataset_info = self._apply_few_shot_sampling(dataset_info, self.seed + i)
            
            # Apply global label offset
            offset = self.total_classes
            
            # 应用全局标签偏移到原始标签
            dataset_info['train_targets'] = dataset_info['train_targets'] + offset
            dataset_info['test_targets'] = dataset_info['test_targets'] + offset
            
            self.datasets.append(dataset_info)
            self.global_label_offset.append(offset)
            self.total_classes += dataset_info['num_classes']
            self.global_class_names.extend(dataset_info['class_names'])
        
        logging.info(f"[BCDM] Total datasets: {len(self.datasets)}")
        logging.info(f"[BCDM] Total classes: {self.total_classes}")
        logging.info(f"[BCDM] Total tasks: {len(self.datasets)}")
    
    def _load_balanced_dataset(self, dataset_name: str) -> Optional[Dict]:
        """加载平衡后的数据集"""
        dataset_path = Path(self.balanced_datasets_root) / dataset_name
        
        if not dataset_path.exists():
            logging.warning(f"[BCDM] Balanced dataset not found: {dataset_path}")
            return None
        
        try:
            # 读取标签文件
            label_file = dataset_path / "label.txt"
            if not label_file.exists():
                logging.error(f"[BCDM] Label file not found: {label_file}")
                return None
            
            with open(label_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            
            # 读取训练和测试数据
            train_data, train_targets = self._load_data_from_directory(dataset_path / "train")
            test_data, test_targets = self._load_data_from_directory(dataset_path / "test")
            
            # 判断是否使用路径
            use_path = all(isinstance(x, str) for x in train_data[:10]) if len(train_data) > 0 else False
            
            # 获取模板（从原始数据集）
            try:
                original_dataset = get_dataset(dataset_name)
                templates = getattr(original_dataset, 'templates', [])
            except:
                templates = []
            
            dataset_info = {
                'name': dataset_name,
                'train_data': np.array(train_data),
                'test_data': np.array(test_data),
                'train_targets': np.array(train_targets, dtype=np.int64),
                'test_targets': np.array(test_targets, dtype=np.int64),
                'num_classes': len(class_names),
                'use_path': use_path,
                'class_names': class_names,
                'templates': templates
            }
            
            logging.info(f"[BCDM] Loaded balanced dataset {dataset_name}: "
                        f"{len(train_data)} train samples, {len(test_data)} test samples, "
                        f"{len(class_names)} classes")
            
            return dataset_info
            
        except Exception as e:
            logging.error(f"[BCDM] Error loading balanced dataset {dataset_name}: {str(e)}")
            return None
    
    def _load_data_from_directory(self, dir_path: Path) -> Tuple[List, List[int]]:
        """从目录结构加载数据"""
        data = []
        targets = []
        
        if not dir_path.exists():
            return data, targets
        
        # 遍历所有类别目录
        for class_dir in sorted(dir_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            try:
                class_id = int(class_dir.name)
            except ValueError:
                continue
            
            # 遍历类别中的所有文件
            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    data.append(str(file_path))
                    targets.append(class_id)
        
        return data, targets
    
    def get_balanced_statistics(self) -> Dict[str, Dict]:
        """获取平衡后数据集的统计信息"""
        stats = {}
        
        for i, dataset in enumerate(self.datasets):
            dataset_name = dataset['name']
            
            # 统计每个类别的样本数
            train_counts = {}
            test_counts = {}
            
            for label in dataset['train_targets']:
                train_counts[label] = train_counts.get(label, 0) + 1
            
            for label in dataset['test_targets']:
                test_counts[label] = test_counts.get(label, 0) + 1
            
            # 计算统计指标
            train_values = list(train_counts.values())
            test_values = list(test_counts.values())
            
            stats[dataset_name] = {
                'num_classes': dataset['num_classes'],
                'total_train_samples': len(dataset['train_data']),
                'total_test_samples': len(dataset['test_data']),
                'train_per_class': {
                    'min': min(train_values) if train_values else 0,
                    'max': max(train_values) if train_values else 0,
                    'mean': np.mean(train_values) if train_values else 0,
                    'std': np.std(train_values) if train_values else 0
                },
                'test_per_class': {
                    'min': min(test_values) if test_values else 0,
                    'max': max(test_values) if test_values else 0,
                    'mean': np.mean(test_values) if test_values else 0,
                    'std': np.std(test_values) if test_values else 0
                }
            }
        
        return stats
    
    def compare_with_original(self) -> Dict[str, Dict]:
        """与原始数据集进行比较"""
        comparison = {}
        
        for dataset_name in self.dataset_names:
            try:
                # 加载原始数据集
                original_dataset = get_dataset(dataset_name)
                
                # 找到对应的平衡数据集
                balanced_dataset = None
                for dataset in self.datasets:
                    if dataset['name'] == dataset_name:
                        balanced_dataset = dataset
                        break
                
                if balanced_dataset is None:
                    continue
                
                # 获取原始数据
                original_train_targets = getattr(original_dataset, 'train_targets', [])
                original_test_targets = getattr(original_dataset, 'test_targets', [])
                original_train_data = getattr(original_dataset, 'train_data', [])
                original_test_data = getattr(original_dataset, 'test_data', [])
                
                # 计算原始统计
                original_train_counts = {}
                original_test_counts = {}
                
                for label in original_train_targets:
                    original_train_counts[int(label)] = original_train_counts.get(int(label), 0) + 1
                
                for label in original_test_targets:
                    original_test_counts[int(label)] = original_test_counts.get(int(label), 0) + 1
                
                # 计算平衡后统计
                balanced_train_counts = {}
                balanced_test_counts = {}
                
                for label in balanced_dataset['train_targets']:
                    balanced_train_counts[int(label)] = balanced_train_counts.get(int(label), 0) + 1
                
                for label in balanced_dataset['test_targets']:
                    balanced_test_counts[int(label)] = balanced_test_counts.get(int(label), 0) + 1
                
                comparison[dataset_name] = {
                    'original': {
                        'total_train_samples': len(original_train_data),
                        'total_test_samples': len(original_test_data),
                        'train_per_class_stats': {
                            'min': min(original_train_counts.values()) if original_train_counts else 0,
                            'max': max(original_train_counts.values()) if original_train_counts else 0,
                            'mean': np.mean(list(original_train_counts.values())) if original_train_counts else 0,
                            'std': np.std(list(original_train_counts.values())) if original_train_counts else 0
                        },
                        'test_per_class_stats': {
                            'min': min(original_test_counts.values()) if original_test_counts else 0,
                            'max': max(original_test_counts.values()) if original_test_counts else 0,
                            'mean': np.mean(list(original_test_counts.values())) if original_test_counts else 0,
                            'std': np.std(list(original_test_counts.values())) if original_test_counts else 0
                        }
                    },
                    'balanced': {
                        'total_train_samples': len(balanced_dataset['train_data']),
                        'total_test_samples': len(balanced_dataset['test_data']),
                        'train_per_class_stats': {
                            'min': min(balanced_train_counts.values()) if balanced_train_counts else 0,
                            'max': max(balanced_train_counts.values()) if balanced_train_counts else 0,
                            'mean': np.mean(list(balanced_train_counts.values())) if balanced_train_counts else 0,
                            'std': np.std(list(balanced_train_counts.values())) if balanced_train_counts else 0
                        },
                        'test_per_class_stats': {
                            'min': min(balanced_test_counts.values()) if balanced_test_counts else 0,
                            'max': max(balanced_test_counts.values()) if balanced_test_counts else 0,
                            'mean': np.mean(list(balanced_test_counts.values())) if balanced_test_counts else 0,
                            'std': np.std(list(balanced_test_counts.values())) if balanced_test_counts else 0
                        }
                    }
                }
                
            except Exception as e:
                logging.error(f"[BCDM] Error comparing dataset {dataset_name}: {str(e)}")
        
        return comparison


def create_balanced_data_manager(dataset_names: List[str], 
                               balanced_datasets_root: str = "balanced_datasets",
                               **kwargs) -> BalancedCrossDomainDataManagerCore:
    """
    创建平衡后的数据管理器
    
    Args:
        dataset_names: 数据集名称列表
        balanced_datasets_root: 平衡数据集根目录
        **kwargs: 其他参数传递给BalancedCrossDomainDataManagerCore
        
    Returns:
        BalancedCrossDomainDataManagerCore实例
    """
    return BalancedCrossDomainDataManagerCore(
        dataset_names=dataset_names,
        balanced_datasets_root=balanced_datasets_root,
        **kwargs
    )


def main():
    """测试函数"""
    # 默认数据集列表
    default_datasets = [
        'cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 'caltech-101', 
        'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 
        'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'
    ]
    
    # 创建平衡数据管理器
    manager = create_balanced_data_manager(
        dataset_names=default_datasets[:3],  # 只测试前3个数据集
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True
    )
    
    # 获取统计信息
    stats = manager.get_balanced_statistics()
    print("平衡后数据集统计信息:")
    for dataset_name, stat in stats.items():
        print(f"{dataset_name}:")
        print(f"  训练样本: {stat['total_train_samples']}, 测试样本: {stat['total_test_samples']}")
        print(f"  训练每类: min={stat['train_per_class']['min']}, max={stat['train_per_class']['max']}")
        print(f"  测试每类: min={stat['test_per_class']['min']}, max={stat['test_per_class']['max']}")
    
    # 与原始数据集比较
    comparison = manager.compare_with_original()
    print("\n与原始数据集比较:")
    for dataset_name, comp in comparison.items():
        print(f"{dataset_name}:")
        print(f"  原始测试样本: {comp['original']['total_test_samples']}")
        print(f"  平衡测试样本: {comp['balanced']['total_test_samples']}")


if __name__ == "__main__":
    main()