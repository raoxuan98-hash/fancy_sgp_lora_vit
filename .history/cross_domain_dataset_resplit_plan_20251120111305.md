# Cross-Domain数据集重新划分策略设计

## 1. 问题分析

### 1.1 当前问题
- **数据不平衡问题**：不同数据集样本数量差异巨大，测试集样本分布极不均匀
- **系统设计问题**：硬编码路径，数据集列表分散定义
- **可扩展性限制**：添加新数据集需要修改多个文件
- **样本分布问题**：没有对每个类别的样本数进行约束

### 1.2 需求约束
- 测试集每个类别的最大样本数约束到128
- 训练集每个类别的最大样本数约束到128
- 将划分后的数据集保存在新的目录下
- 保持与现有系统的兼容性

## 2. 设计目标

### 2.1 功能目标
- 实现自动化的数据集重新划分
- 支持每类样本数的精确控制
- 保持数据的随机性和可重现性
- 支持多种数据集格式（数组、路径）

### 2.2 技术目标
- 模块化设计，易于扩展
- 与现有CrossDomainDataManager集成
- 完善的错误处理和验证机制
- 高效的内存使用和文件管理

## 3. 重新划分算法设计

### 3.1 核心算法策略

**分层采样算法**：
1. 按类别统计原始样本分布
2. 对每类样本随机采样不超过128个
3. 保证训练/测试集的数据代表性
4. 使用固定种子确保可重现性

**约束处理机制**：
- 样本数少于128的类别：保留所有样本
- 样本数超过128的类别：随机采样128个
- 保持类别平衡：确保每个数据集的类别数不变

### 3.2 算法伪代码

```python
def resplit_dataset(dataset_info, max_per_class=128, seed=42):
    """
    数据集重新划分主算法
    
    Args:
        dataset_info: 原始数据集信息
        max_per_class: 每类最大样本数
        seed: 随机种子
    
    Returns:
        重新划分后的数据集信息
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 1. 分离训练和测试数据
    train_data = dataset_info['train_data']
    train_targets = dataset_info['train_targets']
    test_data = dataset_info['test_data']
    test_targets = dataset_info['test_targets']
    
    # 2. 对每个类别进行采样
    resplit_train_data = []
    resplit_train_targets = []
    resplit_test_data = []
    resplit_test_targets = []
    
    for class_id in range(dataset_info['num_classes']):
        # 训练集采样
        train_mask = train_targets == class_id
        train_indices = np.where(train_mask)[0]
        train_sample_size = min(len(train_indices), max_per_class)
        sampled_train_indices = np.random.choice(
            train_indices, size=train_sample_size, replace=False
        )
        
        # 测试集采样
        test_mask = test_targets == class_id
        test_indices = np.where(test_mask)[0]
        test_sample_size = min(len(test_indices), max_per_class)
        sampled_test_indices = np.random.choice(
            test_indices, size=test_sample_size, replace=False
        )
        
        # 收集采样结果
        resplit_train_data.extend(train_data[sampled_train_indices])
        resplit_train_targets.extend(train_targets[sampled_train_indices])
        resplit_test_data.extend(test_data[sampled_test_indices])
        resplit_test_targets.extend(test_targets[sampled_test_indices])
    
    return {
        'train_data': np.array(resplit_train_data),
        'train_targets': np.array(resplit_train_targets),
        'test_data': np.array(resplit_test_data),
        'test_targets': np.array(resplit_test_targets),
        'num_classes': dataset_info['num_classes']
    }
```

## 4. 新目录结构设计

### 4.1 推荐目录结构

```
resplit_datasets/
├── config/
│   ├── resplit_config.yaml          # 重新划分配置
│   └── dataset_mapping.json         # 数据集映射
├── datasets/
│   ├── cifar100_224/
│   │   ├── train/
│   │   │   ├── class_000/
│   │   │   │   ├── sample_00001.jpg
│   │   │   │   └── ...
│   │   │   ├── class_001/
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── class_000/
│   │   │   └── ...
│   │   └── metadata.json            # 数据集元数据
│   └── ...
├── scripts/
│   ├── resplit_main.py              # 主执行脚本
│   ├── validation.py                # 验证脚本
│   └── batch_resplit.py             # 批量处理脚本
├── utils/
│   ├── resplit_data_manager.py      # 重新划分数据管理器
│   ├── file_utils.py               # 文件操作工具
│   └── validation_utils.py         # 验证工具
└── reports/
    ├── resplit_summary.json         # 重新划分汇总
    └── validation_results.json      # 验证结果
```

### 4.2 元数据文件设计

**metadata.json**：
```json
{
    "dataset_name": "cifar100_224",
    "resplit_info": {
        "max_train_per_class": 128,
        "max_test_per_class": 128,
        "random_seed": 42,
        "resplit_date": "2025-11-20"
    },
    "original_stats": {
        "train_samples": 50000,
        "test_samples": 10000,
        "num_classes": 100
    },
    "resplit_stats": {
        "train_samples": 12800,
        "test_samples": 12800,
        "num_classes": 100,
        "effective_train_per_class": 128,
        "effective_test_per_class": 128
    },
    "class_mapping": {
        "class_000": "original_class_000",
        "class_001": "original_class_001"
    },
    "file_structure": {
        "train_dir": "train",
        "test_dir": "test",
        "class_format": "class_{:03d}"
    }
}
```

**resplit_config.yaml**：
```yaml
resplit:
  max_train_samples_per_class: 128
  max_test_samples_per_class: 128
  random_seed: 42
  skip_existing: true
  output_directory: "resplit_datasets"

datasets:
  - name: "cifar100_224"
    enabled: true
    custom_max_train: 128
    custom_max_test: 128
  - name: "cub200_224"
    enabled: true
  - name: "cars196_224"
    enabled: true
  - name: "imagenet-r"
    enabled: true
  - name: "food-101"
    enabled: true
  - name: "mnist"
    enabled: true
  - name: "oxford-iiit-pets"
    enabled: true
  - name: "caltech-101"
    enabled: true
  - name: "dtd"
    enabled: true
  - name: "fgvc-aircraft-2013b-variants102"
    enabled: true
  - name: "oxford-flower-102"
    enabled: true
  - name: "resisc45"
    enabled: true
```

## 5. 实现架构设计

### 5.1 核心类设计

**DatasetResplitter类**：
```python
class DatasetResplitter:
    """数据集重新划分主控制器"""
    
    def __init__(self, config_path="config/resplit_config.yaml"):
        self.config = self._load_config(config_path)
        self.output_dir = self.config['resplit']['output_directory']
        self.max_train = self.config['resplit']['max_train_samples_per_class']
        self.max_test = self.config['resplit']['max_test_samples_per_class']
        self.seed = self.config['resplit']['random_seed']
    
    def resplit_all_datasets(self):
        """重新划分所有启用的数据集"""
        enabled_datasets = [ds for ds in self.config['datasets'] if ds['enabled']]
        
        for dataset_config in enabled_datasets:
            try:
                self._resplit_single_dataset(dataset_config)
                print(f"✓ 成功重新划分数据集: {dataset_config['name']}")
            except Exception as e:
                print(f"✗ 重新划分数据集失败: {dataset_config['name']}, 错误: {e}")
        
        self._generate_summary_report()
    
    def _resplit_single_dataset(self, dataset_config):
        """重新划分单个数据集"""
        dataset_name = dataset_config['name']
        
        # 检查是否跳过现有结果
        if (self.config['resplit']['skip_existing'] and 
            self._check_existing_result(dataset_name)):
            print(f"跳过已存在的重新划分结果: {dataset_name}")
            return
        
        # 加载原始数据集
        original_data = self._load_original_dataset(dataset_name)
        
        # 应用自定义参数（如果存在）
        max_train = dataset_config.get('custom_max_train', self.max_train)
        max_test = dataset_config.get('custom_max_test', self.max_test)
        
        # 执行重新划分
        resplit_data = self._apply_resplit_algorithm(
            original_data, max_train, max_test, self.seed
        )
        
        # 保存结果
        self._save_resplit_dataset(dataset_name, resplit_data)
        
        # 验证结果
        self._validate_resplit_result(dataset_name, resplit_data)
```

**ResplitDataManager类**：
```python
class ResplitDataManager:
    """支持重新划分数据集的跨域数据管理器"""
    
    def __init__(self, dataset_names, resplit_dir="resplit_datasets", **kwargs):
        self.resplit_dir = resplit_dir
        self.dataset_names = dataset_names
        
        # 尝试加载重新划分的数据集
        self.datasets = self._load_resplit_datasets()
        
        # 如果重新划分数据集不存在，则回退到原始数据管理器
        if not self.datasets:
            print("警告：重新划分数据集不存在，使用原始数据管理器")
            self.original_manager = CrossDomainDataManagerCore(dataset_names, **kwargs)
            self.datasets = self.original_manager.datasets
    
    def get_subset(self, task, source="train", cumulative=False, **kwargs):
        """获取数据子集（支持重新划分的数据集）"""
        if self.datasets:
            return self._get_resplit_subset(task, source, cumulative, **kwargs)
        else:
            return self.original_manager.get_subset(task, source, cumulative, **kwargs)
```

### 5.2 辅助工具类

**FileManager类**：
```python
class FileManager:
    """文件操作管理器"""
    
    @staticmethod
    def create_directory_structure(dataset_name, output_dir):
        """创建数据集目录结构"""
        base_dir = os.path.join(output_dir, "datasets", dataset_name)
        
        dirs_to_create = [
            os.path.join(base_dir, "train"),
            os.path.join(base_dir, "test")
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        return base_dir
    
    @staticmethod
    def copy_dataset_files(resplit_data, dataset_name, base_dir):
        """复制或链接数据集文件"""
        # 适用于路径格式的数据集
        if isinstance(resplit_data['train_data'][0], str):
            FileManager._copy_path_based_data(resplit_data, base_dir)
        else:
            # 适用于数组格式的数据集（保存为二进制文件）
            FileManager._save_array_data(resplit_data, base_dir)
```

**ValidationEngine类**：
```python
class ValidationEngine:
    """数据集验证引擎"""
    
    @staticmethod
    def validate_resplit_result(dataset_name, resplit_data, max_per_class=128):
        """验证重新划分结果"""
        errors = []
        
        # 验证样本数约束
        train_class_counts = np.bincount(resplit_data['train_targets'])
        test_class_counts = np.bincount(resplit_data['test_targets'])
        
        for class_id in range(len(train_class_counts)):
            if train_class_counts[class_id] > max_per_class:
                errors.append(f"类别 {class_id} 训练样本数 {train_class_counts[class_id]} 超过限制 {max_per_class}")
        
        for class_id in range(len(test_class_counts)):
            if test_class_counts[class_id] > max_per_class:
                errors.append(f"类别 {class_id} 测试样本数 {test_class_counts[class_id]} 超过限制 {max_per_class}")
        
        # 验证数据一致性
        if len(resplit_data['train_data']) != len(resplit_data['train_targets']):
            errors.append("训练数据与标签数量不匹配")
        
        if len(resplit_data['test_data']) != len(resplit_data['test_targets']):
            errors.append("测试数据与标签数量不匹配")
        
        if errors:
            raise ValueError(f"数据集 {dataset_name} 验证失败:\\n" + "\\n".join(errors))
        
        return True
```

## 6. 集成方案

### 6.1 与现有CrossDomainDataManager集成

```python
# 扩展现有数据管理器
class EnhancedCrossDomainDataManager(CrossDomainDataManagerCore):
    """增强型跨域数据管理器，支持重新划分功能"""
    
    def __init__(self, dataset_names, use_resplit=False, resplit_config=None, **kwargs):
        super().__init__(dataset_names, **kwargs)
        
        self.use_resplit = use_resplit
        self.resplit_config = resplit_config
        
        if use_resplit and resplit_config:
            self.resplit_manager = DatasetResplitter(resplit_config)
            self._load_resplit_metadata()
    
    def get_subset(self, task, source="train", cumulative=False, mode=None, transform=None):
        """获取数据子集（增强版）"""
        if self.use_resplit and hasattr(self, 'resplit_datasets'):
            return self._get_resplit_subset(task, source, cumulative, mode, transform)
        else:
            return super().get_subset(task, source, cumulative, mode, transform)
```

### 6.2 配置管理集成

```python
# 统一配置管理
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file="config/resplit_config.yaml"):
        self.config_file = config_file
        self.config = self._load_yaml_config()
    
    def get_dataset_config(self, dataset_name):
        """获取特定数据集的配置"""
        for dataset in self.config['datasets']:
            if dataset['name'] == dataset_name:
                return dataset
        return None
    
    def update_dataset_config(self, dataset_name, new_config):
        """更新数据集配置"""
        for i, dataset in enumerate(self.config['datasets']):
            if dataset['name'] == dataset_name:
                self.config['datasets'][i].update(new_config)
                break
        
        self._save_yaml_config()
```

## 7. 质量保证机制

### 7.1 数据完整性检查

```python
class DataIntegrityChecker:
    """数据完整性检查器"""
    
    @staticmethod
    def check_file_integrity(dataset_dir):
        """检查文件完整性"""
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 检查目录结构
        required_dirs = ["train", "test"]
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_dir, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"必要目录不存在: {dir_path}")
        
        # 检查类别目录
        for class_id in range(metadata['resplit_stats']['num_classes']):
            class_dir = os.path.join(
                dataset_dir, "train", f"class_{class_id:03d}"
            )
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"类别目录不存在: {class_dir}")
        
        return True
```

### 7.2 统计验证

```python
class StatisticalValidator:
    """统计验证器"""
    
    @staticmethod
    def validate_sample_distribution(metadata):
        """验证样本分布"""
        resplit_stats = metadata['resplit_stats']
        config = metadata['resplit_info']
        
        # 检查每类样本数约束
        expected_max_train = config['max_train_per_class'] * resplit_stats['num_classes']
        expected_max_test = config['max_test_per_class'] * resplit_stats['num_classes']
        
        if resplit_stats['train_samples'] > expected_max_train:
            raise ValueError(f"训练样本数 {resplit_stats['train_samples']} 超过预期最大值 {expected_max_train}")
        
        if resplit_stats['test_samples'] > expected_max_test:
            raise ValueError(f"测试样本数 {resplit_stats['test_samples']} 超过预期最大值 {expected_max_test}")
        
        # 检查类别平衡性
        train_per_class = resplit_stats['train_samples'] / resplit_stats['num_classes']
        test_per_class = resplit_stats['test_samples'] / resplit_stats['num_classes']
        
        if abs(train_per_class - config['max_train_per_class']) > 1:
            warnings.warn(f"训练集每类样本数 {train_per_class:.1f} 与预期 {config['max_train_per_class']} 有差异")
        
        if abs(test_per_class - config['max_test_per_class']) > 1:
            warnings.warn(f"测试集每类样本数 {test_per_class:.1f} 与预期 {config['max_test_per_class']} 有差异")
        
        return True
```

### 7.3 错误处理和恢复

```python
class ResplitErrorHandler:
    """重新划分错误处理器"""
    
    ERROR_TYPES = {
        'FILE_NOT_FOUND': '文件未找到',
        'DATA_CORRUPTION': '数据损坏',
        'CONFIG_ERROR': '配置错误',
        'PERMISSION_ERROR': '权限错误',
        'DISK_SPACE_ERROR': '磁盘空间不足'
    }
    
    @classmethod
    def handle_error(cls, error, dataset_name, context=""):
        """处理错误"""
        error_type = cls._classify_error(error)
        error_message = cls.ERROR_TYPES.get(error_type, '未知错误')
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'error_details': str(error)
        }
        
        # 记录错误
        cls._log_error(log_entry)
        
        # 尝试恢复操作
