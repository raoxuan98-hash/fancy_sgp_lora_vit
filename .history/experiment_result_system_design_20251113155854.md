# SGP-LoRA-VIT 实验结果收集和存储系统设计

## 1. 系统概述

本系统旨在为SGP-LoRA-VIT项目提供一个全面的实验结果收集和存储解决方案。系统重点关注标准化数据格式的设计，确保实验结果能够以统一、结构化的方式存储，便于后续的数据分析和可视化。系统特别关注跨域(Cross-Domain)和域内(Within-Domain)类增量学习实验的差异化数据管理需求。

基于对main.py及其关联代码的分析，当前系统已经具备基本的实验结果记录和多种子统计分析功能，但缺乏统一的结果管理和标准化存储机制。本设计旨在扩展现有功能，提供一个全面的实验结果收集和存储解决方案，为后续的可视化分析奠定坚实基础。

## 2. 结果收集结构设计

### 2.1 基于现有代码的存储结构分析

根据trainer.py中的build_log_dirs函数，当前的实验结果存储结构如下：

```
sldc_logs_{user}/
├── {dataset}_{model_type}/
│   ├── init-{init_cls}_inc-{increment}/
│   │   ├── lrank-{lora_rank}_ltype-{lora_type}/
│   │   │   ├── {method_params}/  # 如: t-1.0_k-log1p_p-1.0
│   │   │   │   ├── opt-{optimizer}_lr-{lrate}_b-{batch_size}_i-{iterations}/
│   │   │   │   │   ├── seed_{seed_id}/
│   │   │   │   │   │   ├── params.json
│   │   │   │   │   │   ├── record.log
│   │   │   │   │   │   └── checkpoints/
│   │   │   │   │   └── aggregate_results.json (多种子时)
│   │   │   │   └── multi_seed_statistics.json (多种子统计)
```

### 2.2 改进的统一存储结构设计

基于现有结构，设计以下改进的存储格式：

```
experiment_results/
├── experiments/                          # 实验配置和元数据
│   ├── configs/                          # 实验配置文件
│   │   ├── main_experiments/             # 主实验配置
│   │   │   ├── cross_domain/            # 跨域主实验配置
│   │   │   └── within_domain/          # 域内主实验配置
│   │   ├── ablation_experiments/         # 消融实验配置
│   │   │   ├── cross_domain/            # 跨域消融实验配置
│   │   │   └── within_domain/          # 域内消融实验配置
│   │   └── parameter_sensitivity/        # 参数敏感性实验配置
│   │       ├── cross_domain/            # 跨域参数敏感性配置
│   │       └── within_domain/          # 域内参数敏感性配置
│   └── metadata/                        # 实验元数据
│       ├── experiment_registry.json      # 实验注册表
│       ├── dataset_info.json            # 数据集信息
│       ├── cross_domain_datasets.json   # 跨域数据集信息
│       └── within_domain_datasets.json  # 域内数据集信息
├── raw_results/                         # 原始实验结果
│   ├── cross_domain/                    # 跨域实验结果
│   │   ├── {experiment_group}/         # 实验组别 (如: cross_domain_elevater)
│   │   │   ├── {model_type}/          # 模型类型 (如: vit-b-p16-mocov3)
│   │   │   │   ├── {timestamp}_{experiment_id}/
│   │   │   │   │   ├── config.json          # 实验配置
│   │   │   │   │   ├── results/             # 各种子结果
│   │   │   │   │   │   ├── seed_{seed_id}/
│   │   │   │   │   │   │   ├── params.json  # 参数配置
│   │   │   │   │   │   │   ├── metrics.json # 指标结果
│   │   │   │   │   │   │   ├── logs/        # 日志文件
│   │   │   │   │   │   │   └── checkpoints/ # 模型检查点
│   │   │   │   │   │   └── training_results.json # 训练结果
│   │   │   │   │   └── aggregate_results.json # 聚合结果
│   │   │   │   │   └── multi_seed_statistics.json # 多种子统计
│   │   │   │   └── summary.json         # 实验摘要
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── within_domain/                   # 域内实验结果
│       ├── {dataset_name}/             # 数据集名称 (如: cifar100_224)
│       │   ├── {model_type}/          # 模型类型 (如: vit-b-p16-mocov3)
│       │   │   ├── {init_cls}_inc-{increment}/  # 任务设置 (如: init-10_inc-10)
│       │   │   │   ├── {lora_config}/  # LoRA配置 (如: lrank-4_ltype-sgp_lora)
│       │   │   │   │   ├── {method_params}/ # 方法参数 (如: t-1.0_k-log1p_p-1.0)
│       │   │   │   │   ├── {optimizer_config}/ # 优化器配置 (如: opt-adamw_lr-0.0001_b-16_i-2000)
│       │   │   │   │   │   ├── seed_{seed_id}/
│       │   │   │   │   │   │   ├── params.json  # 参数配置
│       │   │   │   │   │   │   ├── metrics.json # 指标结果
│       │   │   │   │   │   │   ├── logs/        # 日志文件
│       │   │   │   │   │   │   └── checkpoints/ # 模型检查点
│       │   │   │   │   │   └── training_results.json # 训练结果
│       │   │   │   │   │   └── aggregate_results.json # 聚合结果
│       │   │   │   │   │   └── multi_seed_statistics.json # 多种子统计
│       │   │   │   │   └── summary.json         # 实验摘要
│       │   │   │   │   └── ...
│       │   │   │   └── ...
│       │   │   └── ...
│       │   └── ...
│       └── ...
```

### 2.3 基于现有代码的数据格式定义

#### 2.3.1 实验配置文件格式 (params.json)

基于trainer.py中的_filter_args_by_lora_type函数，params.json包含以下字段：

```json
{
  "dataset": "cars196_224",
  "smart_defaults": true,
  "user": "sgp_lora_vit_main",
  "test": false,
  "memory_size": 0,
  "memory_per_class": 0,
  "fixed_memory": false,
  "shuffle": true,
  "init_cls": 20,
  "increment": 20,
  "model_name": "sldc",
  "vit_type": "vit-b-p16-mocov3",
  "weight_decay": 3e-05,
  "seed_list": [1993, 1996, 1997],
  "iterations": 1500,
  "warmup_ratio": 0.1,
  "ca_epochs": 5,
  "optimizer": "adamw",
  "lrate": 0.0001,
  "batch_size": 16,
  "evaluate_final_only": true,
  "gamma_kd": 0.0,
  "update_teacher_each_task": true,
  "use_aux_for_kd": false,
  "kd_type": "feat",
  "distillation_transform": "linear",
  "eval_only": false,
  "lora_rank": 4,
  "lora_type": "sgp_lora",
  "weight_temp": 1.0,
  "weight_kind": "log1p",
  "weight_p": 1.0,
  "lda_reg_alpha": 0.1,
  "qda_reg_alpha1": 0.2,
  "qda_reg_alpha2": 0.9,
  "qda_reg_alpha3": 0.2,
  "auxiliary_data_path": "/data1/open_datasets",
  "aux_dataset": "imagenet",
  "auxiliary_data_size": 1024,
  "l2_protection": false,
  "l2_protection_lambda": 0.0001,
  "seed": 1993,
  "run_id": 0,
  "cross_domain": false,
  "cross_domain_datasets": ["imagenet-r", "caltech-101", "dtd"],
  "num_shots": 16,
  "num_samples_per_task_for_evaluation": 1000,
  "compensator_types": ["SeqFT", "SeqFT + linear", "SeqFT + weaknonlinear", "SeqFT + Hopfield"]
}
```

#### 2.3.2 训练结果文件格式 (training_results.json)

基于SubspaceLoRA类的analyze_task_results方法，训练结果包含以下字段：

```json
{
  "last_task_id": 10,
  "last_task_accuracies": {
    "SeqFT + LDA": 70.87,
    "SeqFT + QDA": 77.29,
    "SeqFT + attention_transform + LDA": 80.29,
    "SeqFT + attention_transform + QDA": 83.41
  },
  "average_accuracies": {
    "SeqFT + LDA": 78.507,
    "SeqFT + QDA": 83.229,
    "SeqFT + attention_transform + LDA": 85.371,
    "SeqFT + attention_transform + QDA": 87.476
  },
  "per_task_results": {
    "0": {
      "SeqFT + LDA": 90.43,
      "SeqFT + QDA": 91.53,
      "SeqFT + attention_transform + LDA": 90.43,
      "SeqFT + attention_transform + QDA": 91.53
    },
    "1": {
      "SeqFT + LDA": 87.89,
      "SeqFT + QDA": 91.1,
      "SeqFT + attention_transform + LDA": 90.73,
      "SeqFT + attention_transform + QDA": 91.94
    }
  }
}
```

#### 2.3.3 多种子统计文件格式 (multi_seed_statistics.json)

基于trainer.py中的analyze_all_results函数，多种子统计包含以下字段：

```json
{
  "summary": {
    "num_seeds": 3,
    "num_variants": 4,
    "num_tasks": 10,
    "variant_names": ["SeqFT + LDA", "SeqFT + QDA", "SeqFT + attention_transform + LDA", "SeqFT + attention_transform + QDA"],
    "task_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "dataset_names": ["imagenet-r", "cifar100_224", "cub200_224", "cars196_224"]
  },
  "variants": {
    "SeqFT + LDA": {
      "last_task_accuracy": {
        "mean": 70.87,
        "std": 0.5,
        "raw_values": [70.2, 71.5, 70.9]
      },
      "average_accuracy": {
        "mean": 78.507,
        "std": 0.3,
        "raw_values": [78.0, 79.1, 78.4]
      },
      "per_task_accuracies": {
        "0": {
          "dataset_name": "imagenet-r",
          "mean": 90.43,
          "std": 0.2,
          "raw_values": [90.2, 90.6, 90.5]
        },
        "1": {
          "dataset_name": "cifar100_224",
          "mean": 87.89,
          "std": 0.3,
          "raw_values": [87.5, 88.2, 87.9]
        }
      }
    }
  },
  "overall_summary": {
    "SeqFT + LDA": {
      "mean": 78.507,
      "std": 0.3,
      "num_seeds": 3
    },
    "SeqFT + QDA": {
      "mean": 83.229,
      "std": 0.4,
      "num_seeds": 3
    }
  }
}
```

### 2.4 关键指标和元数据定义

#### 2.4.1 核心性能指标

基于SubspaceLoRA类的analyze_task_results方法，系统记录以下核心指标：

1. **Last-Accuracy (最后任务准确率)**: 完成所有任务后，在全部类别上的平均分类准确率
   - 对应代码中的: `last_task_accuracies`

2. **Average-Accuracy (平均准确率)**: 每个任务结束后，在所有已见类别上的准确率的平均值
   - 对应代码中的: `average_accuracies`

3. **Per-Task Accuracies (每任务准确率)**: 每个任务结束时的准确率
   - 对应代码中的: `per_task_results`

4. **Forgetting Measure (遗忘度量)**: 模型在学习新任务后对旧任务性能的下降程度
   - 需要基于per_task_results计算得出


## 4. 总结

本设计提供了一个全面的实验结果收集和存储系统，重点关注数据存储格式的标准化，为后续的可视化分析奠定基础。

### 4.1 核心特点

1. **标准化数据格式**: 基于trainer.py中现有的数据结构，设计了统一的实验结果存储格式
2. **明确区分实验类型**: 清晰区分跨域和域内实验，提供不同的数据组织方式

### 4.2 实施建议

1. **优先实现数据标准化**: 首先实现数据收集和标准化存储功能
2. **逐步扩展功能**: 在数据标准化基础上，逐步添加查询、验证和聚合功能
3. **预留可视化接口**: 为后续的可视化功能预留标准化的数据接口
4. **保持向后兼容**: 提供数据迁移工具，确保现有实验数据的兼容性

该系统能够显著提高实验数据的组织性和可访问性，为SGP-LoRA-VIT项目的实验研究提供强有力的数据管理基础。通过标准化的数据格式，研究人员可以轻松地提取和分析实验结果，为后续的可视化和深入分析奠定坚实基础。

## 5. 自动化工具设计

### 5.1 结果收集脚本

#### 5.1.1 实验监控器

基于trainer.py中的现有功能，扩展实现：

```python
class ExperimentMonitor:
    """实验监控器，实时跟踪实验进度和结果"""
    def __init__(self, experiment_config):
        self.config = experiment_config
        self.progress_tracker = ProgressTracker()
        
    def monitor_experiment(self, experiment_id):
        """监控实验执行过程"""
        pass
        
    def collect_results(self, experiment_id):
        """收集实验结果"""
        pass
        
    def validate_results(self, results):
        """验证结果完整性"""
        pass
```

#### 5.1.2 结果聚合器

基于trainer.py中的analyze_all_results函数，扩展实现：

```python
class ResultAggregator:
    """结果聚合器，将多个实验结果进行汇总分析"""
    def __init__(self, result_storage):
        self.storage = result_storage
        
    def aggregate_multi_seed(self, experiment_id):
        """聚合多种子结果"""
        # 基于现有的analyze_all_results函数
        pass
        
    def aggregate_cross_dataset(self, experiment_ids):
        """聚合跨数据集结果"""
        pass
        
    def generate_summary(self, experiment_ids):
        """生成结果摘要"""
        pass
```

### 5.2 可视化生成脚本

#### 5.2.1 图表生成器

```python
class ChartGenerator:
    """图表生成器，自动生成各类可视化图表"""
    def __init__(self, style_config):
        self.style = style_config
        
    def generate_performance_comparison(self, results):
        """生成性能对比图"""
        pass
        
    def generate_sensitivity_analysis(self, results):
        """生成敏感性分析图"""
        pass
        
    def generate_ablation_study(self, results):
        """生成消融实验图"""
        pass
        
    def generate_cross_domain_analysis(self, results):
        """生成跨域分析图"""
        pass
        
    def generate_within_domain_analysis(self, results):
        """生成域内分析图"""
        pass
```

#### 5.2.2 报告生成器

```python
class ReportGenerator:
    """报告生成器，自动生成实验报告"""
    def __init__(self, template_dir):
        self.templates = template_dir
        
    def generate_html_report(self, experiment_ids):
        """生成HTML报告"""
        pass
        
    def generate_pdf_report(self, experiment_ids):
        """生成PDF报告"""
        pass
        
    def generate_latex_tables(self, results):
        """生成LaTeX表格"""
        pass
```

### 5.3 实验管理系统

#### 5.3.1 配置管理器

```python
class ConfigManager:
    """配置管理器，管理实验配置和参数"""
    def __init__(self, config_dir):
        self.config_dir = config_dir
        
    def load_config(self, config_id):
        """加载配置文件"""
        pass
        
    def save_config(self, config, config_id):
        """保存配置文件"""
        pass
        
    def validate_config(self, config):
        """验证配置有效性"""
        pass
        
    def generate_experiment_configs(self, base_config, parameter_grid):
        """生成实验配置网格"""
        pass
```

#### 5.3.2 实验调度器

```python
class ExperimentScheduler:
    """实验调度器，管理实验执行顺序和资源分配"""
    def __init__(self, resource_manager):
        self.resources = resource_manager
        
    def schedule_experiment(self, experiment_config):
        """调度实验执行"""
        pass
        
    def monitor_resources(self):
        """监控资源使用情况"""
        pass
        
    def optimize_schedule(self, pending_experiments):
        """优化实验调度"""
        pass
```

## 6. 实验版本控制和结果追踪

### 6.1 版本控制策略

#### 6.1.1 代码版本控制
- 使用Git管理代码版本
- 为每个实验创建特定的代码标签
- 记录代码变更对实验结果的影响

#### 6.1.2 配置版本控制
- 版本化实验配置文件
- 追踪参数变更历史
- 支持配置回滚和比较

#### 6.1.3 结果版本控制
- 为每个实验结果分配唯一版本号
- 建立结果之间的依赖关系
- 支持结果版本比较和回溯

### 6.2 结果追踪机制

#### 6.2.1 实验血缘追踪
- 记录实验的输入数据来源
- 追踪实验参数变更历史
- 建立实验之间的依赖关系

#### 6.2.2 可重现性保证
- 记录完整的实验环境信息
- 保存随机种子和确定性设置
- 提供实验重现指南

## 7. 系统实现架构

### 7.1 模块化设计

```python
# 核心模块结构
experiment_system/
├── core/                    # 核心功能模块
│   ├── __init__.py
│   ├── config.py            # 配置管理
│   ├── storage.py           # 存储管理
│   ├── collector.py         # 结果收集
│   ├── processor.py         # 数据处理
│   └── validator.py         # 结果验证
├── analysis/                # 分析模块
│   ├── __init__.py
│   ├── statistics.py        # 统计分析
│   ├── comparison.py        # 比较分析
│   └── sensitivity.py       # 敏感性分析
├── visualization/           # 可视化模块
│   ├── __init__.py
│   ├── charts.py            # 图表生成
│   ├── reports.py           # 报告生成
│   └── dashboard.py         # 仪表板
├── management/              # 管理模块
│   ├── __init__.py
│   ├── scheduler.py         # 实验调度
│   ├── monitor.py           # 实验监控
│   └── tracker.py           # 结果追踪
└── utils/                   # 工具模块
    ├── __init__.py
    ├── file_utils.py        # 文件操作
    ├── data_utils.py        # 数据处理
    └── plot_utils.py        # 绘图工具
```

### 7.2 接口设计

#### 7.2.1 统一API接口

```python
class ExperimentSystemAPI:
    """实验系统统一API接口"""
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.initialize_components()
        
    def create_experiment(self, experiment_config):
        """创建新实验"""
        pass
        
    def run_experiment(self, experiment_id):
        """运行实验"""
        pass
        
    def collect_results(self, experiment_id):
        """收集结果"""
        pass
        
    def analyze_results(self, experiment_ids):
        """分析结果"""
        pass
        
    def generate_report(self, experiment_ids, report_type):
        """生成报告"""
        pass
```

#### 7.2.2 插件接口

```python
class PluginInterface:
    """插件接口，支持扩展功能"""
    def process_results(self, results):
        """处理结果"""
        pass
        
    def generate_visualization(self, results):
        """生成可视化"""
        pass
        
    def validate_experiment(self, config):
        """验证实验配置"""
        pass
```

## 8. 部署和使用指南

### 8.1 系统部署

#### 8.1.1 环境要求
- Python 3.8+
- PyTorch 1.8+
- 推荐配置：至少16GB内存，GPU支持

#### 8.1.2 安装步骤

```bash
# 1. 克隆代码仓库
git clone https://github.com/your-repo/experiment_system.git
cd experiment_system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境
python setup.py configure

# 4. 初始化数据库
python setup.py init_db
```

### 8.2 使用指南

#### 8.2.1 基本使用流程

```python
# 1. 创建实验配置
config = {
    "experiment_type": "main",
    "dataset_config": {
        "type": "cross_domain",
        "datasets": ["imagenet-r", "cifar100_224"],
        "init_cls": 20,
        "increment": 20
    },
    "model_config": {
        "model_name": "sldc",
        "vit_type": "vit-b-p16-mocov3",
        "lora_rank": 4,
        "lora_type": "sgp_lora"
    },
    "training_config": {
        "optimizer": "adamw",
        "lrate": 0.0001,
        "batch_size": 16,
        "iterations": 1500,
        "seed_list": [1993, 1996, 1997]
    }
}

# 2. 运行实验
experiment_id = api.create_experiment(config)
api.run_experiment(experiment_id)

# 3. 收集和分析结果
results = api.collect_results(experiment_id)
analysis = api.analyze_results([experiment_id])

# 4. 生成报告
api.generate_report([experiment_id], "html")
```

#### 8.2.2 高级功能使用

```python
# 批量实验管理
batch_config = generate_batch_configs(base_config, parameter_grid)
experiment_ids = [api.create_experiment(cfg) for cfg in batch_config]

# 并行执行
api.run_experiments_parallel(experiment_ids)

# 跨实验比较
comparison_results = api.compare_experiments(experiment_ids)

# 参数敏感性分析
sensitivity_results = api.analyze_sensitivity(experiment_ids, target_params)
```

## 9. 总结

本设计提供了一个全面的实验结果收集和可视化系统，具有以下特点：

1. **基于现有代码**: 充分利用trainer.py中已有的结果记录和统计分析功能
2. **明确区分实验类型**: 清晰区分跨域和域内实验，提供不同的分析视角
3. **统一的数据格式**: 标准化的实验结果存储格式，便于分析和比较
4. **自动化处理**: 自动化的结果收集、聚合和统计分析
5. **丰富的可视化**: 多种图表类型，支持不同分析需求
6. **灵活的扩展性**: 模块化设计，支持功能扩展
7. **完整的追踪**: 实验版本控制和结果血缘追踪
8. **易用的接口**: 简洁的API设计，降低使用门槛

该系统能够显著提高实验效率，促进结果的可重现性和可比性，为SGP-LoRA-VIT项目的实验研究提供强有力的支持。
## 10. 实现细节和扩展建议

### 10.1 与现有代码的集成

#### 10.1.1 扩展trainer.py

基于现有的analyze_all_results函数，建议添加以下功能：

```python
def enhanced_analyze_all_results(all_results: dict, dataset_names: list = None, save_json: bool = True, 
                             output_path: str = None, 
                             enable_visualization: bool = False,
                             enable_cross_domain_analysis: bool = False) -> dict:
    """
    增强版多种子统计分析函数
    
    新增功能:
    - 可视化生成选项
    - 跨域实验专门分析
    - 自动报告生成
    - 结果验证和异常检测
    """
    # 基于现有实现扩展
    statistics_results = analyze_all_results(all_results, dataset_names, save_json, output_path)
    
    # 新增功能实现
    if enable_visualization:
        generate_basic_visualizations(statistics_results, output_path)
    
    if enable_cross_domain_analysis and dataset_names:
        cross_domain_stats = analyze_cross_domain_patterns(statistics_results, dataset_names)
        statistics_results["cross_domain_analysis"] = cross_domain_stats
    
    # 结果验证
    validation_results = validate_experiment_results(statistics_results)
    statistics_results["validation"] = validation_results
    
    return statistics_results
```

#### 10.1.2 扩展SubspaceLoRA类

在SubspaceLoRA类中添加更多指标记录：

```python
class EnhancedSubspaceLoRA(SubspaceLoRA):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)
        self.training_metrics = {
            "loss_history": [],
            "accuracy_history": [],
            "timing_history": [],
            "memory_usage": []
        }
    
    def enhanced_loop(self, data_manager) -> Dict[str, Any]:
        """增强版循环函数，记录更多训练指标"""
        # 扩展现有loop函数
        results = super().loop(data_manager)
        
        # 添加额外指标
        results["training_metrics"] = self.training_metrics
        results["model_efficiency"] = self.calculate_model_efficiency()
        
        return results
    
    def calculate_model_efficiency(self) -> Dict[str, Any]:
        """计算模型效率指标"""
        trainable_params = self.count_trainable_parameters()
        total_params = self.count_total_parameters()
        
        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "efficiency_ratio": (trainable_params["total"] / total_params) * 100,
            "lora_efficiency": (trainable_params["lora"] / trainable_params["total"]) * 100
        }
```

### 10.2 可视化实现细节

#### 10.2.1 基于matplotlib/seaborn的图表生成

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_performance_comparison_charts(statistics_results, output_dir):
    """生成性能对比图表"""
    # 提取数据
    overall_summary = statistics_results.get("overall_summary", {})
    
    # 创建DataFrame
    df = pd.DataFrame(overall_summary).T
    df['method'] = df.index
    df['mean'] = df['mean'].astype(float)
    df['std'] = df['std'].astype(float)
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 柱状图：平均准确率
    sns.barplot(data=df, x='method', y='mean', ax=ax1, palette='viridis')
    ax1.set_title('Average Accuracy Comparison')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 误差条图：准确率±标准差
    ax2.errorbar(df['method'], df['mean'], df['std'], 
                 fmt='o', color='skyblue', ecolor='darkblue', capsize=5)
    ax2.set_title('Accuracy with Standard Deviation')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
```

#### 10.2.2 跨域实验专门可视化

```python
def generate_cross_domain_heatmaps(statistics_results, output_dir):
    """生成跨域实验热力图"""
    if "cross_domain_analysis" not in statistics_results:
        return
    
    cross_domain_data = statistics_results["cross_domain_analysis"]
    
    # 创建热力图数据矩阵
    datasets = cross_domain_data["dataset_names"]
    methods = list(cross_domain_data["performance_matrix"].keys())
    
    performance_matrix = np.array([
        cross_domain_data["performance_matrix"][method] for method in methods
    ])
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_matrix, 
                xticklabels=datasets, 
                yticklabels=methods,
                annot=True, 
                cmap='YlOrRd',
                fmt='.2f')
    plt.title('Cross-Domain Performance Heatmap')
    plt.xlabel('Target Dataset')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cross_domain_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
```

### 10.3 自动化报告生成

#### 10.3.1 HTML报告模板

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>实验结果报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .chart { text-align: center; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { font-weight: bold; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SGP-LoRA-VIT 实验结果报告</h1>
        <p>生成时间: {{timestamp}}</p>
        <p>实验配置: {{experiment_config}}</p>
    </div>
    
    <div class="section">
        <h2>性能概览</h2>
        <div class="chart">
            <img src="performance_comparison.png" alt="性能对比图">
        </div>
        <table>
            <tr>
                <th>方法</th>
                <th>平均准确率</th>
                <th>标准差</th>
                <th>最后任务准确率</th>
            </tr>
            {{performance_table_rows}}
        </table>
    </div>
    
    <div class="section">
        <h2>详细分析</h2>
        {{detailed_analysis_content}}
    </div>
</body>
</html>
```

### 10.4 部署和维护建议

#### 10.4.1 系统部署步骤

1. **环境准备**
   ```bash
   # 创建虚拟环境
   python -m venv experiment_system_env
   source experiment_system_env/bin/activate
   
   # 安装依赖
   pip install -r requirements.txt
   ```

2. **配置初始化**
   ```bash
   # 创建配置目录
   mkdir -p experiment_results/experiments/configs
   mkdir -p experiment_results/raw_results
   mkdir -p experiment_results/processed_results
   
   # 初始化实验注册表
   python tools/init_experiment_registry.py
   ```

3. **与现有代码集成**
   ```bash
   # 备份原始trainer.py
   cp trainer.py trainer.py.backup
   
   # 应用增强补丁
   python tools/apply_enhancements.py
   ```

#### 10.4.2 维护和更新策略

1. **定期数据备份**
   - 每日自动备份实验结果
   - 保留重要实验的多个版本
   - 使用Git LFS管理大型结果文件

2. **系统监控**
   - 监控磁盘空间使用情况
   - 跟踪实验执行状态
   - 记录系统性能指标

3. **版本兼容性**
   - 保持向后兼容性
   - 提供数据迁移工具
   - 定期更新文档和示例

## 11. 总结

本设计提供了一个全面的实验结果收集和可视化系统，具有以下特点：

1. **基于现有代码**: 充分利用trainer.py中已有的结果记录和统计分析功能
2. **明确区分实验类型**: 清晰区分跨域和域内实验，提供不同的分析视角
3. **统一的数据格式**: 标准化的实验结果存储格式，便于分析和比较
4. **自动化处理**: 自动化的结果收集、聚合和统计分析
5. **丰富的可视化**: 多种图表类型，支持不同分析需求
6. **灵活的扩展性**: 模块化设计，支持功能扩展
7. **完整的追踪**: 实验版本控制和结果血缘追踪
8. **易用的接口**: 简洁的API设计，降低使用门槛

该系统能够显著提高实验效率，促进结果的可重现性和可比性，为SGP-LoRA-VIT项目的实验研究提供强有力的支持。