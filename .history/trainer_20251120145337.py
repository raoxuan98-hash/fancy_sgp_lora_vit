import os
import sys
import logging
import torch
import random
import numpy as np
from collections.abc import Mapping, Sequence
# from models.subspace_lora import SubspaceLoRA
from models.subspace_lora import SubspaceLoRA
from utils.data_manager import WithinDomainDataManager, CrossDomainDataManagerCore
from utils.balanced_cross_domain_data_manager import BalancedCrossDomainDataManagerCore
from utils.toolkit import count_parameters
import re

def train(args):
    all_results = {}
    
    for run_id, seed in enumerate(args['seed_list']):
        args['seed'], args['run_id'] = seed, run_id
        logfile_head, logfile_name = build_log_dirs(args)
        args['log_path'] = logfile_name
        
        # Configure logging with unbuffered file handler for real-time updates
        log_file_path = os.path.join(logfile_name, 'record.log')
        
        # 清除现有的日志处理器，避免冲突
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(filename=log_file_path, mode='a', encoding='utf-8')
        file_handler.stream.reconfigure(line_buffering=True)  # Enable line buffering
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s [%(filename)s] => %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 配置根日志记录器
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 打印日志文件路径，方便用户查找
        print(f"📁 日志文件路径: {log_file_path}")
        print(f"💡 提示: 使用 'tail -f {log_file_path}' 实时查看日志")
        print("-" * 80)
        
        args['log_path'] = logfile_name
        results = train_single_run(args)
        all_results[f"seed_{seed}"] = results
    
    # 在所有种子运行完成后，进行统计分析
    if len(all_results) > 1:  # 只有多于一个种子时才进行统计分析
        dataset_names = args.get('cross_domain_datasets', None)
        analyze_all_results(all_results, dataset_names, save_json=True)
    

def train_single_run(args, return_model: bool = False):
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    print_args(args)
    
    # Initialize data manager and model

    if args['cross_domain']:
        # 使用平衡后的cross-domain数据集
        data_manager = BalancedCrossDomainDataManagerCore(
            dataset_names=args['cross_domain_datasets'],
            balanced_datasets_root="balanced_datasets",
            shuffle=args['shuffle'],
            seed=args['seed'],
            num_shots=args.get('num_shots', 0),
            num_samples_per_task_for_evaluation=args.get('num_samples_per_task_for_evaluation', 0),
            use_balanced_datasets=True)
    else:
        data_manager = WithinDomainDataManager(
            dataset_name=args['dataset'],
            shuffle=args['shuffle'],
            seed=args['seed'],
            init_cls=args['init_cls'],
            increment=args['increment'],
            args=args)
    
    model = SubspaceLoRA(args)
    logging.info(f'All params: {count_parameters(model.network)}')
    logging.info(f'Trainable params: {count_parameters(model.network, True)}')
    final_results = model.loop(data_manager)
    
    # 添加log_path到结果中，以便aggregate_seed_results可以找到它
    if 'log_path' in args:
        final_results['log_path'] = args['log_path']
    
    if return_model:
        return final_results, model
    return final_results

def set_device(device_type):
    """Properly set the device (either CPU or GPU) based on input"""
    if isinstance(device_type, (list, tuple)):
        return [torch.device(f'cuda:{d}' if d != -1 else 'cpu') for d in device_type]
    return torch.device('cuda' if device_type != -1 else 'cpu')

def set_random(seed):
    """Set random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    """Log the arguments for this run"""
    for key, value in args.items():
        logging.info(f'{key}: {value}')


import os
from pathlib import Path
import json

def _filter_args_by_lora_type(args: dict) -> dict:
    """
    过滤参数字典，只保留与当前LoRA类型相关的参数
    这样可以避免在params.json中保存不相关的参数，导致日志命名混乱
    """
    lora_type = args.get('lora_type', 'basic_lora')
    filtered_args = args.copy()
    
    # 定义每种LoRA类型相关的参数
    sgp_lora_params = {'weight_temp', 'weight_kind', 'weight_p'}
    nsp_lora_params = {'nsp_eps', 'nsp_weight'}
    
    # 移除与当前LoRA类型不相关的参数
    if lora_type == 'sgp_lora':
        # 保留SGP参数，移除NSP参数
        for param in nsp_lora_params:
            filtered_args.pop(param, None)
    elif lora_type == 'nsp_lora':
        # 保留NSP参数，移除SGP参数
        for param in sgp_lora_params:
            filtered_args.pop(param, None)
    elif lora_type == 'basic_lora':
        # 移除所有LoRA特定参数
        for param in sgp_lora_params.union(nsp_lora_params):
            filtered_args.pop(param, None)
    elif lora_type == 'full':
        # 移除所有LoRA特定参数
        for param in sgp_lora_params.union(nsp_lora_params):
            filtered_args.pop(param, None)
    
    return filtered_args

def build_log_dirs(args: dict, root_dir="."):
    """
    根据 args 构建多级日志目录，确保不同 LoRA 类型的参数正确分离
    
    目录结构改进：
    - 顶层目录现在包含实验类型标识（cross_domain 或 within_domain）
    - cross_domain实验：sldc_logs_{user}_cross_domain/{datasets}_{vit_type}/...
    - within_domain实验：sldc_logs_{user}_within_domain/{dataset}_{vit_type}/...
    
    这样可以明确区分cross-domain和within-domain的实验结果，避免混淆
    """

    def sanitize_filename(s: str) -> str:
        """移除或替换文件名中的非法字符"""
        # Windows 非法字符: \ / : * ? " < > |
        s = re.sub(r'[\\/:*?"<>|]', '_', str(s))
        # 可选：压缩连续下划线
        s = re.sub(r'_+', '_', s)
        return s.strip('_')

    def short(s: str, maxlen=40):
        """截断过长字符串，不加 hash，仅保留可读性"""
        s = sanitize_filename(str(s))
        if len(s) <= maxlen:
            return s
        return s[:maxlen].rstrip('_')  # 避免截断在下划线处

    def _get_lora_specific_params(lora_type: str, args: dict) -> list:
        """获取特定 LoRA 类型的参数，避免交叉污染"""
        params = []
        
        if lora_type == 'sgp_lora':
            # SGP LoRA 特有参数
            if 'weight_temp' in args:
                params.append(f"t-{short(args['weight_temp'])}")
            if 'weight_kind' in args:
                params.append(f"k-{short(args['weight_kind'])}")
            # 始终包含 weight_p 参数，即使是默认值，以确保不同参数组合的实验结果被正确区分
            if 'weight_p' in args:
                params.append(f"p-{short(args['weight_p'])}")
                
        elif lora_type == 'nsp_lora':
            # NSP LoRA 特有参数
            if 'nsp_eps' in args:
                params.append(f"eps-{short(args['nsp_eps'])}")
            if 'nsp_weight' in args:
                params.append(f"w-{short(args['nsp_weight'])}")
                
        elif lora_type == 'basic_lora':
            # Basic LoRA 通常没有额外参数，但可以在这里添加
            pass
            
        elif lora_type == 'full':
            # Full fine-tuning 可能有的参数
            pass
            
        return params

    def _get_kd_params(args: dict) -> list:
        """获取知识蒸馏相关参数，统一命名规则"""
        kd_params = []
        
        if args.get('gamma_kd', 0.0) > 0.0:
            kd_params.append(f"kd-{short(args['gamma_kd'])}")
            if 'kd_type' in args:
                kd_params.append(f"type-{short(args['kd_type'])}")
            if 'distillation_transform' in args:
                kd_params.append(f"dt-{short(args['distillation_transform'])}")
            if args.get('use_aux_for_kd', False):
                kd_params.append("aux")
            # 添加update_teacher_each_task参数，简写为utt
            if 'update_teacher_each_task' in args:
                kd_params.append(f"utt-{short(args['update_teacher_each_task'])}")
                
        return kd_params

    def _validate_parameters(args: dict) -> None:
        """验证参数组合的合理性"""
        lora_type = args.get('lora_type', 'basic_lora')
        
        # 检查 LoRA 特定参数是否被误用
        if lora_type != 'sgp_lora':
            sgp_params = ['weight_temp', 'weight_kind', 'weight_p']
            for param in sgp_params:
                if param in args and args[param] is not None:
                    logging.warning(f"⚠️ Parameter '{param}' is being used with lora_type='{lora_type}', but it's specific to sgp_lora")
        
        if lora_type != 'nsp_lora':
            nsp_params = ['nsp_eps', 'nsp_weight']
            for param in nsp_params:
                if param in args and args[param] is not None:
                    logging.warning(f"⚠️ Parameter '{param}' is being used with lora_type='{lora_type}', but it's specific to nsp_lora")

    # 参数验证
    _validate_parameters(args)

    # 确定实验类型：cross-domain 或 within-domain
    is_cross_domain = args.get('cross_domain', False)
    
    # 顶层：模型、用户信息和实验类型（明确区分cross-domain和within-domain）
    experiment_type = "cross_domain" if is_cross_domain else "within_domain"
    base_dir = os.path.join(
        root_dir,
        f"{short(args['model_name'])}_logs_{short(args['user'])}_{experiment_type}"
    )

    # 根据实验类型构建不同的二级目录结构
    if is_cross_domain:
        # 跨域实验：使用order标识而不是具体的数据集列表
        if 'cross_domain_datasets' in args:
            # 使用order1作为当前数据集顺序的标识，将来有其他顺序时可命名为order2, order3等
            task_dir = os.path.join(
                base_dir,
                f"order1_{short(args['vit_type'])}",
                f"shots-{short(args.get('num_shots', 0))}",
                f"lrank-{short(args.get('lora_rank', 'NA'))}_ltype-{short(args.get('lora_type', 'NA'))}"
            )
        else:
            # 如果没有指定跨域数据集，使用默认标识
            task_dir = os.path.join(
                base_dir,
                f"unknown_{short(args['vit_type'])}",
                f"shots-{short(args.get('num_shots', 0))}",
                f"lrank-{short(args.get('lora_rank', 'NA'))}_ltype-{short(args.get('lora_type', 'NA'))}"
            )
    else:
        # 域内实验：使用传统的init_cls和increment参数
        task_dir = os.path.join(
            base_dir,
            f"{short(args['dataset'])}_{short(args['vit_type'])}",
            f"init-{short(args['init_cls'])}_inc-{short(args['increment'])}",
            f"lrank-{short(args.get('lora_rank', 'NA'))}_ltype-{short(args.get('lora_type', 'NA'))}"
        )

    # 三级：LoRA 特定参数（只包含相关的）
    lora_params = _get_lora_specific_params(args.get('lora_type', 'basic_lora'), args)
    
    # 四级：知识蒸馏参数（如果有）
    kd_params = _get_kd_params(args)
    
    # 合并 LoRA 和 KD 参数
    method_params = lora_params + kd_params
    
    # 构建方法参数目录
    if method_params:
        method_subdir = "_".join(method_params)
        method_dir = os.path.join(task_dir, short(method_subdir, maxlen=80))
    else:
        method_dir = task_dir

    # 五级：优化器和训练参数（不包含种子，种子将在子目录中处理）
    opt_params = [
        f"opt-{args['optimizer']}",
        f"lr-{short(args['lrate'])}",
        f"b-{args['batch_size']}",
        f"i-{args['iterations']}"
    ]
    opt_str = "_".join(opt_params)
    opt_dir = os.path.join(method_dir, short(opt_str, maxlen=80))

    # === 逐级创建目录 ===
    abs_log_dir = os.path.abspath(opt_dir)
    current = Path(abs_log_dir).root
    for part in Path(abs_log_dir).parts[1:]:
        current = Path(current) / part
        current.mkdir(exist_ok=True)

    # 保存过滤后的参数到 JSON，避免参数交叉污染
    filtered_args = _filter_args_by_lora_type(args)
    params_json = Path(abs_log_dir) / "params.json"
    if not params_json.exists():
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(filtered_args, f, ensure_ascii=False, indent=2)

    # 为每个种子创建子目录
    seed_dir = os.path.join(abs_log_dir, f"seed_{args['seed']}")
    os.makedirs(seed_dir, exist_ok=True)

    # 注意：这里不能使用 logging.info，因为日志还没有配置
    # 目录信息会在日志配置完成后通过 print_args 函数记录

    return os.path.dirname(abs_log_dir), str(seed_dir)

def analyze_all_results(all_results: dict, dataset_names: list = [], save_json: bool = True, output_path: str = "") -> dict:
    """
    分析all_results中多个随机种子的结果，计算平均值和标准差并记录到日志
    
    Args:
        all_results: 包含多个随机种子结果的字典
        dataset_names: 数据集名称列表，用于日志输出
        save_json: 是否将统计结果保存为JSON文件
        output_path: JSON文件保存路径，如果为None则自动生成
    
    Returns:
        dict: 包含统计结果的字典
    """
    import numpy as np
    import json
    from pathlib import Path
    
    if not all_results:
        logging.warning("📊 all_results为空，无法进行统计分析")
        return {}
    
    # 获取所有种子和变体名称
    seed_keys = list(all_results.keys())
    if len(seed_keys) == 0:
        logging.warning("📊 没有找到任何种子结果")
        return {}
    
    # 从第一个种子结果中获取变体名称和任务信息
    first_seed_result = all_results[seed_keys[0]]
    variant_names = set()
    
    # 从last_task_accuracies获取变体名称
    if 'last_task_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['last_task_accuracies'].keys())
    
    # 从average_accuracies获取变体名称
    if 'average_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['average_accuracies'].keys())
    
    # 从per_task_results获取变体名称
    if 'per_task_results' in first_seed_result:
        for task_result in first_seed_result['per_task_results'].values():
            variant_names.update(task_result.keys())
    
    variant_names = sorted(list(variant_names))
    
    if not variant_names:
        logging.warning("📊 没有找到任何变体名称")
        return {}
    
    # 获取任务ID列表
    task_ids = []
    if 'per_task_results' in first_seed_result:
        task_ids = sorted(first_seed_result['per_task_results'].keys())
    
    num_seeds = len(seed_keys)
    logging.info(f"📊 开始分析 {num_seeds} 个随机种子的实验结果")
    logging.info(f"📊 发现 {len(variant_names)} 个变体: {', '.join(variant_names)}")
    if task_ids:
        logging.info(f"📊 发现 {len(task_ids)} 个任务: {', '.join(map(str, task_ids))}")
    
    # 初始化统计结果字典
    statistics_results = {
        "summary": {
            "num_seeds": num_seeds,
            "num_variants": len(variant_names),
            "num_tasks": len(task_ids),
            "variant_names": variant_names,
            "task_ids": task_ids,
            "dataset_names": dataset_names
        },
        "variants": {}
    }
    
    # 记录统计结果
    logging.info("=" * 80)
    logging.info("📈 多种子统计分析结果")
    logging.info("=" * 80)
    
    for variant in variant_names:
        logging.info(f"\n🔍 变体: {variant}")
        logging.info("-" * 60)
        
        # 初始化变体统计结果
        variant_stats = {
            "last_task_accuracy": {},
            "average_accuracy": {},
            "per_task_accuracies": {},
            "class_wise_accuracy": {}
        }
        
        # 收集最后任务准确率数据
        last_task_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'last_task_accuracies' in seed_result and variant in seed_result['last_task_accuracies']:
                last_task_accs.append(seed_result['last_task_accuracies'][variant])
        
        if last_task_accs:
            mean_last = np.mean(last_task_accs)
            std_last = np.std(last_task_accs)
            variant_stats["last_task_accuracy"] = {
                "mean": float(round(mean_last, 2)),
                "std": float(round(std_last, 2)),
                "raw_values": [float(round(acc, 2)) for acc in last_task_accs]
            }
            logging.info(f"  最后任务准确率: {mean_last:.2f}% ± {std_last:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in last_task_accs])}")
        else:
            variant_stats["last_task_accuracy"] = {"error": "无数据"}
            logging.info(f"  最后任务准确率: 无数据")
        
        # 收集平均准确率数据
        avg_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'average_accuracies' in seed_result and variant in seed_result['average_accuracies']:
                avg_accs.append(seed_result['average_accuracies'][variant])
        
        if avg_accs:
            mean_avg = np.mean(avg_accs)
            std_avg = np.std(avg_accs)
            variant_stats["average_accuracy"] = {
                "mean": float(round(mean_avg, 2)),
                "std": float(round(std_avg, 2)),
                "raw_values": [float(round(acc, 2)) for acc in avg_accs]
            }
            logging.info(f"  平均准确率: {mean_avg:.2f}% ± {std_avg:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in avg_accs])}")
        else:
            variant_stats["average_accuracy"] = {"error": "无数据"}
            logging.info(f"  平均准确率: 无数据")
        
        # 收集class-wise平均准确率数据（仅cross-domain场景）
        class_wise_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if ('class_wise_accuracies' in seed_result and
                variant in seed_result['class_wise_accuracies']):
                class_wise_accs.append(seed_result['class_wise_accuracies'][variant])
        
        if class_wise_accs:
            mean_class_wise = np.mean(class_wise_accs)
            std_class_wise = np.std(class_wise_accs)
            variant_stats["class_wise_accuracy"] = {
                "mean": float(round(mean_class_wise, 2)),
                "std": float(round(std_class_wise, 2)),
                "raw_values": [float(round(acc, 2)) for acc in class_wise_accs]
            }
            logging.info(f"  Class-wise平均准确率: {mean_class_wise:.2f}% ± {std_class_wise:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in class_wise_accs])}")
        else:
            variant_stats["class_wise_accuracy"] = {"error": "无数据"}
            logging.info(f"  Class-wise平均准确率: 无数据")
        
        # 收集平均class-wise准确率数据（新增功能：所有任务的class-wise平均准确度的平均值）
        average_class_wise_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if ('average_class_wise_accuracies' in seed_result and
                variant in seed_result['average_class_wise_accuracies']):
                average_class_wise_accs.append(seed_result['average_class_wise_accuracies'][variant])
        
        if average_class_wise_accs:
            mean_avg_class_wise = np.mean(average_class_wise_accs)
            std_avg_class_wise = np.std(average_class_wise_accs)
            variant_stats["average_class_wise_accuracy"] = {
                "mean": float(round(mean_avg_class_wise, 2)),
                "std": float(round(std_avg_class_wise, 2)),
                "raw_values": [float(round(acc, 2)) for acc in average_class_wise_accs]
            }
            logging.info(f"  平均Class-wise准确率(所有任务): {mean_avg_class_wise:.2f}% ± {std_avg_class_wise:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in average_class_wise_accs])}")
        else:
            variant_stats["average_class_wise_accuracy"] = {"error": "无数据"}
            logging.info(f"  平均Class-wise准确率(所有任务): 无数据")
        
        # 记录每个任务的class-wise准确度（独立显示，不依赖于average_class_wise_accs）
        if task_ids:
            logging.info(f"  各任务Class-wise准确率:")
            # 从第一个种子获取每个任务的class-wise准确度
            first_seed_result = all_results[seed_keys[0]]
            if 'per_task_class_wise_accuracies' in first_seed_result and variant in first_seed_result['per_task_class_wise_accuracies']:
                per_task_accs = first_seed_result['per_task_class_wise_accuracies'][variant]
                for i, task_id in enumerate(task_ids):
                    if i < len(per_task_accs):
                        dataset_name = dataset_names[task_id - 1] if dataset_names and task_id - 1 < len(dataset_names) else f"Task {task_id}"
                        logging.info(f"    {dataset_name}: {per_task_accs[i]:.2f}%")
            else:
                logging.info(f"    无per_task_class_wise_accuracies数据")
        
        # 收集每个任务的准确率数据
        if task_ids:
            logging.info(f"  各任务准确率:")
            for task_id in task_ids:
                task_accs = []
                for seed_key in seed_keys:
                    seed_result = all_results[seed_key]
                    if ('per_task_results' in seed_result and
                        task_id in seed_result['per_task_results'] and
                        variant in seed_result['per_task_results'][task_id]):
                        # 确保只获取数值类型的数据
                        task_value = seed_result['per_task_results'][task_id][variant]
                        if isinstance(task_value, (int, float)):
                            task_accs.append(task_value)
                
                if task_accs:
                    mean_task = np.mean(task_accs)
                    std_task = np.std(task_accs)
                    dataset_name = dataset_names[task_id - 1] if dataset_names and task_id - 1 < len(dataset_names) else f"Task {task_id}"
                    variant_stats["per_task_accuracies"][str(task_id)] = {
                        "dataset_name": dataset_name,
                        "mean": float(round(mean_task, 2)),
                        "std": float(round(std_task, 2)),
                        "raw_values": [float(round(acc, 2)) for acc in task_accs]
                    }
                    logging.info(f"    {dataset_name}: {mean_task:.2f}% ± {std_task:.2f}%")
                    logging.info(f"      详细数据: {', '.join([f'{acc:.2f}%' for acc in task_accs])}")
                else:
                    dataset_name = dataset_names[task_id - 1] if dataset_names and task_id - 1 < len(dataset_names) else f"Task {task_id}"
                    variant_stats["per_task_accuracies"][str(task_id)] = {
                        "dataset_name": dataset_name,
                        "error": "无数据"
                    }
                    logging.info(f"    {dataset_name}: 无数据")
    
        statistics_results["variants"][variant] = variant_stats
    
    # 记录整体统计摘要
    logging.info("\n" + "=" * 80)
    logging.info("📋 整体性能摘要")
    logging.info("=" * 80)
    
    # 添加整体摘要到统计结果
    statistics_results["overall_summary"] = {}
    
    for variant in variant_names:
        # 收集平均准确率用于整体比较
        avg_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'average_accuracies' in seed_result and variant in seed_result['average_accuracies']:
                avg_accs.append(seed_result['average_accuracies'][variant])
        
        if avg_accs:
            mean_avg = np.mean(avg_accs)
            std_avg = np.std(avg_accs)
            statistics_results["overall_summary"][variant] = {
                "mean": float(round(mean_avg, 2)),
                "std": float(round(std_avg, 2)),
                "num_seeds": len(avg_accs)
            }
            logging.info(f"  {variant:<20}: {mean_avg:.2f}% ± {std_avg:.2f}% (基于 {len(avg_accs)} 个种子)")
    
    logging.info("=" * 80)
    logging.info(f"📊 统计分析完成，共分析 {num_seeds} 个种子，{len(variant_names)} 个变体")
    logging.info("=" * 80)
    
    # 保存JSON文件
    if save_json:
        if not output_path:
            # 自动生成输出路径
            if 'log_path' in first_seed_result:
                log_dir = Path(first_seed_result['log_path']).parent
            else:
                log_dir = Path("./statistics_results")
            log_dir.mkdir(exist_ok=True)
            output_path = str(log_dir / "multi_seed_statistics.json")
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(output_path_obj), 'w', encoding='utf-8') as f:
            json.dump(statistics_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"📁 统计结果已保存到: {output_path_obj}")
    
    return statistics_results