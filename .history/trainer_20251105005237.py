import os
import sys
import logging
import torch
import random
import numpy as np
from collections.abc import Mapping, Sequence
# from models.subspace_lora import SubspaceLoRA
from models.subspace_lora import SubspaceLoRA
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import re

def train(args):
    all_results = {}
    
    for run_id, seed in enumerate(args['seed_list']):
        args['seed'], args['run_id'] = seed, run_id
        logfile_head, logfile_name = build_log_dirs(args)
        args['log_path'] = logfile_name
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(filename)s] => %(message)s',
            handlers=[
                logging.FileHandler(filename=os.path.join(logfile_name, 'record.log')),
                logging.StreamHandler(sys.stdout)])
        
        args['log_path'] = logfile_name
        results = train_single_run(args)
        all_results[f"seed_{seed}"] = results
    aggregated = aggregate_seed_results(all_results)
    return {
        'seeds': all_results,
        'aggregate': aggregated,
    }

def train_single_run(args, return_model: bool = False):
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    print_args(args)
    
    # Initialize data manager and model
    data_manager = DataManager(
        dataset_name=args['dataset'],
        shuffle=args['shuffle'],
        seed=args['seed'],
        init_cls=args['init_cls'],
        increment=args['increment'])
    
    model = SubspaceLoRA(args)
    logging.info(f'All params: {count_parameters(model.network)}')
    logging.info(f'Trainable params: {count_parameters(model.network, True)}')
    final_results = model.loop(data_manager)
    if return_model:
        return final_results, model
    return final_results


def Bayesian_evaluate(args):
    """
    Similar to `train_single_run`, but evaluates the model every 5 tasks and returns the evaluation result.
    
    Args:
        args: Configuration arguments (same as in train_single_run)
        data_manager: DataManager object that handles datasets and task splits
    
    Yields:
        Task results after every 5 tasks for evaluation.
    """
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    device = set_device(args['device'])
    args['device'] = device

    print_args(args)

    logfile_head, logfile_name = build_log_dirs(args)
    args['log_path'] = logfile_name

    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=os.path.join(logfile_name, 'record.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


    # Initialize data manager and model
    data_manager = DataManager(
        dataset_name=args['dataset'],
        shuffle=args['shuffle'],
        seed=args['seed'],
        init_cls=args['init_cls'],
        increment=args['increment'])
    
    model = SubspaceLoRA(args)
    logging.info(f'All params: {count_parameters(model.network)}')
    logging.info(f'Trainable params: {count_parameters(model.network, True)}')

    # Initialize result storage
    task_results = {
        "original_fc": [],
        "linear_fc": []}
    
    model._eval_tasks = model._compute_eval_milestones(data_manager.nb_tasks)

    logging.info(f"Classifier refinement scheduled at tasks: {sorted(model._eval_tasks)}")

    model.data_manager = data_manager
    # Train and evaluate in tasks

    for task_id in range(data_manager.nb_tasks):
        # Incremental training
        model.incremental_train(data_manager)
        if (model._cur_task + 1) in [5, 10]:
            model.refine_classifiers()
            # logging.info(f"Evaluating after task {model._cur_task}...")
            eval_result = model.eval_task()
            # Store the evaluation results
            task_results["original_fc"].append(eval_result.original_fc)
            task_results["linear_fc"].append(eval_result.linear_fc)
            # Yield evaluation results after every 5 tasks
            logging.info(f"Evaluation after task {task_id + 1} -> Original FC: {eval_result.original_fc:.2f}% | Compensated: {eval_result.linear_fc:.2f}%")
            
            if (model._cur_task + 1) == 5:
                flag = 0
            elif (model._cur_task + 1) == 10:
                flag = 1
            yield task_results, flag

        model.after_task()
    # Return the aggregated task results after all tasks
    return task_results

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

# --------- NEW: compact float → short string ----------
def _fmt(x, *, digits=4):
    """
    压缩数值到短字符串：0.5 -> 0p5, 1e-3 -> 1e-03, 0.200 -> 0p2
    作用：减少路径长度、避免小数点过多。
    """
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    try:
        s = f"{float(x):.{digits}g}"
        s = s.replace('.', 'p')
        return s
    except Exception:
        s = str(x)
        s = s.replace('.', 'p')
        return s

from pathlib import Path
import os

from pathlib import Path
import os

import os
from pathlib import Path
import hashlib
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
    """根据 args 构建多级日志目录，确保不同 LoRA 类型的参数正确分离"""

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

    # 顶层：模型和用户信息
    base_dir = os.path.join(
        root_dir,
        f"{short(args['model_name'])}_logs_{short(args['user'])}",
        f"{short(args['dataset'])}_{short(args['vit_type'])}"
    )

    # 二级：任务设置
    task_dir = os.path.join(
        base_dir,
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

    # 五级：优化器和训练参数
    opt_params = [
        f"opt-{args['optimizer']}",
        f"lr-{short(args['lrate'])}",
        f"b-{args['batch_size']}",
        f"i-{args['iterations']}",
        f"s-{args['seed']}"
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

    # 记录生成的目录结构（用于调试）
    logging.info(f"📁 Log directory created: {abs_log_dir}")
    logging.info(f"   LoRA params: {lora_params}")
    logging.info(f"   KD params: {kd_params}")
    
    # 记录过滤信息
    original_params = set(args.keys())
    filtered_params = set(filtered_args.keys())
    removed_params = original_params - filtered_params
    if removed_params:
        logging.info(f"   过滤掉的参数: {sorted(removed_params)}")

    return os.path.dirname(abs_log_dir), str(abs_log_dir)

    
def aggregate_seed_results(seed_results):
    """Aggregate evaluation statistics from multiple random seeds."""

    if isinstance(seed_results, Mapping):
        records = list(seed_results.values())
    elif isinstance(seed_results, Sequence) and not isinstance(seed_results, (str, bytes)):
        records = list(seed_results)
    else:
        records = [seed_results]

    if not records:
        logging.warning("⚠️ No seed results provided for aggregation.")
        return {"final_task": {}, "average_across_tasks": {}}

    # Collect all variant names across all seeds
    all_variants = set()
    for res in records:
        all_variants.update(res.get("last_task_accuracies", {}).keys())
        all_variants.update(res.get("average_accuracies", {}).keys())
    all_variants = sorted(all_variants)

    # Initialize containers
    final_task_values = {variant: [] for variant in all_variants}
    avg_task_values = {variant: [] for variant in all_variants}

    # Populate with data from each seed
    for res in records:
        last_acc = res.get("last_task_accuracies", {})
        avg_acc = res.get("average_accuracies", {})

        for variant in all_variants:
            final_task_values[variant].append(last_acc.get(variant, 0.0))
            avg_task_values[variant].append(avg_acc.get(variant, 0.0))

    # Compute mean and std
    final_task_stats = {}
    avg_task_stats = {}

    if not all_variants:
        logging.warning("⚠️ No accuracy statistics found in seed results.")
        return {
            "final_task": final_task_stats,
            "average_across_tasks": avg_task_stats,
        }

    for variant in all_variants:
        f_vals = np.array(final_task_values[variant])
        a_vals = np.array(avg_task_values[variant])

        final_task_stats[variant] = (float(np.mean(f_vals)), float(np.std(f_vals)))
        avg_task_stats[variant] = (float(np.mean(a_vals)), float(np.std(a_vals)))

    # === 📊 Log Aggregated Results ===
    logging.info("📈 Aggregated Results Across Random Seeds:")
    logging.info("  ── Final Task Accuracy (Mean ± Std) ──")
    for variant in all_variants:
        mean, std = final_task_stats[variant]
        logging.info(f"      {variant:<20} : {mean:.2f}% ± {std:.2f}%")

    logging.info("  ── Average Accuracy Across Tasks (Mean ± Std) ──")
    for variant in all_variants:
        mean, std = avg_task_stats[variant]
        logging.info(f"      {variant:<20} : {mean:.2f}% ± {std:.2f}%")

    # === 🗂️ SAVE AGGREGATED RESULTS TO FILE ===
    # 保存聚合结果到JSON文件
    import time
    
    # 尝试从第一个种子的结果中获取log_path
    if isinstance(seed_results, Mapping) and len(seed_results) > 0:
        first_seed_key = list(seed_results.keys())[0]
        first_seed_result = seed_results[first_seed_key]
        
        # 查找log_path
        log_path = None
        if isinstance(first_seed_result, dict) and 'log_path' in first_seed_result:
            log_path = first_seed_result['log_path']
        elif isinstance(first_seed_result, dict) and 'per_task_results' in first_seed_result:
            # 尝试从子结构中查找
            for key, value in first_seed_result.items():
                if isinstance(value, dict) and 'log_path' in value:
                    log_path = value['log_path']
                    break
        
        if log_path:
            log_dir = Path(log_path).parent
            aggregate_file = log_dir / "aggregate_results.json"
            
            # 准备保存的数据
            save_data = {
                "final_task_stats": {k: {"mean": v[0], "std": v[1]} for k, v in final_task_stats.items()},
                "average_across_tasks_stats": {k: {"mean": v[0], "std": v[1]} for k, v in avg_task_stats.items()},
                "seed_list": list(seed_results.keys()),
                "num_seeds": len(seed_results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "variants": all_variants
            }
            
            with open(aggregate_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"💾 Aggregated results saved to: {aggregate_file}")
        else:
            logging.warning("⚠️ Could not find log_path for saving aggregated results.")
    else:
        logging.warning("⚠️ No seed results available for saving aggregated results.")

    # Return structured stats
    return {
        "final_task": final_task_stats,
        "average_across_tasks": avg_task_stats,
    }
