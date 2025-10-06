import os
import sys
import logging
import torch
import random
import numpy as np
from models.subspace_lora import SubspaceLoRA
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


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
    aggregate_seed_results(all_results)
    return all_results

def train_single_run(args):
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

def build_log_dirs(args: dict, root_dir="."):
    """根据 args 构建多级日志目录并逐级创建，避免 WinError 206"""

    def short(s: str, maxlen=20):
        """截断并加hash避免重复"""
        s = str(s)
        if len(s) <= maxlen:
            return s
        h = hashlib.md5(s.encode()).hexdigest()[:6]
        return s[:maxlen] + "_" + h

    # 顶层
    base_dir = os.path.join(
        root_dir,
        f"{short(args['model_name'])}_logs_{short(args['user'])}",
        f"{short(args['dataset'])}_{short(args['vit_type'])}"
    )

    # 二级
    task_dir = os.path.join(
        base_dir,
        f"init-{args['init_cls']}_inc-{args['increment']}",
        f"lrank-{args.get('lora_rank','NA')}_ltype-{short(args.get('lora_type','NA'))}"
    )

    # 三级参数（合并成一个字符串再hash）
    lora_params = []
    if 'weight_temp' in args:
        lora_params.append(f"t={args['weight_temp']}")
    if args.get('lora_type') == 'sgp_lora':
        lora_params.append(f"k={args.get('weight_kind')}")
        lora_params.append(f"p={args.get('weight_p')}")
    if args.get('lora_type') == 'nsp_lora':
        lora_params.append(f"eps={args.get('nsp_eps')}")
        lora_params.append(f"w={args.get('nsp_weight')}")
    lora_hash = ""
    if lora_params:
        param_str = "_".join(map(str, lora_params))
        lora_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    lora_dir = os.path.join(task_dir, f"lora-{lora_hash}") if lora_hash else task_dir

    # 四级参数（同样hash）
    other_params = []
    for k in ['compensate', 'kd_type', 'gamma_kd', 'gamma_norm']:
        if k in args:
            other_params.append(f"{k}={args[k]}")
    other_hash = ""
    if other_params:
        other_str = "_".join(map(str, other_params))
        other_hash = hashlib.md5(other_str.encode()).hexdigest()[:8]
    other_dir = os.path.join(lora_dir, f"other-{other_hash}") if other_hash else lora_dir

    # 五级优化器层
    opt_str = f"opt-{args['optimizer']}_lr-{args['lrate']}_b-{args['batch_size']}_i-{args['iterations']}_s-{args['seed']}"
    opt_dir = os.path.join(other_dir, short(opt_str, maxlen=40))

    # === 逐级创建 ===
    abs_log_dir = os.path.abspath(opt_dir)
    current = Path(abs_log_dir).root
    for part in Path(abs_log_dir).parts[1:]:
        current = Path(current) / part
        current.mkdir(exist_ok=True)

    # 保存原始参数到 JSON（防止hash丢信息）
    params_json = Path(abs_log_dir) / "params.json"
    if not params_json.exists():
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=False, indent=2)

    return os.path.dirname(abs_log_dir), str(abs_log_dir)

    
def aggregate_seed_results(seed_results):
    if not seed_results:
        logging.warning("⚠️ No seed results provided for aggregation.")
        return {"final_task": {}, "average_across_tasks": {}}

    # Collect all variant names across all seeds
    all_variants = set()
    for res in seed_results:
        all_variants.update(res.get("last_task_accuracies", {}).keys())
        all_variants.update(res.get("average_accuracies", {}).keys())
    all_variants = sorted(all_variants)

    # Initialize containers
    final_task_values = {variant: [] for variant in all_variants}
    avg_task_values = {variant: [] for variant in all_variants}

    # Populate with data from each seed
    for res in seed_results:
        last_acc = res.get("last_task_accuracies", {})
        avg_acc = res.get("average_accuracies", {})

        for variant in all_variants:
            final_task_values[variant].append(last_acc.get(variant, 0.0))
            avg_task_values[variant].append(avg_acc.get(variant, 0.0))

    # Compute mean and std
    final_task_stats = {}
    avg_task_stats = {}

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

    # Return structured stats
    return {
        "final_task": final_task_stats,
        "average_across_tasks": avg_task_stats}