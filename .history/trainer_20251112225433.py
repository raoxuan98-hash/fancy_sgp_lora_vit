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
    

def train_single_run(args, return_model: bool = False):
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    print_args(args)
    
    # Initialize data manager and model

    if args['cross_domain']:
        data_manager = CrossDomainDataManagerCore(
            dataset_names=args['dataset'],
            shuffle=args['shuffle'],
            seed=args['seed'])
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