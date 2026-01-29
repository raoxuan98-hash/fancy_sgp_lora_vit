#!/usr/bin/env python3
"""
测试脚本：验证多个随机种子的结果聚合是否正常工作
"""

import os
import sys
import json
import shutil
from pathlib import Path

def test_aggregation():
    """测试聚合逻辑"""
    
    # 模拟参数
    test_args = {
        'model_name': 'sldc',
        'user': 'test_user',
        'dataset': 'cifar100_224',
        'vit_type': 'vit-b-p16-mocov3',
        'init_cls': 10,
        'increment': 10,
        'lora_rank': 4,
        'lora_type': 'sgp_lora',
        'weight_temp': 2.0,
        'weight_kind': 'log1p',
        'weight_p': 1.0,
        'optimizer': 'adamw',
        'lrate': 0.0001,
        'batch_size': 16,
        'iterations': 50,  # 减少迭代次数以加快测试
        'gamma_kd': 0.0,
        'seed_list': [1993, 1996],  # 只用两个种子进行测试
        'smart_defaults': False,
        'shuffle': True,
        'memory_size': 0,
        'memory_per_class': 0,
        'fixed_memory': False,
        'warmup_ratio': 0.1,
        'ca_epochs': 5,
        'evaluate_final_only': True,
        'update_teacher_each_task': True,
        'use_aux_for_kd': False,
        'kd_type': 'feat',
        'distillation_transform': 'linear',
        'eval_only': False,
        'lda_reg_alpha': 0.10,
        'qda_reg_alpha1': 0.20,
        'qda_reg_alpha2': 0.90,
        'qda_reg_alpha3': 0.20,
        'auxiliary_data_path': '/data1/open_datasets',
        'aux_dataset': 'imagenet',
        'auxiliary_data_size': 1024,
        'l2_protection': False,
        'l2_protection_lambda': 1e-4,
        'weight_decay': 3e-5,
        'head_scale': 1.0
    }
    
    # 清理之前的测试结果
    test_log_dir = Path("sldc_logs_test_user")
    if test_log_dir.exists():
        shutil.rmtree(test_log_dir)
    
    print("🧪 开始测试多种子结果聚合...")
    print(f"📋 测试参数: {test_args['lora_type']} on {test_args['dataset']}")
    print(f"🌱 测试种子: {test_args['seed_list']}")
    print("-" * 80)
    
    # 导入并运行训练
    try:
        from trainer import train
        results = train(test_args)
        
        # 检查结果结构
        assert 'seeds' in results, "结果中缺少'seeds'键"
        assert 'aggregate' in results, "结果中缺少'aggregate'键"
        
        # 检查种子结果
        seeds = results['seeds']
        assert len(seeds) == len(test_args['seed_list']), f"种子数量不匹配: {len(seeds)} vs {len(test_args['seed_list'])}"
        
        for seed_key in test_args['seed_list']:
            seed_key_str = f"seed_{seed_key}"
            assert seed_key_str in seeds, f"缺少种子{seed_key}的结果"
        
        # 检查聚合结果
        aggregate = results['aggregate']
        assert 'final_task' in aggregate, "聚合结果中缺少'final_task'"
        assert 'average_across_tasks' in aggregate, "聚合结果中缺少'average_across_tasks'"
        
        # 检查聚合结果文件是否存在
        shared_log_dir = None
        for seed_result in seeds.values():
            if 'shared_log_dir' in seed_result:
                shared_log_dir = Path(seed_result['shared_log_dir'])
                break
        
        assert shared_log_dir is not None, "找不到共享日志目录"
        assert shared_log_dir.exists(), "共享日志目录不存在"
        
        aggregate_file = shared_log_dir / "aggregate_results.json"
        assert aggregate_file.exists(), "聚合结果文件不存在"
        
        # 检查聚合结果文件内容
        with open(aggregate_file, 'r', encoding='utf-8') as f:
            aggregate_data = json.load(f)
        
        assert 'final_task_stats' in aggregate_data, "聚合文件中缺少'final_task_stats'"
        assert 'average_across_tasks_stats' in aggregate_data, "聚合文件中缺少'average_across_tasks_stats'"
        assert 'seed_list' in aggregate_data, "聚合文件中缺少'seed_list'"
        assert 'num_seeds' in aggregate_data, "聚合文件中缺少'num_seeds'"
        
        # 检查种子列表
        seed_list = aggregate_data['seed_list']
        assert len(seed_list) == len(test_args['seed_list']), "聚合文件中的种子数量不匹配"
        
        # 检查标准差是否为0（如果是0，说明没有正确聚合多个种子）
        for variant, stats in aggregate_data['final_task_stats'].items():
            std = stats['std']
            if std == 0.0:
                print(f"⚠️ 警告: 变体{variant}的标准差为0，可能没有正确聚合多个种子")
            else:
                print(f"✅ 变体{variant}的标准差为{std:.2f}，聚合正常")
        
        print("\n🎉 测试通过！多种子结果聚合工作正常。")
        print(f"📁 聚合结果保存在: {aggregate_file}")
        print(f"🌱 包含种子: {seed_list}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_aggregation()
    sys.exit(0 if success else 1)