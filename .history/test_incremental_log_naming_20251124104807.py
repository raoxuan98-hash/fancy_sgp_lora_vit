#!/usr/bin/env python3
"""
测试增量拆分参数在日志命名中的体现
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

from trainer import build_log_dirs

def test_incremental_log_naming():
    """测试增量拆分参数在日志命名中的体现"""
    
    print("🧪 测试增量拆分参数在日志命名中的体现")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        {
            "name": "启用增量拆分，2个拆分",
            "args": {
                'model_name': 'sldc',
                'user': 'test_user', 
                'cross_domain': True,
                'cross_domain_datasets': ['cifar100_224', 'imagenet-r'],
                'vit_type': 'vit-b-p16',
                'num_shots': 64,
                'lora_rank': 4,
                'lora_type': 'sgp_lora',
                'weight_temp': 2.0,
                'weight_kind': 'log1p',
                'weight_p': 1.0,
                'optimizer': 'adamw',
                'lrate': 1e-4,
                'batch_size': 16,
                'iterations': 2000,
                'seed': 1993,
                'enable_incremental_split': True,
                'num_incremental_splits': 2,
                'incremental_split_seed': 42
            }
        },
        {
            "name": "启用增量拆分，5个拆分",
            "args": {
                'model_name': 'sldc',
                'user': 'test_user',
                'cross_domain': True, 
                'cross_domain_datasets': ['cifar100_224', 'imagenet-r'],
                'vit_type': 'vit-b-p16',
                'num_shots': 64,
                'lora_rank': 4,
                'lora_type': 'sgp_lora',
                'weight_temp': 2.0,
                'weight_kind': 'log1p',
                'weight_p': 1.0,
                'optimizer': 'adamw',
                'lrate': 1e-4,
                'batch_size': 16,
                'iterations': 2000,
                'seed': 1993,
                'enable_incremental_split': True,
                'num_incremental_splits': 5,
                'incremental_split_seed': 42
            }
        },
        {
            "name": "禁用增量拆分",
            "args": {
                'model_name': 'sldc',
                'user': 'test_user',
                'cross_domain': True,
                'cross_domain_datasets': ['cifar100_224', 'imagenet-r'], 
                'vit_type': 'vit-b-p16',
                'num_shots': 64,
                'lora_rank': 4,
                'lora_type': 'sgp_lora',
                'weight_temp': 2.0,
                'weight_kind': 'log1p',
                'weight_p': 1.0,
                'optimizer': 'adamw',
                'lrate': 1e-4,
                'batch_size': 16,
                'iterations': 2000,
                'seed': 1993,
                'enable_incremental_split': False,
                'num_incremental_splits': 2,
                'incremental_split_seed': 42
            }
        },
        {
            "name": "Within-domain实验启用增量拆分",
            "args": {
                'model_name': 'sldc',
                'user': 'test_user',
                'cross_domain': False,
                'dataset': 'cifar100_224',
                'vit_type': 'vit-b-p16',
                'init_cls': 10,
                'increment': 10,
                'lora_rank': 4,
                'lora_type': 'basic_lora',
                'optimizer': 'adamw',
                'lrate': 1e-4,
                'batch_size': 16,
                'iterations': 2000,
                'seed': 1993,
                'enable_incremental_split': True,
                'num_incremental_splits': 3,
                'incremental_split_seed': 42
            }
        }
    ]
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 测试用例 {i}: {test_case['name']}")
            print("-" * 50)
            
            try:
                # 调用 build_log_dirs 函数
                logfile_head, logfile_name = build_log_dirs(test_case['args'], root_dir=temp_dir)
                
                # 提取目录路径中的关键信息
                log_path = Path(logfile_name)
                print(f"📁 日志路径: {logfile_name}")
                
                # 检查路径中是否包含增量拆分参数
                log_parts = logfile_name.split('/')
                log_content = '/'.join(log_parts)  # 用于字符串搜索
                
                # 验证增量拆分参数是否包含在路径中
                has_incremental_params = False
                if test_case['args']['enable_incremental_split']:
                    if 'inc_split-enabled' in log_content and f"splits-{test_case['args']['num_incremental_splits']}" in log_content:
                        has_incremental_params = True
                        print("✅ 正确包含增量拆分参数: inc_split-enabled")
                        print(f"✅ 正确包含拆分数量: splits-{test_case['args']['num_incremental_splits']}")
                    else:
                        print("❌ 缺少增量拆分参数")
                else:
                    if 'inc_split-disabled' in log_content:
                        has_incremental_params = True
                        print("✅ 正确包含禁用增量拆分的标识: inc_split-disabled")
                    else:
                        print("❌ 缺少禁用增量拆分的标识")
                
                if has_incremental_params:
                    print(f"🎯 测试用例 {i} 通过")
                else:
                    print(f"❌ 测试用例 {i} 失败")
                    
            except Exception as e:
                print(f"❌ 测试用例 {i} 执行出错: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")

if __name__ == "__main__":
    test_incremental_log_naming()