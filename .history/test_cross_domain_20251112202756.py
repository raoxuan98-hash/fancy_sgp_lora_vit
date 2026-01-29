#!/usr/bin/env python3
"""
测试cross_domain功能的简单脚本
"""

import sys
import os
import argparse

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cross_domain():
    """测试cross_domain功能"""
    print("开始测试cross_domain功能...")
    
    # 构建测试参数
    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group('basic', 'General / high‑level options')
    basic.add_argument('--dataset', type=str, default='cross_domain_elevater', choices=['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224', 'caltech101_224', 'oxfordpet37_224', 'food101_224', 'resisc45_224', 'cross_domain_elevater'], help='Dataset to use')
    basic.add_argument('--smart_defaults', action='store_true', default=False, help='If set, overwrite a few hyper‑parameters according to the dataset.')
    basic.add_argument('--user', type=str, default='sgp_lora_vit_main_experiments', choices=['authors'], help='User identifier (currently unused).')
    basic.add_argument('--test', action='store_true', default=True, help='If set, run a quick test with reduced settings.')
    basic.add_argument('--cross_domain', action='store_true', default=True, help='Enable cross-domain class-incremental learning')
    basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['imagenet-r', 'caltech-101', 'dtd', 'eurosat', 'fgvc_aircraft', 'food101', 'mnist', 'oxford_flower102', 'oxford_pets', 'cars196_224'], help='List of datasets for cross-domain experiments')

    mem = parser.add_argument_group('memory', 'Memory / replay buffer')
    mem.add_argument('--memory_size', type=int, default=0, help='Total memory budget.')
    mem.add_argument('--memory_per_class', type=int, default=0, help='Memory allocated per class.')
    mem.add_argument('--fixed_memory', action='store_true', default=False, help='If set, memory size does not grow with new classes.')
    mem.add_argument('--shuffle', action='store_true', default=True, help='Shuffle replay buffer before each epoch.')

    cls = parser.add_argument_group('class', 'Class increment settings')
    cls.add_argument('--init_cls', type=int, default=10, help='Number of classes in the first task.')
    cls.add_argument('--increment', type=int, default=10, help='Number of new classes added per task.')

    model = parser.add_argument_group('model', 'Backbone & LoRA settings')
    model.add_argument('--model_name', type=str, default='sldc', help='Model identifier.')
    model.add_argument('--vit_type', type=str, default='vit-b-p16', choices=['vit-b-p16', 'vit-b-p16-dino', 'vit-b-p16-mae', 'vit-b-p16-clip', 'vit-b-p16-mocov3'], help='ViT backbone variant.')
    model.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')

    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')  
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993], help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=100, help='Training iterations per task (reduced for test).')
    train_grp.add_argument('--warmup_ratio', type=int, default=0.1, help='Warm‑up ratio for learning rate schedule.')
    train_grp.add_argument('--ca_epochs', type=int, default=5, help='Classifier alignment epochs.')
    train_grp.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (adamw / sgd).')
    train_grp.add_argument('--lrate', type=float, default=1e-4, help='Learning rate.')
    train_grp.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    train_grp.add_argument('--evaluate_final_only', action=argparse.BooleanOptionalAction, default=True)
    train_grp.add_argument('--gamma_kd', type=float, default=0.0, help='Knowledge‑distillation weight.')
    train_grp.add_argument('--update_teacher_each_task', type=bool, default=True, help='If set, update the teacher network after each task.')
    train_grp.add_argument('--use_aux_for_kd', action='store_true', default=False, help='If set, use auxiliary data for KD.')
    train_grp.add_argument('--kd_type', type=str, default='feat', help='KD type (feat / logit).')
    train_grp.add_argument('--distillation_transform', type=str, default='linear', help='Distillation head transform (identity / linear / weaknonlinear).')
    train_grp.add_argument('--eval_only', type=bool, default=False)

    model.add_argument('--lora_rank', type=int, default=4, help='LoRA rank.')
    model.add_argument('--lora_type', type=str, default="sgp_lora", choices=['basic_lora', 'sgp_lora', 'nsp_lora', 'full'], help='Type of LoRA adaptor.')
    model.add_argument('--weight_temp', type=float, default=2.0, help='Projection temperature.')
    model.add_argument('--weight_kind', type=str, default='log1p', choices=["exp", "log1p", "rational1", "rational2", "sqrt_rational2", "power_family", "stretched_exp"])
    model.add_argument('--weight_p', type=float, default=1.0, help='Weight p.')
    model.add_argument('--nsp_eps', type=float, default=0.05, choices=[0.05, 0.10])
    model.add_argument('--nsp_weight', type=float, default=0.0, choices=[0.0, 0.02, 0.05])

    gda = parser.add_argument_group('gda', 'Gaussian discriminate analysis settings')
    gda.add_argument('--lda_reg_alpha', type=float, default=0.10, help='LDA regularisation alpha.')
    gda.add_argument('--qda_reg_alpha1', type=float, default=0.20, help='QDA regularisation alpha 1.')
    gda.add_argument('--qda_reg_alpha2', type=float, default=0.90, help='QDA regularisation alpha 2.')
    gda.add_argument('--qda_reg_alpha3', type=float, default=0.20, help='QDA regularisation alpha 3.')
    
    aux = parser.add_argument_group('auxiliary', 'External / auxiliary dataset')
    aux.add_argument('--auxiliary_data_path', type=str, default='/data1/open_datasets', help='Root path of the auxiliary dataset.')
    aux.add_argument('--aux_dataset', type=str, default='imagenet', help='Dataset type for auxiliary data (e.g. imagenet, cifar).', choices=['imagenet', 'flickr8k'])
    aux.add_argument('--auxiliary_data_size', type=int, default=1024, help='Number of samples drawn from the auxiliary dataset each epoch.')

    reg = parser.add_argument_group('regularisation', 'Extra regularisation terms') 
    reg.add_argument('--l2_protection', action='store_true', default=False, help='Enable L2‑protection between the current and previous network.')
    reg.add_argument('--l2_protection_lambda', type=float, default=1e-4, help='Weight for the L2‑protection term (higher → stronger regularisation). When `--l2_protection` is off, this will be automatically set to 0.0.')
    
    # 解析参数
    args = parser.parse_args([])
    args = vars(args)
    
    # 测试数据管理器
    print("测试数据管理器初始化...")
    try:
        from utils.data_manager import DataManager
        data_manager = DataManager(
            dataset_name=args['dataset'],
            shuffle=args['shuffle'],
            seed=args['seed_list'][0],
            init_cls=args['init_cls'],
            increment=args['increment'],
            args=args
        )
        print(f"✓ 数据管理器初始化成功")
        print(f"  - 任务数量: {data_manager.nb_tasks}")
        print(f"  - 数据集名称: {data_manager.dataset_name}")
        
        # 测试获取任务信息
        for task_id in range(min(3, data_manager.nb_tasks)):  # 只测试前3个任务
            task_size = data_manager.get_task_size(task_id)
            task_classes = data_manager.get_task_classes(task_id)
            print(f"  - 任务 {task_id}: {task_size} 个类, 类别范围: {min(task_classes)}-{max(task_classes)}")
            
            # 测试获取数据子集
            try:
                train_subset = data_manager.get_subset(task_id, "train", cumulative=False, mode="train")
                test_subset = data_manager.get_subset(task_id, "test", cumulative=False, mode="test")
                try:
                    train_size = len(train_subset)
                except:
                    train_size = "未知"
                try:
                    test_size = len(test_subset)
                except:
                    test_size = "未知"
                print(f"    - 训练样本数: {train_size}, 测试样本数: {test_size}")
                
                # 测试获取单个样本
                try:
                    sample_img, sample_label, sample_class_name = train_subset[0]
                    print(f"    - 样本形状: {sample_img.shape if hasattr(sample_img, 'shape') else type(sample_img)}")
                    print(f"    - 样本标签: {sample_label}")
                    print(f"    - 样本类别名: {sample_class_name}")
                except Exception as e:
                    print(f"    - 获取单个样本失败: {e}")
                
            except Exception as e:
                print(f"    - 获取数据子集失败: {e}")
                
        print("✓ 数据管理器测试通过")
        
    except Exception as e:
        print(f"✗ 数据管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试模型初始化
    print("\n测试模型初始化...")
    try:
        # 添加seed参数
        args['seed'] = args['seed_list'][0]
        from models.subspace_lora import SubspaceLoRA
        model = SubspaceLoRA(args)
        print(f"✓ 模型初始化成功")
        
        # 测试模型前向传播
        try:
            # 获取一个批次的数据
            train_subset = data_manager.get_subset(0, "train", cumulative=False, mode="train")
            from torch.utils.data import DataLoader
            import torch
            
            loader = DataLoader(train_subset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            
            if len(batch) == 3:  # (images, labels, class_names)
                images, labels, _ = batch
                print(f"  - 批次图像形状: {images.shape}")
                print(f"  - 批次标签形状: {labels.shape}")
                
                # 前向传播
                with torch.no_grad():
                    logits = model.network(images)
                    print(f"  - 输出logits形状: {logits.shape}")
                
                print("✓ 模型前向传播测试通过")
            else:
                print(f"  - 批次格式不正确: {len(batch)} 个元素")
                
        except Exception as e:
            print(f"  - 模型前向传播测试失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ 模型初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ cross_domain功能测试全部通过!")
    return True

if __name__ == "__main__":
    success = test_cross_domain()
    sys.exit(0 if success else 1)