"""
分类器消融实验配置文件
"""

# 实验参数配置
EXPERIMENT_CONFIG = {
    # 数据配置
    'num_shots': 128,
    'model_name': "vit-b-p16-clip",
    'cross_domain_datasets': [
        'cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 
        'caltech-101', 'dtd', 'fgvc-aircraft-2013b-variants102', 
        'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'
    ],
    
    # 网络配置
    'adapt_backbone': True,
    'iterations': 0,
    
    # 训练配置
    'batch_size': {
        'adapt': 24,
        'feature_extraction': 64
    },
    'learning_rate': {
        'vit': 1e-5,
        'classifier': 1e-3
    },
    'ema_beta': 0.90,
    
    # 特征缓存配置
    'cache_dir': "cached_data/classifier_ablation",
    
    # 实验1: 性能曲面等高线图
    'exp1': {
        'alpha1_range': (0, 5.0),
        'alpha2_range': (0, 5.0),
        'alpha1_points': 11,
        'alpha2_points': 11,
        'alpha3_fixed': 0.5
    },
    
    # 实验2: 参数敏感性分析
    'exp2': {
        'alpha1_range': (0.0, 5.0),
        'alpha2_range': (0.0, 5.0),
        'alpha_sum': 3.0,
        'fixed_points': 11
    },
    
    # 实验3: 分类器对比
    'exp3': {
        'sgd_epochs': 5,
        'sgd_lr': 0.01,
        'rgda_params': [
            (0.5, 0.5, "RGDA(0.5,0.5)"),
            (1.0, 2.0, "RGDA(1.0,2.0)"),
            (2.0, 3.0, "RGDA(2.0,3.0)")
        ]
    },
    
    # 实验4: 效率对比
    'exp4': {
        'rgda_alpha1': 1.0,
        'rgda_alpha2': 2.0,
        'lda_reg_alpha': 0.3,
        'sgd_epochs': 5,
        'sgd_lr': 0.01
    },
    
    # 输出配置
    'output_dir': "实验结果保存/分类器消融实验",
    
    # 快速测试配置
    'quick_test': {
        'enabled': True,
        'exp1_points': 5,
        'exp2_points': 5,
        'sgd_epochs': 2
    }
}