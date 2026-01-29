# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import argparse
from trainer import train

def set_smart_defaults(ns):
    """为cross-domain实验设置智能默认值"""
    if not ns.smart_defaults:
        return ns
    
    if ns.lora_type == 'full':
        ns.lrate = 1e-3
        ns.optimizer = 'sgd'
        ns.head_scale = 1.0

    if ns.test:
        ns.seed_list = [1993]
        ns.iterations = 100

    return ns


def main(args):
    """cross-domain实验主函数"""
    # 设置cross_domain为True，确保使用cross-domain数据管理器
    args['cross_domain'] = True
    results = train(args)

def build_parser() -> argparse.ArgumentParser:
    """构建cross-domain实验的参数解析器"""
    parser = argparse.ArgumentParser(description='Cross-Domain Incremental Learning Experiments')
    
    # 基本参数
    basic = parser.add_argument_group('basic', 'General / high‑level options')
    basic.add_argument('--smart_defaults', action='store_true', default=False, 
                      help='If set, overwrite a few hyper‑parameters according to the dataset.')
    basic.add_argument('--user', type=str, default='2025-11-29-test', choices=['authors'], 
                      help='User identifier (currently unused).')
    basic.add_argument('--test', action='store_true', default=False, 
                      help='If set, run a quick test with reduced settings.')
    
    # 跨域数据集参数
    cd = parser.add_argument_group('cross_domain', 'Cross-domain experiment settings')
    cd.add_argument('--cross_domain_datasets', type=str, nargs='+', 
                   default=['cifar100_224', 'imagenet-r', 'cars196_224', 'cub200_224', 'caltech-101', 'oxford-flower-102', 'food-101'], 
                   help='List of datasets for cross-domain experiments')
    cd.add_argument('--num_shots', type=int, default=64, 
                   help='Number of samples per class for few-shot learning. If > 0, randomly sample num_shots samples per class.')
    
    # 增量拆分参数
    inc = parser.add_argument_group('incremental', 'Incremental split settings for cross-domain datasets')
    inc.add_argument('--enable_incremental_split', action="store_true", default=False,
                     help='Enable incremental split for cross-domain datasets. If set, each dataset will be split into multiple incremental subsets.')
    inc.add_argument('--num_incremental_splits', type=int, default=2,
                     help='Number of incremental splits per dataset when enable_incremental_split is True.')
    inc.add_argument('--incremental_split_seed', type=int, default=42,
                     help='Random seed for incremental split to ensure reproducibility.')
    
    # 内存/回放缓冲区参数
    mem = parser.add_argument_group('memory', 'Memory / replay buffer')
    mem.add_argument('--memory_size', type=int, default=0, help='Total memory budget.')
    mem.add_argument('--memory_per_class', type=int, default=0, help='Memory allocated per class.')
    mem.add_argument('--fixed_memory', action="store_true", default=False, 
                   help='If set, memory size does not grow with new classes.')
    mem.add_argument('--shuffle', action='store_true', default=True, 
                   help='Shuffle replay buffer before each epoch.')

    # 模型参数
    model = parser.add_argument_group('model', 'Backbone & LoRA settings')
    model.add_argument('--model_name', type=str, default='sldc', help='Model identifier.')
    model.add_argument('--vit_type', type=str, default='vit-b-p16', 
                      choices=['vit-b-p16', 'vit-b-p16-dino', 'vit-b-p16-mae', 'vit-b-p16-clip', 'vit-b-p16-mocov3'], 
                      help='ViT backbone variant.')
    model.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')
    model.add_argument('--lora_rank', type=int, default=4, help='LoRA rank.')
    model.add_argument('--lora_type', type=str, default="basic_lora", 
                      choices=['basic_lora', 'sgp_lora', 'nsp_lora', 'full'], 
                      help='Type of LoRA adaptor.')
    model.add_argument('--weight_temp', type=float, default=2.0, help='Projection temperature.')
    model.add_argument('--weight_kind', type=str, default='log1p', 
                      choices=["exp", "log1p", "rational1", "rational2", "sqrt_rational2", "power_family", "stretchqed_exp"])
    model.add_argument('--weight_p', type=float, default=1.0, help='Weight p.')
    model.add_argument('--nsp_eps', type=float, default=0.05, choices=[0.05, 0.10])
    model.add_argument('--nsp_weight', type=float, default=0.0, choices=[0.0, 0.02, 0.05])

    # 训练参数
    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')  
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993, 199], 
                          help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=10, help='Training iterations per task.')
    train_grp.add_argument('--warmup_ratio', type=int, default=0.10, help='Warm‑up ratio for learning rate schedule.')
    train_grp.add_argument('--ca_epochs', type=int, default=5, help='Classifier alignment epochs.')
    train_grp.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (adamw / sgd).')
    train_grp.add_argument('--lrate', type=float, default=1e-4, help='Learning rate.')
    train_grp.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    train_grp.add_argument('--evaluate_final_only', action="store_true", default=True)
    train_grp.add_argument('--gamma_kd', type=float, default=0.0, help='Knowledge‑distillation weight.')
    train_grp.add_argument('--update_teacher_each_task', action="store_true", default=True, 
                          help='If set, update the teacher network after each task.')
    train_grp.add_argument('--use_aux_for_kd', action='store_true', default=False, 
                          help='If set, use auxiliary data for KD.')
    train_grp.add_argument('--kd_type', type=str, default='cos', help='KD type (feat / cos).')
    train_grp.add_argument('--distillation_transform', type=str, default='linear', 
                          help='Distillation head transform (identity / linear / weaknonlinear).')
    train_grp.add_argument('--eval_only', action='store_true', default=False)

    # GDA参数
    gda = parser.add_argument_group('gda', 'Gaussian discriminate analysis settings')
    gda.add_argument('--lda_reg_alpha', type=float, default=0.10, help='LDA regularisation alpha.')
    gda.add_argument('--qda_reg_alpha1', type=float, default=0.20, help='QDA regularisation alpha 1.')
    gda.add_argument('--qda_reg_alpha2', type=float, default=2.00, help='QDA regularisation alpha 2.')
    gda.add_argument('--qda_reg_alpha3', type=float, default=0.50, help='QDA regularisation alpha 3.')
    
    # 辅助数据集参数
    aux = parser.add_argument_group('auxiliary', 'External / auxiliary dataset')
    aux.add_argument('--auxiliary_data_path', type=str, default='/data1/open_datasets', 
                   help='Root path of the auxiliary dataset.')
    aux.add_argument('--aux_dataset', type=str, default='imagenet', 
                   help='Dataset type for auxiliary data (e.g. imagenet, cifar).', 
                   choices=['imagenet', 'flickr8k'])
    aux.add_argument('--auxiliary_data_size', type=int, default=2048, 
                   help='Number of samples drawn from the auxiliary dataset each epoch.')
    aux.add_argument('--feature_combination_type', type=str, default="combined", 
                   choices=['combined', 'aux_only', 'current_only'], 
                   help='Type of feature combination.')

    # 补偿器参数
    comp = parser.add_argument_group('compensator', 'Distribution compensator settings')
    comp.add_argument('--compensator_types', type=str, nargs='+', 
                     default=['SeqFT', 'SeqFT + weaknonlinear', 'SeqFT + Hopfield'],
                     choices=['SeqFT', 'SeqFT + linear', 'SeqFT + weaknonlinear', 'SeqFT + Hopfield', 'SeqFT + rff'],
                     help='Types of compensators to use. Default is all four types.')
    comp.add_argument('--hopfield_temp', type=float, default=0.1, 
                    help='Temperature parameter for Hopfield attention compensator.')
    comp.add_argument('--hopfield_topk', type=int, default=500, 
                    help='Top-k parameter for Hopfield attention compensator.')
    
    return parser

# 主程序入口
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args = set_smart_defaults(args)
    args = vars(args)
    main(args)