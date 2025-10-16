import argparse
import os
from trainer import train

def set_smart_defaults(ns):
    if not ns.smart_defaults:
        return ns
    if ns.dataset == 'cars196_224':
        ns.init_cls, ns.increment, ns.iterations = 5, 5, 1000
    elif ns.dataset == 'imagenet-r':
        ns.init_cls, ns.increment, ns.iterations = 5, 5, 1500
    elif ns.dataset == 'cifar100_224':
        ns.init_cls, ns.increment, ns.iterations = 2, 3, 1500
    elif ns.dataset == 'cub200_224':
        ns.init_cls, ns.increment, ns.iterations = 5, 5, 1000

    if ns.lora_type == 'full':
        ns.lrate = 1e-3
        ns.optimizer = 'sgd'
        ns.head_scale = 1.0

    if ns.test:
        ns.seed_list = [1993] 

    return ns

# def set_smart_defaults(ns):
#     if not ns.smart_defaults:
#         return ns
#     if ns.dataset == 'cars196_224':
#         ns.init_cls, ns.increment, ns.iterations = 20, 4, 1000
#     elif ns.dataset == 'imagenet-r':
#         ns.init_cls, ns.increment, ns.iterations = 20, 4, 1500
#     elif ns.dataset == 'cifar100_224':
#         ns.init_cls, ns.increment, ns.iterations = 10, 2, 1000
#     elif ns.dataset == 'cub200_224':
#         ns.init_cls, ns.increment, ns.iterations = 20, 4, 1000

#     if ns.lora_type == 'full':
#         ns.lrate = 1e-3
#         ns.optimizer = 'sgd'
#         ns.head_scale = 1.0

#     if ns.test:
#         ns.seed_list = [1993] 

#     return ns

def main(args):
    train(args)

def build_parser() -> argparse.ArgumentParser:

    # import os 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group('basic', 'General / high‑level options')
    basic.add_argument('--dataset', type=str, default='cub200_224', choices=['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224', 'caltech101_224', 'oxfordpet37_224', 'food101_224', 'resisc45_224'], help='Dataset to use')
    basic.add_argument('--smart_defaults', action='store_true', default=False, help='If set, overwrite a few hyper‑parameters according to the dataset.')
    basic.add_argument('--user', type=str, default='rebuttal_40tasks', choices=['authors'], help='User identifier (currently unused).')
    basic.add_argument('--test', action='store_true', default=True, help='If set, run a quick test with reduced settings.')

    mem = parser.add_argument_group('memory', 'Memory / replay buffer')
    mem.add_argument('--memory_size', type=int, default=0, help='Total memory budget.')
    mem.add_argument('--memory_per_class', type=int, default=0, help='Memory allocated per class.')
    mem.add_argument('--fixed_memory', action='store_true', default=False, help='If set, memory size does not grow with new classes.')
    mem.add_argument('--shuffle', action='store_true', default=True, help='Shuffle replay buffer before each epoch.')

    cls = parser.add_argument_group('class', 'Class increment settings')
    cls.add_argument('--init_cls', type=int, default=20, help='Number of classes in the first task.')
    cls.add_argument('--increment', type=int, default=20, help='Number of new classes added per task.')

    model = parser.add_argument_group('model', 'Backbone & LoRA settings')
    model.add_argument('--model_name', type=str, default='sldc', help='Model identifier.')
    model.add_argument('--vit_type', type=str, default='vit-b-p16-mocov3', choices=['vit-b-p16', 'vit-b-p16-dino', 'vit-b-p16-mae', 'vit-b-p16-clip', 'vit-b-p16-mocov3'], help='ViT backbone variant.')
    model.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')

    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')  
    train_grp.add_argument('--sce_a', type=float, default=0.5, help='Symmetric cross‑entropy A.')
    train_grp.add_argument('--sce_b', type=float, default=0.5, help='Symmetric cross‑entropy B.')
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993], help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=50, help='Training iterations per task.')
    train_grp.add_argument('--warmup_ratio', type=int, default=0.1, help='Warm‑up ratio for learning rate schedule.')
    train_grp.add_argument('--ca_epochs', type=int, default=5, help='Classifier alignment epochs.')
    train_grp.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (adamw / sgd).')
    train_grp.add_argument('--lrate', type=float, default=1e-4, help='Learning rate.')
    train_grp.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    train_grp.add_argument('--evaluate_final_only', action=argparse.BooleanOptionalAction, default=True)
    train_grp.add_argument('--gamma_norm', type=float, default=0.1, help='Norm regularisation weight.')
    train_grp.add_argument('--gamma_kd', type=float, default=0.5, help='Knowledge‑distillation weight.')
    train_grp.add_argument('--update_teacher_each_task', type=bool, default=False, help='If set, update the teacher network after each task.')
    train_grp.add_argument('--use_aux_for_kd', action='store_true', default=False, help='If set, use auxiliary data for KD.')
    train_grp.add_argument('--kd_type', type=str, default='feat', help='KD type (feat / logit).')
    train_grp.add_argument('--compensate', type=bool, default=True)
    train_grp.add_argument('--eval_only', type=bool, default=False)

    model.add_argument('--lora_rank', type=int, default=4, help='LoRA rank.')
    model.add_argument('--lora_type', type=str, default="basic_lora", choices=['basic_lora', 'sgp_lora', 'nsp_lora', 'full'], help='Type of LoRA adaptor.')
    model.add_argument('--weight_temp', type=float, default=1.0, help='Projection temperature.')
    model.add_argument('--weight_kind', type=str, default='log1p', choices=["exp", "log1p", "rational1", "rational2", "sqrt_rational2", "power_family", "stretched_exp"])
    model.add_argument('--weight_p', type=float, default=1.0, help='Weight p.')
    model.add_argument('--nsp_eps', type=float, default=0.05, choices=[0.05, 0.10])
    model.add_argument('--nsp_weight', type=float, default=0.0, choices=[0.0, 0.02, 0.05])
    
    aux = parser.add_argument_group('auxiliary', 'External / auxiliary dataset')
    aux.add_argument('--auxiliary_data_path', type=str, default='/data1/open_datasets', help='Root path of the auxiliary dataset.')
    aux.add_argument('--aux_dataset', type=str, default='imagenet', help='Dataset type for auxiliary data (e.g. imagenet, cifar).', choices=['imagenet', 'flickr8k'])
    aux.add_argument('--auxiliary_data_size', type=int, default=1024, help='Number of samples drawn from the auxiliary dataset each epoch.')

    reg = parser.add_argument_group('regularisation', 'Extra regularisation terms') 
    reg.add_argument('--l2_protection', action='store_true', default=False, help='Enable L2‑protection between the current and previous network.')
    reg.add_argument('--l2_protection_lambda', type=float, default=1e-4, help='Weight for the L2‑protection term (higher → stronger regularisation). When `--l2_protection` is off, this will be automatically set to 0.0.')
    return parser

# In[]
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args = set_smart_defaults(args)
    args = vars(args)
    main(args)