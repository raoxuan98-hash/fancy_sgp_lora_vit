# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


import argparse
from trainer import train

def set_smart_defaults(ns):
    if not ns.smart_defaults:
        return ns
    if ns.dataset == 'cars196_224':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 1500
    elif ns.dataset == 'imagenet-r':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 2000
    elif ns.dataset == 'cifar100_224':
        ns.init_cls, ns.increment, ns.iterations = 10, 10, 2000
    elif ns.dataset == 'cub200_224':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 1500

    if ns.lora_type == 'full':
        ns.lrate = 1e-3
        ns.optimizer = 'sgd'
        ns.head_scale = 1.0

    if ns.test:
        ns.seed_list = [1993]
        ns.iterations = 100

    return ns


def main(args):
    results = train(args)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group('basic', 'General / high‑level options')
    basic.add_argument('--dataset', type=str, default='cifar100_224', choices=['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224', 'caltech101_224', 'oxfordpet37_224', 'food101_224', 'resisc45_224', 'cross_domain_elevater'], help='Dataset to use')
    basic.add_argument('--smart_defaults', action='store_true', default=False, help='If set, overwrite a few hyper‑parameters according to the dataset.')
    basic.add_argument('--user', type=str, default='sgp_lora_vit_main_experiments', choices=['authors'], help='User identifier (currently unused).')
    basic.add_argument('--test', action='store_true', default=False, help='If set, run a quick test with reduced settings.')
    basic.add_argument('--cross_domain', type=bool, default=True, help='Enable cross-domain class-incremental learning')
    basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['resisc45', 'imagenet-r', 'caltech-101', 'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'], help='List of datasets for cross-domain experiments')
    # basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['imagenet-r', 'caltech-101', 'dtd'], help='List of datasets for cross-domain experiments')
    basic.add_argument('--num_shots', type=int, default=2, help='Number of samples per class for few-shot learning. If > 0, randomly sample num_shots samples per class.')
    basic.add_argument('--num_samples_per_task_for_evaluation', type=int, default=0, help='Number of samples per task for evaluation. If > 0, randomly sample this many samples from each test task for fast evaluation.')

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
    model.add_argument('--vit_type', type=str, default='vit-b-p16-mocov3', choices=['vit-b-p16', 'vit-b-p16-dino', 'vit-b-p16-mae', 'vit-b-p16-clip', 'vit-b-p16-mocov3'], help='ViT backbone variant.')
    model.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')

    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')  
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993, 1996, 1997], help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=2000, help='Training iterations per task.')
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
    
    comp = parser.add_argument_group('compensator', 'Distribution compensator settings')
    comp.add_argument('--compensator_types', type=str, nargs='+', default=['SeqFT', 'SeqFT + linear', 'SeqFT + Hopfield'],
                     choices=['SeqFT', 'SeqFT + linear', 'SeqFT + weaknonlinear', 'SeqFT + Hopfield'],
                     help='Types of compensators to use. Default is all four types.')
    
    return parser

# In[]
if __name__ == '__main__':
    
    parser = build_parser()
    args = parser.parse_args()
    args = set_smart_defaults(args)
    args = vars(args)
    main(args)