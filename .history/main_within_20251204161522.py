
# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import argparse
from trainer import train

def set_smart_defaults(ns):
    """为within-domain实验设置智能默认参数"""
    if not ns.smart_defaults:
        return ns
    if ns.dataset == 'cars196_224':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 600
    elif ns.dataset == 'imagenet-r':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 1500
    elif ns.dataset == 'cifar100_224':
        ns.init_cls, ns.increment, ns.iterations = 10, 10, 1500
    elif ns.dataset == 'cub200_224':
