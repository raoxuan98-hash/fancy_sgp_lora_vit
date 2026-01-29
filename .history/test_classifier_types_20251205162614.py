
#!/usr/bin/env python3
"""
测试脚本：验证classifier_types参数是否正确传递到build_classifiers
"""

import sys
import os
sys.path.append('/home/raoxuan/projects/low_rank_rda')

from main import build_parser

def test_classifier_types_parameter():
    """测试classifier_types参数是否正确添加"""
    parser = build_parser()
    
    # 测试默认参数
    args = parser.parse_args([])
    print("默认参数测试:")
    print(f"  classifier_types: {args.classifier_types}")
    
    # 测试指定单个分类器类型
    args = parser.parse_args(['--classifier_types', 'lda'])
    print("\n指定单个分类器类型测试:")
    print(f"  classifier_types: {args.classifier_types}")
    
    # 测试指定多个分类器类型
    args = parser.parse_args(['--classifier_types', 'lda', 'qda', 'sgd'])
    print("\n指定多个分类器类型测试:")
    print(f"  classifier_types: {args.classifier_types}")
    
    # 测试参数验证（应该会失败）
    try:
        args = parser.parse_args(['--classifier_types', 'invalid_type'])
        print("\n无效分类器类型测试: 意外通过!")
    except SystemExit:
        print("\n无效分类器类型测试: 正确拒绝了无效类型")
    
    print("\n✅ classifier_types参数测试完成")

def test_parameter_chain():
    """测试参数传递链条"""
    from classifier.classifier_builder import ClassifierReconstructor
    import torch
    
    # 模拟参数
    args = {
        'lda_reg_alpha': 0.1,
