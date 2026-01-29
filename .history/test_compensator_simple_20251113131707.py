#!/usr/bin/env python3
"""
测试补偿器控制功能的简单脚本（不依赖main.py）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compensator.distribution_compensator import DistributionCompensator

def test_default_compensators():
    """测试默认情况下使用所有补偿器"""
    print("测试1: 默认情况下使用所有补偿器")
    compensator = DistributionCompensator()
    expected_variants = ["SeqFT", "SeqFT + linear_transform", "SeqFT + weaknonlinear_transform", "SeqFT + attention_transform"]
    
    assert set(compensator.compensator_types) == set(expected_variants), f"默认补偿器类型不匹配: {compensator.compensator_types}"
    assert set(compensator.variants.keys()) == set(expected_variants), f"默认变体不匹配: {compensator.variants.keys()}"
    print("✓ 默认情况下使用所有补偿器 - 通过")

def test_partial_compensators():
    """测试只使用部分补偿器"""
    print("\n测试2: 只使用部分补偿器")
    partial_compensators = ["SeqFT", "SeqFT + linear_transform"]
    compensator = DistributionCompensator(compensator_types=partial_compensators)
    
    assert compensator.compensator_types == partial_compensators, f"指定的补偿器类型不匹配: {compensator.compensator_types}"
    assert set(compensator.variants.keys()) == set(partial_compensators), f"指定的变体不匹配: {compensator.variants.keys()}"
    print("✓ 只使用部分补偿器 - 通过")

def test_single_compensator():
    """测试只使用一个补偿器"""
    print("\n测试3: 只使用一个补偿器")
    single_compensator = ["SeqFT"]
    compensator = DistributionCompensator(compensator_types=single_compensator)
    
    assert compensator.compensator_types == single_compensator, f"单个补偿器类型不匹配: {compensator.compensator_types}"
    assert set(compensator.variants.keys()) == set(single_compensator), f"单个变体不匹配: {compensator.variants.keys()}"
    print("✓ 只使用一个补偿器 - 通过")

def test_argument_parsing_simple():
    """测试简单的命令行参数解析"""
    print("\n测试4: 简单的命令行参数解析")
    import argparse
    
    # 创建一个简单的解析器，只测试我们的参数
    parser = argparse.ArgumentParser()
    comp = parser.add_argument_group('compensator', 'Distribution compensator settings')
    comp.add_argument('--compensator_types', type=str, nargs='+', 
                     default=['SeqFT', 'SeqFT + linear_transform', 'SeqFT + weaknonlinear_transform', 'SeqFT + attention_transform'], 
                     choices=['SeqFT', 'SeqFT + linear_transform', 'SeqFT + weaknonlinear_transform', 'SeqFT + attention_transform'],
                     help='Types of compensators to use. Default is all four types.')
    
    # 测试默认值
    args = parser.parse_args([])
    expected_default = ['SeqFT', 'SeqFT + linear_transform', 'SeqFT + weaknonlinear_transform', 'SeqFT + attention_transform']
    assert args.compensator_types == expected_default, f"默认补偿器参数不匹配: {args.compensator_types}"
    print("✓ 默认补偿器参数 - 通过")
    
    # 测试指定部分补偿器
    args = parser.parse_args(['--compensator_types', 'SeqFT', 'SeqFT + linear_transform'])
    expected_partial = ['SeqFT', 'SeqFT + linear_transform']
    assert args.compensator_types == expected_partial, f"部分补偿器参数不匹配: {args.compensator_types}"
    print("✓ 部分补偿器参数 - 通过")

if __name__ == "__main__":
    print("开始测试补偿器控制功能...")
    
    try:
        test_default_compensators()
        test_partial_compensators()
        test_single_compensator()
        test_argument_parsing_simple()
        
        print("\n🎉 所有测试通过！补偿器控制功能正常工作。")
        
        print("\n使用示例:")
        print("1. 使用所有补偿器（默认）:")
        print("   python main.py")
        print("\n2. 只使用SeqFT和linear_transform补偿器:")
        print("   python main.py --compensator_types SeqFT 'SeqFT + linear_transform'")
        print("\n3. 只使用SeqFT补偿器:")
        print("   python main.py --compensator_types SeqFT")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        sys.exit(1)