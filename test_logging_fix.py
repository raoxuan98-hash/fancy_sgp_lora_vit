#!/usr/bin/env python3
"""
简单的日志测试脚本，验证修复后的日志配置是否正常工作
"""

import os
import sys
import logging
import tempfile

def test_logging_configuration():
    """测试新的日志配置方法"""
    
    # 创建临时日志文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp_file:
        log_file_path = tmp_file.name
    
    print(f"测试日志文件: {log_file_path}")
    
    # 使用新的日志配置方法
    log_file_path = log_file_path
    
    # 清除现有的日志处理器，避免冲突
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(filename=log_file_path, mode='a', encoding='utf-8')
    file_handler.stream.reconfigure(line_buffering=True)  # Enable line buffering
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s [%(filename)s] => %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 测试日志消息
    print("开始测试日志...")
    logging.info("这是第一条测试日志消息")
    logging.info("测试参数: dataset=cifar100, batch_size=16")
    logging.info("测试完成")
    
    # 检查日志文件内容
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n日志文件内容长度: {len(content)} 字符")
        print("日志文件内容:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
        if content.strip():
            print("✅ 日志配置工作正常！")
            return True
        else:
            print("❌ 日志文件为空")
            return False
            
    except Exception as e:
        print(f"❌ 读取日志文件时出错: {e}")
        return False
    finally:
        # 清理临时文件
        try:
            os.unlink(log_file_path)
        except:
            pass

if __name__ == "__main__":
    print("🧪 测试日志配置修复")
    print("=" * 60)
    
    success = test_logging_configuration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 日志配置测试通过！")
        print("   现在应该可以正常记录日志了")
    else:
        print("❌ 日志配置测试失败")
        print("   需要进一步调试")
