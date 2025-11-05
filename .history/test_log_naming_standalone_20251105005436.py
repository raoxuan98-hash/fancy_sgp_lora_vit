#!/usr/bin/env python3
"""
Standalone test script to verify the new log naming system works correctly
for different LoRA types and knowledge distillation configurations.
This script only tests the build_log_dirs function without importing the entire project.
"""

import sys
import os
import tempfile
import json
import logging
import io

# Import only the necessary functions from trainer.py
def _fmt(x, *, digits=4):
    """
    压缩数值到短字符串：0.5 -> 0p5, 1e-3 -> 1e-03, 0.200 -> 0p2
    作用：减少路径长度、避免小数点过多。
    """
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    try:
        s = f"{float(x):.{digits}g}"
        s = s.replace('.', 'p')
        return s
    except Exception:
        s = str(x)
        s = s.replace('.', 'p')
        return s

def sanitize_filename(s: str) -> str:
    """移除或替换文件名中的非法字符"""
    import re
    # Windows 非法字符: \ / : * ? " < > |
    s = re.sub(r'[\\/:*?"<>|]', '_', str(s))
    # 可选：压缩连续下划线
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

def short(s: str, maxlen=40):
    """截断过长字符串，不加 hash，仅保留可读性"""
    s = sanitize_filename(str(s))
    if len(s) <= maxlen:
        return s
    return s[:maxlen].rstrip('_')  # 避免截断在下划线处

def _filter_args_by_lora_type(args: dict) -> dict:
    """
