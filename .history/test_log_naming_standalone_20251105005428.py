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
