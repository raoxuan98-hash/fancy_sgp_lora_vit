#!/usr/bin/env python3
"""
Test script to verify the new log naming system works correctly
for different LoRA types and knowledge distillation configurations.
"""

import sys
import os
import tempfile
import shutil
from trainer import build_log_dirs

def test_sgp_lora_naming():
    """Test SGP LoRA naming without KD"""
