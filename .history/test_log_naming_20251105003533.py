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
    print("🧪 Testing SGP LoRA naming...")
    
    args = {
        'model_name': 'sldc',
        'user': 'test_user',
        'dataset': 'cifar100_224',
        'vit_type': 'vit-b-p16-mocov3',
        'init_cls': 10,
        'increment': 10,
        'lora_rank': 4,
        'lora_type': 'sgp_lora',
        'weight_temp': 2.0,
        'weight_kind': 'log1p',
        'weight_p': 1.0,
        'gamma_kd': 0.0,  # No KD
        'optimizer': 'adamw',
        'lrate': 1e-4,
        'batch_size': 16,
        'iterations': 2000,
        'seed': 1993
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        _, log_path = build_log_dirs(args, root_dir=temp_dir)
        print(f"   Generated path: {log_path}")
        
        # Verify SGP parameters are included
        assert "t-2p0" in log_path, f"weight_temp not found in {log_path}"
        assert "k-log1p" in log_path, f"weight_kind not found in {log_path}"
        assert "p-1" in log_path, f"weight_p not found in {log_path}"
        
        # Verify NSP parameters are NOT included
        assert "eps-" not in log_path, f"nsp_eps found in {log_path} but shouldn't be"
        assert "w-" not in log_path or "w-" not in log_path.split("ltype-sgp_lora")[1].split("/")[0], f"nsp_weight found in {log_path} but shouldn't be"
        
        # Verify KD parameters are NOT included
        assert "kd-" not in log_path, f"KD parameters found but gamma_kd=0"
        
    print("   ✅ SGP LoRA naming test passed")

def test_nsp_lora_naming():
    """Test NSP LoRA naming without KD"""
    print("🧪 Testing NSP LoRA naming...")
    
    args = {
        'model_name': 'sldc',
        'user': 'test_user',
        'dataset': 'cifar100_224',
        'vit_type': 'vit-b-p16-mocov3',
        'init_cls': 10,
        'increment': 10,
        'lora_rank': 4,
        'lora_type': 'nsp_lora',
        'nsp_eps': 0.05,
        'nsp_weight': 0.02,
        'gamma_kd': 0.0,  # No KD
        'optimizer': 'adamw',
        'lrate': 1e-4,
        'batch_size': 16,
        'iterations': 2000,
        'seed': 1993
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        _, log_path = build_log_dirs(args, root_dir=temp_dir)
        print(f"   Generated path: {log_path}")
        
        # Verify NSP parameters are included
        assert "eps-0p05" in log_path, f"nsp_eps not found in {log_path}"
        assert "w-0p02" in log_path, f"nsp_weight not found in {log_path}"
        
        # Verify SGP parameters are NOT included
        assert "t-" not in log_path or "t-" not in log_path.split("ltype-nsp_lora")[1].split("/")[0], f"weight_temp found in {log_path} but shouldn't be"
        assert "k-" not in log_path or "k-" not in log_path.split("ltype-nsp_lora")[1].split("/")[0], f"weight_kind found in {log_path} but shouldn't be"
        
        # Verify KD parameters are NOT included
        assert "kd-" not in log_path, f"KD parameters found but gamma_kd=0"
        
    print("   ✅ NSP LoRA naming test passed")

def test_basic_lora_with_kd():
    """Test Basic LoRA with Knowledge Distillation"""
    print("🧪 Testing Basic LoRA with KD...")
    
