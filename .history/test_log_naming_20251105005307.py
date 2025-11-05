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
        
        # Verify params.json doesn't contain NSP parameters
        import json
        params_file = os.path.join(log_path, "params.json")
        with open(params_file, 'r') as f:
            params = json.load(f)
        assert 'nsp_eps' not in params, f"nsp_eps found in params.json but shouldn't be"
        assert 'nsp_weight' not in params, f"nsp_weight found in params.json but shouldn't be"
        
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
        
        # Verify params.json doesn't contain SGP parameters
        import json
        params_file = os.path.join(log_path, "params.json")
        with open(params_file, 'r') as f:
            params = json.load(f)
        assert 'weight_temp' not in params, f"weight_temp found in params.json but shouldn't be"
        assert 'weight_kind' not in params, f"weight_kind found in params.json but shouldn't be"
        assert 'weight_p' not in params, f"weight_p found in params.json but shouldn't be"
        
    print("   ✅ NSP LoRA naming test passed")

def test_basic_lora_with_kd():
    """Test Basic LoRA with Knowledge Distillation"""
    print("🧪 Testing Basic LoRA with KD...")
    
    args = {
        'model_name': 'sldc',
        'user': 'test_user',
        'dataset': 'cifar100_224',
        'vit_type': 'vit-b-p16-mocov3',
        'init_cls': 10,
        'increment': 10,
        'lora_rank': 4,
        'lora_type': 'basic_lora',
        'gamma_kd': 0.1,  # With KD
        'kd_type': 'feat',
        'distillation_transform': 'linear',
        'use_aux_for_kd': True,
        'update_teacher_each_task': True,
        'optimizer': 'adamw',
        'lrate': 1e-4,
        'batch_size': 16,
        'iterations': 2000,
        'seed': 1993
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        _, log_path = build_log_dirs(args, root_dir=temp_dir)
        print(f"   Generated path: {log_path}")
        
        # Verify KD parameters are included
        assert "kd-0p1" in log_path, f"gamma_kd not found in {log_path}"
        assert "type-feat" in log_path, f"kd_type not found in {log_path}"
        assert "dt-linear" in log_path, f"distillation_transform not found in {log_path}"
        assert "aux" in log_path, f"use_aux_for_kd not found in {log_path}"
        assert "utt-1" in log_path, f"update_teacher_each_task not found in {log_path}"
        
        # Verify LoRA parameters are NOT included (basic_lora has none)
        assert "t-" not in log_path or "t-" not in log_path.split("ltype-basic_lora")[1].split("/")[0], f"weight_temp found in {log_path} but shouldn't be"
        assert "eps-" not in log_path, f"nsp_eps found in {log_path} but shouldn't be"
        
        # Verify params.json doesn't contain any LoRA-specific parameters
        import json
        params_file = os.path.join(log_path, "params.json")
        with open(params_file, 'r') as f:
            params = json.load(f)
        assert 'weight_temp' not in params, f"weight_temp found in params.json but shouldn't be"
        assert 'weight_kind' not in params, f"weight_kind found in params.json but shouldn't be"
        assert 'weight_p' not in params, f"weight_p found in params.json but shouldn't be"
        assert 'nsp_eps' not in params, f"nsp_eps found in params.json but shouldn't be"
        assert 'nsp_weight' not in params, f"nsp_weight found in params.json but shouldn't be"
        
    print("   ✅ Basic LoRA with KD naming test passed")

def test_sgp_lora_with_kd():
    """Test SGP LoRA with Knowledge Distillation"""
    print("🧪 Testing SGP LoRA with KD...")
    
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
        'gamma_kd': 0.05,  # With KD
        'kd_type': 'logit',
        'distillation_transform': 'weaknonlinear',
        'use_aux_for_kd': False,
        'optimizer': 'adamw',
        'lrate': 1e-4,
        'batch_size': 16,
        'iterations': 2000,
        'seed': 1993
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        _, log_path = build_log_dirs(args, root_dir=temp_dir)
        print(f"   Generated path: {log_path}")
        
        # Verify both SGP and KD parameters are included
        assert "t-2p0" in log_path, f"weight_temp not found in {log_path}"
        assert "k-log1p" in log_path, f"weight_kind not found in {log_path}"
        assert "p-1" in log_path, f"weight_p not found in {log_path}"
        assert "kd-0p05" in log_path, f"gamma_kd not found in {log_path}"
        assert "type-logit" in log_path, f"kd_type not found in {log_path}"
        assert "dt-weaknonlinear" in log_path, f"distillation_transform not found in {log_path}"
        assert "aux" not in log_path, f"use_aux_for_kd found but should be False"
        assert "utt-1" in log_path, f"update_teacher_each_task not found in {log_path}"
        
        # Verify params.json doesn't contain NSP parameters
        import json
        params_file = os.path.join(log_path, "params.json")
        with open(params_file, 'r') as f:
            params = json.load(f)
        assert 'nsp_eps' not in params, f"nsp_eps found in params.json but shouldn't be"
        assert 'nsp_weight' not in params, f"nsp_weight found in params.json but shouldn't be"
        
    print("   ✅ SGP LoRA with KD naming test passed")

def test_parameter_cross_contamination_warning():
    """Test that warnings are issued for parameter cross-contamination"""
    print("🧪 Testing parameter cross-contamination warnings...")
    
    # Test SGP parameters with basic_lora (should warn)
    args = {
        'model_name': 'sldc',
        'user': 'test_user',
        'dataset': 'cifar100_224',
        'vit_type': 'vit-b-p16-mocov3',
        'init_cls': 10,
        'increment': 10,
        'lora_rank': 4,
        'lora_type': 'basic_lora',  # But using SGP parameters
        'weight_temp': 2.0,
        'weight_kind': 'log1p',
        'gamma_kd': 0.0,
        'optimizer': 'adamw',
        'lrate': 1e-4,
        'batch_size': 16,
        'iterations': 2000,
        'seed': 1993
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Capture logging output
        import logging
        import io
        
        # Create a string buffer to capture log messages
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        logger = logging.getLogger()
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)
        
        try:
            _, log_path = build_log_dirs(args, root_dir=temp_dir)
            log_output = log_buffer.getvalue()
            
            # Check for warning messages
            assert "weight_temp" in log_output and "specific to sgp_lora" in log_output, \
                f"Expected warning about weight_temp not found in: {log_output}"
            assert "weight_kind" in log_output and "specific to sgp_lora" in log_output, \
                f"Expected warning about weight_kind not found in: {log_output}"
                
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)
            handler.close()
    
    print("   ✅ Parameter cross-contamination warning test passed")

def main():
    """Run all tests"""
    print("🚀 Testing new log naming system...")
    print("=" * 60)
    
    try:
        test_sgp_lora_naming()
        test_nsp_lora_naming()
        test_basic_lora_with_kd()
        test_sgp_lora_with_kd()
        test_parameter_cross_contamination_warning()
        
        print("=" * 60)
        print("🎉 All tests passed! The new log naming system works correctly.")
        print("\n📋 Summary of improvements:")
        print("   ✅ LoRA-specific parameters are properly isolated")
        print("   ✅ No cross-contamination between different LoRA types")
        print("   ✅ Knowledge distillation parameters are clearly separated")
        print("   ✅ Warnings are issued for inappropriate parameter usage")
        print("   ✅ Directory structure is clean and descriptive")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
