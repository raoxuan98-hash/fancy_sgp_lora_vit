#!/usr/bin/env python3
"""Test script for cross_domain_gda_param_optimize.py"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test import
try:
    from scripts.cross_domain_gda_param_optimize import main, parse_args, _prepare_args
    print("✅ Successfully imported cross_domain_gda_param_optimize module")
except ImportError as e:
    print(f"❌ Failed to import cross_domain_gda_param_optimize: {e}")
    sys.exit(1)

# Test argument parsing
try:
    args = parse_args()
    print(f"✅ Argument parsing successful. Output dir: {args.output_dir}")
except Exception as e:
    print(f"❌ Argument parsing failed: {e}")
    sys.exit(1)

# Test _prepare_args function
try:
    from scripts.cross_domain_gda_param_optimize import _import_default_args
    base_args = _import_default_args()
    
    # Test with imagenet-r dataset
    prepared_args = _prepare_args(base_args, "imagenet-r", 20, 1993)
    print(f"✅ _prepare_args successful for imagenet-r")
    print(f"   - dataset: {prepared_args['dataset']}")
    print(f"   - cross_domain: {prepared_args['cross_domain']}")
    print(f"   - cross_domain_datasets: {prepared_args['cross_domain_datasets']}")
    print(f"   - iterations: {prepared_args['iterations']}")
    print(f"   - vit_type: {prepared_args['vit_type']}")
    
    # Test with caltech-101 dataset
    prepared_args = _prepare_args(base_args, "caltech-101", 20, 1993)
    print(f"✅ _prepare_args successful for caltech-101")
    print(f"   - dataset: {prepared_args['dataset']}")
    print(f"   - cross_domain: {prepared_args['cross_domain']}")
    print(f"   - cross_domain_datasets: {prepared_args['cross_domain_datasets']}")
    print(f"   - iterations: {prepared_args['iterations']}")
    print(f"   - vit_type: {prepared_args['vit_type']}")
    
except Exception as e:
    print(f"❌ _prepare_args failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! The script should work correctly.")