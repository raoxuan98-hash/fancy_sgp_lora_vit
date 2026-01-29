#!/usr/bin/env python3
"""
Test script for cross-domain data loading functionality.
"""

import sys
import logging
from utils.data_manager import DataManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_cross_domain_data_loading():
    """Test cross-domain data loading functionality."""
    
    print("=" * 80)
    print("Testing Cross-Domain Data Loading")
    print("=" * 80)
    
    # Test 1: Create DataManager with cross-domain enabled
    print("\n[Test 1] Creating DataManager with cross-domain mode...")
    args = {
        'cross_domain': True,
        'cross_domain_datasets': [
            'caltech-101', 'dtd', 'eurosat', 'fgvc_aircraft',
            'food101', 'mnist', 'oxford_flower102', 'oxford_pets',
            'cars196_224', 'imagenet-r'
        ]
    }
    
    try:
        dm = DataManager(
            dataset_name='cross_domain_elevater',
            shuffle=False,
            seed=1993,
            init_cls=10,  # These are ignored in cross-domain mode
            increment=10,
            args=args
        )
        print("✓ DataManager created successfully")
    except Exception as e:
        print(f"✗ Failed to create DataManager: {e}")
        return False
    
    # Test 2: Check number of tasks
    print("\n[Test 2] Checking number of tasks...")
    try:
        nb_tasks = dm.nb_tasks
        print(f"Number of tasks: {nb_tasks}")
        if nb_tasks == 10:
            print("✓ Correct number of tasks (10)")
        else:
            print(f"✗ Expected 10 tasks, got {nb_tasks}")
            return False
    except Exception as e:
        print(f"✗ Failed to get number of tasks: {e}")
        return False
    
    # Test 3: Check task sizes
    print("\n[Test 3] Checking task sizes...")
    try:
        expected_task_sizes = {
            0: 102,  # caltech-101
            1: 47,   # dtd
            2: 10,   # eurosat_clip
            3: 102,  # fgvc-aircraft-2013b-variants102
            4: 101,  # food-101
            5: 10,   # mnist
            6: 102,  # oxford-flower-102
            7: 37,   # oxford-iiit-pets
            8: 196,  # stanford-cars
            9: 200   # imagenet-r
        }
        
        for task_id in range(dm.nb_tasks):
            task_size = dm.get_task_size(task_id)
            expected = expected_task_sizes.get(task_id, -1)
            print(f"  Task {task_id}: {task_size} classes (expected: {expected})")
            if task_size != expected:
                print(f"✗ Task {task_id} size mismatch")
                return False
        
        print("✓ All task sizes correct")
    except Exception as e:
        print(f"✗ Failed to check task sizes: {e}")
        return False
    
    # Test 4: Check global label mapping
    print("\n[Test 4] Checking global label mapping...")
    try:
        total_classes = 0
        for task_id in range(dm.nb_tasks):
            task_classes = dm.get_task_classes(task_id, cumulative=False)
            print(f"  Task {task_id}: classes {min(task_classes)}-{max(task_classes)} ({len(task_classes)} classes)")
            
            # Check that classes are contiguous
            if task_classes != list(range(min(task_classes), max(task_classes) + 1)):
                print(f"✗ Task {task_id} classes are not contiguous")
                return False
            
            total_classes += len(task_classes)
        
        print(f"✓ Global label mapping correct (total classes: {total_classes})")
    except Exception as e:
        print(f"✗ Failed to check global label mapping: {e}")
        return False
    
    # Test 5: Load data for each task
    print("\n[Test 5] Loading data for each task...")
    try:
        for task_id in range(min(3, dm.nb_tasks)):  # Test first 3 tasks to save time
            print(f"  Testing task {task_id}...")
            
            # Get train subset
            train_subset = dm.get_subset(task_id, source="train", cumulative=False, mode="train")
            print(f"    Train samples: {len(train_subset)}")
            
            # Get test subset
            test_subset = dm.get_subset(task_id, source="test", cumulative=False, mode="test")
            print(f"    Test samples: {len(test_subset)}")
            
            # Check that we can access a sample
            if len(train_subset) > 0:
                sample = train_subset[0]
                if isinstance(sample, tuple) and len(sample) == 2:
                    print(f"    Sample format: (data, label) - label: {sample[1]}")
                else:
                    print(f"    Sample format: {type(sample)}")
            
            # Test cumulative mode
            cum_train_subset = dm.get_subset(task_id, source="train", cumulative=True, mode="train")
            print(f"    Cumulative train samples: {len(cum_train_subset)}")
            
            if task_id > 0 and len(cum_train_subset) <= len(train_subset):
                print(f"✗ Cumulative data should have more samples than non-cumulative")
                return False
        
        print("✓ Data loading successful for all tested tasks")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test cumulative mode across all tasks
    print("\n[Test 6] Testing cumulative mode...")
    try:
        # Get cumulative data for last task
        last_task_id = dm.nb_tasks - 1
        cum_train_subset = dm.get_subset(last_task_id, source="train", cumulative=True, mode="train")
        cum_test_subset = dm.get_subset(last_task_id, source="test", cumulative=True, mode="test")
        
        print(f"  Cumulative train samples (all tasks): {len(cum_train_subset)}")
        print(f"  Cumulative test samples (all tasks): {len(cum_test_subset)}")
        
        # Check that cumulative data has more samples than last task alone
        last_train_subset = dm.get_subset(last_task_id, source="train", cumulative=False, mode="train")
        if len(cum_train_subset) <= len(last_train_subset):
            print("✗ Cumulative data should have more samples than last task alone")
            return False
        
        print("✓ Cumulative mode working correctly")
    except Exception as e:
        print(f"✗ Failed to test cumulative mode: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("Cross-domain data loading is working correctly.")
    print("=" * 80)
    
    return True


def test_within_domain_fallback():
    """Test that within-domain mode still works (fallback test)."""
    
    print("\n" + "=" * 80)
    print("Testing Within-Domain Fallback")
    print("=" * 80)
    
    print("\n[Test] Creating DataManager for within-domain dataset...")
    try:
        dm = DataManager(
            dataset_name='cifar100_224',
            shuffle=True,
            seed=1993,
            init_cls=10,
            increment=10,
            args={}  # No cross_domain flag
        )
        print("✓ DataManager created successfully")
        
        print(f"Number of tasks: {dm.nb_tasks}")
        print("✓ Within-domain mode working")
        
    except Exception as e:
        print(f"✗ Within-domain mode failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = True
    
    # Test cross-domain mode
    success &= test_cross_domain_data_loading()
    
    # Test within-domain fallback
    success &= test_within_domain_fallback()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)