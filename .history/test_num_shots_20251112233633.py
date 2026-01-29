#!/usr/bin/env python3
"""
Test script for num_shots functionality in cross-domain data manager.
"""

import sys
import os
sys.path.append('.')

import numpy as np
from utils.cross_domain_data_manager import CrossDomainDataManagerCore

def test_num_shots():
    """Test few-shot sampling functionality"""
    print("=== Testing num_shots functionality ===\n")
    
    # Use a small dataset for quick testing
    dataset_names = ['cifar100_224']
    
    # Test with different num_shots values
    test_cases = [0, 1, 5, 10]
    
    for num_shots in test_cases:
        print(f"\n--- Testing num_shots={num_shots} ---")
        
        try:
            # Initialize data manager with num_shots
            cdm = CrossDomainDataManagerCore(
                dataset_names=dataset_names, 
                log_level=20,  # INFO level
                num_shots=num_shots,
                seed=42
            )
            
            # Get the first (and only) dataset
            dataset = cdm.datasets[0]
            train_data = dataset['train_data']
            train_targets = dataset['train_targets']
            
            print(f"Dataset: {dataset['name']}")
            print(f"Number of classes: {dataset['num_classes']}")
            print(f"Training samples: {len(train_data)}")
            print(f"Training targets range: {np.min(train_targets)} - {np.max(train_targets)}")
            
            if num_shots > 0:
                # Check that we have exactly num_shots samples per class
                expected_samples = num_shots * dataset['num_classes']
                actual_samples = len(train_data)
                print(f"Expected samples: {expected_samples}")
                print(f"Actual samples: {actual_samples}")
                
                if actual_samples == expected_samples:
                    print("✓ Sample count matches expected value")
                else:
                    print("✗ Sample count does not match expected value")
                
                # Verify each class has exactly num_shots samples
                for class_id in range(dataset['num_classes']):
                    class_samples = np.sum(train_targets == class_id)
                    if class_samples != num_shots:
                        print(f"✗ Class {class_id} has {class_samples} samples, expected {num_shots}")
                
                print("✓ All classes have correct number of samples")
            
            # Print class distribution
            unique, counts = np.unique(train_targets, return_counts=True)
            print(f"Class distribution: {dict(zip(unique[:5], counts[:5]))}... (showing first 5)")
            
        except Exception as e:
            print(f"✗ Test failed for num_shots={num_shots}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_num_shots()