#!/usr/bin/env python3
"""
Test script to verify real-time logging functionality.
This script simulates the logging setup used in trainer.py and tests
whether log messages are written to file immediately.
"""

import os
import sys
import logging
import time
import tempfile
from pathlib import Path

def test_realtime_logging():
    """Test if logging writes to file in real-time"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'test_record.log')
        
        print(f"Testing real-time logging to: {log_file}")
        
        # Configure logging exactly as in trainer.py (after fix)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(filename)s] => %(message)s',
            handlers=[
                logging.FileHandler(filename=log_file, mode='a', encoding='utf-8', buffering=1),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Test logging messages
        logging.info("=== Real-time Logging Test Started ===")
        logging.info("This message should appear immediately in the log file")
        
        # Check if message is in file immediately
        time.sleep(0.1)  # Small delay to allow file write
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            print(f"\nLog file content length: {len(content)} characters")
            print("First few lines of log file:")
            print("-" * 50)
            lines = content.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(line)
            print("-" * 50)
            
            # Verify messages are present
            if "Real-time Logging Test Started" in content:
                print("✅ SUCCESS: Log messages are written to file immediately!")
                return True
            else:
                print("❌ FAILED: Log messages not found in file")
                return False
        else:
            print("❌ FAILED: Log file was not created")
            return False

def test_old_buffered_logging():
    """Test the old buffered logging behavior for comparison"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, 'test_old_buffered.log')
        
        print(f"\nTesting old buffered logging to: {log_file}")
        
        # Configure logging with old settings (buffered)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(filename)s] => %(message)s',
            handlers=[
                logging.FileHandler(filename=log_file),  # No buffering parameter
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Test logging messages
        logging.info("=== Old Buffered Logging Test ===")
        logging.info("This message might be buffered")
        
        # Check file immediately (might not contain messages yet due to buffering)
        time.sleep(0.1)
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            print(f"Log file content length: {len(content)} characters")
            
            if "Old Buffered Logging Test" in content:
                print("✅ Message found (buffering may have flushed already)")
            else:
                print("⚠️  Message not found immediately (due to buffering)")
        else:
            print("❌ Log file was not created")

if __name__ == "__main__":
    print("🧪 Testing Real-Time Logging Fix")
    print("=" * 60)
    
    success = test_realtime_logging()
    
    # Also test the old behavior for comparison
    test_old_buffered_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Real-time logging is working correctly!")
        print("   Log files should now update immediately when running main.py")
    else:
        print("❌ Real-time logging test failed")
        print("   Check the logging configuration in trainer.py")
