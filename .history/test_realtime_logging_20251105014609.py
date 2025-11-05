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
