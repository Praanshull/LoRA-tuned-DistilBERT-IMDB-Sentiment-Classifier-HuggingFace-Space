#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sentiment Analyzer - Entry Point
Fix for macOS multiprocessing issues
"""

import os
import sys

# CRITICAL: Set these BEFORE any other imports
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Force fork mode (prevents some macOS issues)
import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set

if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Sentiment Analyzer...")
    print("="*60)
    
    # Import and launch AFTER environment setup
    from app.interface import launch_app
    launch_app()