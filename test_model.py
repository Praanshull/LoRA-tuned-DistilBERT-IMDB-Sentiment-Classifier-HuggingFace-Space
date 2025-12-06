#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify model loading works correctly
Run this BEFORE launching the full app
"""

import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("="*60)
print("🧪 Testing Model Loading...")
print("="*60)

# Test 1: Check if model directory exists
print("\n1️⃣ Checking model directory...")
model_path = "models/merged-model"
if os.path.exists(model_path):
    print(f"   ✓ Model directory found: {model_path}")
    files = os.listdir(model_path)
    print(f"   ✓ Files found: {len(files)} files")
    essential_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    for ef in essential_files:
        if ef in files or any(ef.replace('.bin', '') in f for f in files):
            print(f"   ✓ {ef} present")
        else:
            print(f"   ⚠️  {ef} missing")
else:
    print(f"   ❌ Model directory NOT found: {model_path}")
    print(f"   Current directory: {os.getcwd()}")
    exit(1)

# Test 2: Try to load model
print("\n2️⃣ Testing model loading...")
try:
    from app.inference import load_model, predict_with_scores
    
    print("   Loading model...")
    load_model()
    print("   ✓ Model loaded successfully!")
    
    # Test 3: Try a prediction
    print("\n3️⃣ Testing prediction...")
    test_text = "This movie was great!"
    label, prob, neg_p, pos_p, _ = predict_with_scores(test_text)
    print(f"   Test input: '{test_text}'")
    print(f"   ✓ Prediction: {label}")
    print(f"   ✓ Confidence: {prob:.4f}")
    print(f"   ✓ Negative prob: {neg_p:.4f}")
    print(f"   ✓ Positive prob: {pos_p:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("   You can now run: python run.py")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("❌ TESTS FAILED")
    print("   Please fix the errors above before running the app")
    print("="*60)
    exit(1)