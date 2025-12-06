import os
# Set environment variables BEFORE importing torch/transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_DIR = "models/merged-model"
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

# Global variables - will be loaded lazily
tokenizer = None
model = None
pipe = None
_MODEL_LOADED = False

def load_model():
    """Lazy load the model, tokenizer, and pipeline"""
    global tokenizer, model, pipe, _MODEL_LOADED
    
    if _MODEL_LOADED:
        return  # Already loaded
    
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        
        # Force CPU on macOS to avoid issues
        device = -1  # Always use CPU on local macOS
        pipe = pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=tokenizer,
            return_all_scores=True, 
            device=device
        )
        _MODEL_LOADED = True
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def predict_with_scores(text):
    """Predict sentiment with proper error handling"""
    try:
        # Ensure model is loaded
        if not _MODEL_LOADED:
            load_model()
        
        out = pipe(text)[0]
        neg_p = pos_p = 0.0
        for item in out:
            if item["label"] in ["NEGATIVE", "LABEL_0"]:
                neg_p = float(item["score"])
            elif item["label"] in ["POSITIVE", "LABEL_1"]:
                pos_p = float(item["score"])

        if pos_p >= neg_p:
            return "POSITIVE", pos_p, neg_p, pos_p, 1
        else:
            return "NEGATIVE", neg_p, neg_p, pos_p, 0
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR", 0.0, 0.0, 0.0, -1

def get_model():
    """Get the loaded model (loads if necessary)"""
    if not _MODEL_LOADED:
        load_model()
    return model

def get_tokenizer():
    """Get the loaded tokenizer (loads if necessary)"""
    if not _MODEL_LOADED:
        load_model()
    return tokenizer

def get_pipeline():
    """Get the loaded pipeline (loads if necessary)"""
    if not _MODEL_LOADED:
        load_model()
    return pipe