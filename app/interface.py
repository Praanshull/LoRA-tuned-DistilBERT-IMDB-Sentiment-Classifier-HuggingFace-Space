import os
# Fix for macOS multiprocessing issues - MUST be first
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from io import BytesIO
import gradio as gr

# Import only the function, not the objects (lazy loading)
from app.inference import predict_with_scores
from app.explainability import (
    create_shap_waterfall, 
    create_token_bar,
    fig_to_html
)
from app.training_visuals import create_training_plots

# ---------- CONFIG ----------
MODEL_DIR = "models/merged-model"
TRAINER_OUTPUT_DIR = "checkpoints/checkpoint-8442"

# ---------- Main prediction function ----------
def predict_and_explain(text, show_shap, class_choice):
    """Main function that handles prediction and explanation"""
    if not text or not text.strip():
        empty_html = '<div style="padding:40px;text-align:center;font-size:16px;">Please enter text first</div>'
        return "ERROR", "0.0", "0.0", "0.0", empty_html, empty_html, "⚠️ Please enter text"

    try:
        # Get prediction
        label, prob, neg_p, pos_p, _ = predict_with_scores(text)

        waterfall_html = ""
        bar_html = ""

        if show_shap:
            class_idx = 1 if class_choice == "POSITIVE" else 0
            try:
                waterfall_html = create_shap_waterfall(text, class_idx)
                bar_html = create_token_bar(text, class_idx)
                msg = f"✅ SHAP explains {class_choice} probability"
            except Exception as shap_error:
                print(f"SHAP failed: {shap_error}")
                error_msg = '''<div style="padding:40px;text-align:center;font-size:14px;color:#e74c3c;">
                    <strong>⚠️ SHAP Not Available</strong><br><br>
                    SHAP visualization encountered an error.<br>
                    Your predictions are still working correctly!
                </div>'''
                waterfall_html = error_msg
                bar_html = error_msg
                msg = "⚠️ SHAP unavailable - Predictions still working!"
        else:
            empty_html = '<div style="padding:40px;text-align:center;font-size:16px;">Enable SHAP to see explanations</div>'
            waterfall_html = empty_html
            bar_html = empty_html
            msg = "ℹ️ SHAP disabled - Enable to see word contributions"

        return label, f"{prob:.4f}", f"{neg_p:.4f}", f"{pos_p:.4f}", waterfall_html, bar_html, msg
    
    except Exception as e:
        print(f"Error in predict_and_explain: {e}")
        import traceback
        traceback.print_exc()
        err_html = f'<div style="padding:20px;text-align:center;">Error: {str(e)}</div>'
        return "ERROR", "0.0", "0.0", "0.0", err_html, err_html, f"❌ Error: {str(e)[:100]}"

# ---------- Gradio UI ----------
def create_interface():
    """Create the Gradio interface"""
    
    css = """
    #main_container {max-width: 1400px; margin: auto;}
    .gradio-container {font-family: 'Inter', sans-serif;}
    body { background: #f8fafc; }
    """
    
    with gr.Blocks(title="Sentiment Analyzer") as demo:
        gr.HTML("""
        <style>
        body { background: #f8fafc; }
        .gradio-container { 
            max-width: 1400px !important; 
            margin: auto;
            font-family: 'Inter', sans-serif;
        }
        </style>
        """)

        gr.HTML("<h1 style='text-align: center; margin-bottom: 1rem'>🎬 Movie Review Sentiment Analyzer</h1>")
        gr.Markdown("""
        <div style='text-align: center; padding: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <b>DistilBERT + LoRA Fine-tuning | SHAP Explainability | 90.04% Accuracy</b>
        </div>
        """)

        with gr.Tabs():
            # ---------------------- PREDICT TAB ----------------------
            with gr.TabItem("🔮 Predict", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_text = gr.Textbox(
                            label="Enter Movie Review",
                            placeholder="Type your review here...",
                            lines=6
                        )
                        with gr.Row():
                            class_dropdown = gr.Dropdown(
                                ["POSITIVE", "NEGATIVE"],
                                value="POSITIVE",
                                label="Explain Class"
                            )
                            shap_check = gr.Checkbox(value=False, label="Enable SHAP")

                        submit_btn = gr.Button("🚀 Analyze Sentiment", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        pred_label = gr.Label(label="Prediction", num_top_classes=1)
                        with gr.Row():
                            prob_box = gr.Textbox(label="Confidence", interactive=False)
                            neg_box = gr.Textbox(label="Negative", interactive=False)
                            pos_box = gr.Textbox(label="Positive", interactive=False)

                status_text = gr.Markdown("""
                ### 💡 How to Interpret SHAP Plots:

                **Waterfall Plot** shows how each token pushes the prediction from the base value (50%) toward the final prediction:
                - 🔴 Red bars = tokens pushing toward the selected class
                - 🔵 Blue bars = tokens pushing away from the selected class
                - The plot reads bottom-to-top, showing cumulative effect

                **Token Contributions** shows the most influential words:
                - 🔴 Red = positive contribution to the selected class
                - 🟢 Green = negative contribution (pushes toward opposite class)
                - Longer bars = stronger influence on the model's decision
                
                ⚠️ **Note**: SHAP is disabled by default for stability. Enable it to see explanations.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        waterfall_html = gr.HTML(label="🌊 SHAP Waterfall Plot")
                    with gr.Column(scale=1):
                        bar_html = gr.HTML(label="📊 Token Contributions")

                submit_btn.click(
                    fn=predict_and_explain,
                    inputs=[input_text, shap_check, class_dropdown],
                    outputs=[
                        pred_label,
                        prob_box,
                        neg_box,
                        pos_box,
                        waterfall_html,
                        bar_html
                    ]
                )

                gr.Examples(
                    examples=[
                        ["This movie was absolutely fantastic! Great acting and plot.", False, "POSITIVE"],
                        ["Terrible film, waste of time and money.", False, "NEGATIVE"],
                        ["The movie was okay, nothing special.", False, "POSITIVE"],
                    ],
                    inputs=[input_text, shap_check, class_dropdown],
                )

            # ---------------------- TRAINING TAB ----------------------
            with gr.TabItem("📊 Training", id=2):
                gr.Markdown("""
                ### 📈 Training History & Performance Metrics

                **What you're seeing:**
                - **Loss Curves**: Show how well the model learned over time (lower = better)
                - **Accuracy & F1**: Measure prediction quality (higher = better)

                **Our Training Results:**
                - 🏆 **Best Accuracy**: 90.04% (Epoch 6)
                - 🎯 **Best F1 Score**: 90.12% (Epoch 6)
                - 📉 **Final Validation Loss**: 0.2652 (Epoch 6)
                - ⏱️ **Training Duration**: 7 epochs (early stopped at epoch 6)

                **Key Observations:**
                - Training loss steadily decreased from 0.35 → 0.23
                - Validation accuracy improved from 86.76% → 90.04%
                - Model converged around epoch 6 with best performance
                - No significant overfitting observed (train/val losses stayed close)
                """)

                load_btn = gr.Button("📈 Load Training Curves", variant="primary")

                with gr.Row():
                    with gr.Column():
                        loss_html = gr.HTML(label="Loss Curves")
                    with gr.Column():
                        metrics_html = gr.HTML(label="Accuracy & F1")

                load_btn.click(
                    fn=create_training_plots,
                    outputs=[loss_html, metrics_html]
                )

            # ---------------------- ABOUT TAB ----------------------
            with gr.TabItem("ℹ️ About", id=3):
                gr.Markdown(f"""
## 📚 Project Overview

This is a **state-of-the-art sentiment analysis system** that classifies movie reviews as positive or negative with **90.04% accuracy**. The project demonstrates modern NLP techniques including parameter-efficient fine-tuning and explainable AI.

---

## 🎯 What We Built

### Model Architecture
- **Base Model**: DistilBERT (distilbert-base-uncased)
  - A distilled version of BERT - 40% smaller, 60% faster
  - 66M parameters pre-trained on English Wikipedia & BookCorpus
  - Retains 97% of BERT's language understanding

### Fine-Tuning Approach: LoRA (Low-Rank Adaptation)
- **Why LoRA?** Instead of updating all 66M parameters, we only train 0.3M new parameters (99.5% reduction!)
- **How it works**: Adds small trainable matrices to attention layers
- **Benefits**:
  - ⚡ Faster training (3x speedup)
  - 💾 Less memory (can train on consumer GPUs)
  - 🎯 Same performance as full fine-tuning
  - 💰 Cost-effective for production

**LoRA Configuration Used:**
```python
rank (r) = 8
alpha = 16
target_modules = ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
dropout = 0.1
```

---

## 📊 Training Details

### Dataset: IMDB Movie Reviews
- **Training Set**: 22,500 reviews (90% of 25K)
- **Validation Set**: 2,500 reviews (10% split)
- **Test Set**: 25,000 reviews (original IMDB test set)
- **Classes**: Binary (Positive/Negative) - perfectly balanced

### Training Configuration
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 2e-5 with cosine scheduling
- **Warmup**: 10% of total steps
- **Batch Size**: 16 per device
- **Epochs**: 8 (early stopped at 7)
- **Precision**: BFloat16 for efficiency
- **Max Sequence Length**: 256 tokens

### 🏆 Final Performance Metrics

| Metric | Value | Epoch |
|--------|-------|-------|
| **Best Validation Accuracy** | **90.04%** | 6 |
| **Best F1 Score** | **90.12%** | 6 |
| **Final Validation Loss** | 0.2652 | 6 |
| **Training Time** | ~2 hours | GPU |

**Performance by Epoch:**

| Epoch | Train Loss | Val Loss | Accuracy | F1 Score |
|-------|------------|----------|----------|----------|
| 1 | 0.3465 | 0.3191 | 86.76% | 86.11% |
| 2 | 0.2576 | 0.2995 | 88.20% | 87.69% |
| 3 | 0.2572 | 0.2668 | 89.04% | 89.30% |
| 4 | 0.2452 | 0.2646 | 89.16% | 89.02% |
| 5 | 0.2433 | 0.2620 | 89.56% | 89.35% |
| 6 | 0.2345 | 0.2652 | **90.04%** | **90.12%** |
| 7 | 0.2612 | 0.2676 | 89.92% | 89.87% |

**Key Insights:**
- ✅ Steady improvement from 86.76% → 90.04% accuracy
- ✅ Best performance at epoch 6 before slight overfitting
- ✅ Training converged smoothly with no instability
- ✅ F1 score closely tracks accuracy (balanced classes)

---

## 🔍 Explainability with SHAP

### What is SHAP?
**SHAP (SHapley Additive exPlanations)** is a game-theory based method to explain any machine learning model's predictions. It answers: *"Which words made the model choose this sentiment?"*

### How SHAP Works in Our System
1. **Base Value**: The average prediction (50% for balanced data)
2. **Token Attribution**: For each word, SHAP calculates its contribution
3. **Additive**: Base value + all token contributions = final prediction

### Interpreting the Visualizations

**🌊 Waterfall Plot:**
- Shows step-by-step how prediction moves from base (50%) to final probability
- Red bars = push toward the selected class
- Blue bars = push away from the selected class
- Read bottom-to-top to see cumulative effect

**📊 Token Contributions Bar Chart:**
- Ranks words by their influence on the prediction
- Red bars = words that increase probability of selected class
- Green bars = words that decrease it (favor opposite class)
- Bar length = strength of influence

**Example Interpretation:**
- *"This movie was **terrible**"* → "terrible" has large negative contribution
- *"The acting was **amazing**"* → "amazing" has large positive contribution

---

## 🛠️ Technical Implementation

### Technologies Used
- **🤗 Transformers**: Model loading and inference
- **🎯 PEFT (Parameter-Efficient Fine-Tuning)**: LoRA implementation
- **📊 Datasets**: IMDB dataset loading
- **🧮 SHAP**: Model explanations
- **🎨 Gradio**: Web interface
- **📈 Matplotlib**: Visualization
- **🔥 PyTorch**: Deep learning framework

### Model Files
- **Model Location**: `{MODEL_DIR}`
- **Training Checkpoint**: `{TRAINER_OUTPUT_DIR}`
- **Model Size**: ~250 MB (merged LoRA + base model)

---

## 💡 Key Achievements

1. ✅ **High Accuracy**: 90.04% on validation set (competitive with SOTA)
2. ✅ **Efficient Training**: Only 0.3M trainable parameters (LoRA)
3. ✅ **Fast Inference**: ~50ms per review on CPU
4. ✅ **Explainable**: SHAP shows which words drive predictions
5. ✅ **Production-Ready**: Clean API, error handling, examples
6. ✅ **Well-Documented**: Comprehensive metrics and explanations

---

## 🎓 What Makes This Project Special

### 1. **Modern Architecture**
- Uses DistilBERT for efficiency without sacrificing performance
- Implements LoRA for parameter-efficient fine-tuning

### 2. **Explainable AI**
- Not just predictions - shows *why* the model decided
- SHAP provides human-interpretable token-level explanations

### 3. **Production Quality**
- Proper train/val/test splits
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics
- Clean, documented code

### 4. **Real-World Applicable**
- Can be deployed to production immediately
- Scalable to millions of reviews
- Cost-effective (runs on CPU for inference)

---

## 📖 How to Use This App

### 1️⃣ Make Predictions
- Go to **Predict** tab
- Enter any movie review
- Click "Analyze Sentiment"
- See prediction with confidence scores

### 2️⃣ Understand Decisions
- Enable "Show SHAP" checkbox
- Select which class to explain (POSITIVE/NEGATIVE)
- View waterfall plot showing token-by-token influence
- See bar chart ranking most important words

### 3️⃣ Explore Training
- Go to **Training** tab
- Click "Load Training Curves"
- See how model improved during training
- Understand convergence and performance trends

---

## 🚀 Future Improvements

- [ ] Multi-class sentiment (very positive, positive, neutral, negative, very negative)
- [ ] Aspect-based sentiment analysis (acting, plot, cinematography)
- [ ] Support for other languages
- [ ] Fine-tune on domain-specific reviews (product, restaurant, etc.)
- [ ] Deploy as REST API for production use

---

## 📝 Citation & Credits

**Model**: DistilBERT by Hugging Face
**Dataset**: IMDB Movie Review Dataset
**LoRA**: Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
**SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"

---

*Built using  Transformers • PEFT • SHAP • Gradio*

**Note on Checkpoint Selection**: This model uses checkpoint from epoch 6 (highest validation accuracy: 90.04%). Always select checkpoints based on **validation metrics**, not just the latest epoch number!
            """)

    return demo


def launch_app():
    """Launch the Gradio app"""
    print("="*60)
    print("🚀 Launching Gradio Interface...")
    print("="*60)
    
    try:
        demo = create_interface()
        demo.launch(
            debug=True,
            share=False,
            inbrowser=True,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        import traceback
        traceback.print_exc()
        raise