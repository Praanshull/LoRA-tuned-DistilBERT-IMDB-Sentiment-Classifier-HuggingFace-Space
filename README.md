# Sentiment Analyzer — DistilBERT + LoRA + IMDB

A lightweight, explainable sentiment classifier fine-tuned on the IMDB dataset using LoRA.  
Deployed with a Gradio interface on Hugging Face Spaces.

---

## 🚀 Demo
Try the live demo here:

➡️ **Hugging Face Space:**  
https://huggingface.co/spaces/Praanshull/sentiment-analyzer-app

---

## 🧠 Model

The fine-tuned model is hosted separately on Hugging Face:

➡️ **Model Repo:**  
https://huggingface.co/Praanshull/sentiment-analyzer-model

This project automatically loads the model via:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("Praanshull/sentiment-analyzer-model")
tokenizer = AutoTokenizer.from_pretrained("Praanshull/sentiment-analyzer-model")
