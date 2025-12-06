import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from app.explainability import fig_to_html

TRAINER_OUTPUT_DIR = "checkpoints/checkpoint-8442"

def load_training_logs():
    """Load training logs from checkpoint directory"""
    ts_path = Path(TRAINER_OUTPUT_DIR) / "trainer_state.json"
    if ts_path.exists():
        with open(ts_path, 'r') as f:
            j = json.load(f)
        df = pd.DataFrame(j.get("log_history", []))
        train_df = df[df["loss"].notna()][["step","loss"]].copy()
        eval_df = df[df["eval_loss"].notna()][["step","eval_loss","eval_accuracy","eval_f1"]].copy()

        if "eval_accuracy" in eval_df.columns:
            eval_df["eval_accuracy"] = eval_df["eval_accuracy"].apply(
                lambda x: x["accuracy"] if isinstance(x, dict) else x)
        if "eval_f1" in eval_df.columns:
            eval_df["eval_f1"] = eval_df["eval_f1"].apply(
                lambda x: x["f1"] if isinstance(x, dict) else x)
        return train_df, eval_df
    return None, None

def create_training_plots():
    """Create training plots - Returns HTML for both loss and metrics plots"""
    try:
        train_df, eval_df = load_training_logs()

        if train_df is None or eval_df is None or len(eval_df) == 0:
            return '<div style="padding:40px;text-align:center;font-size:16px;">No training logs found</div>', \
                   '<div style="padding:40px;text-align:center;font-size:16px;">No training logs found</div>'

        # Loss plot
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(train_df["step"], train_df["loss"], label="Train Loss",
                linewidth=2.5, marker='o', markersize=4)
        ax1.plot(eval_df["step"], eval_df["eval_loss"], label="Val Loss",
                linewidth=2.5, marker='s', markersize=4)
        ax1.set_xlabel("Training Step", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Loss", fontsize=13, fontweight='bold')
        ax1.set_title("Training vs Validation Loss", fontsize=15, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(alpha=0.3)

        html1 = fig_to_html(fig1)

        # Metrics plot
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.plot(eval_df["step"], eval_df["eval_accuracy"], marker="o",
                label="Accuracy", linewidth=2.5, markersize=6)
        ax2.plot(eval_df["step"], eval_df["eval_f1"], marker="s",
                label="F1 Score", linewidth=2.5, markersize=6)
        ax2.set_xlabel("Training Step", fontsize=13, fontweight='bold')
        ax2.set_ylabel("Score", fontsize=13, fontweight='bold')
        ax2.set_title("Validation Metrics", fontsize=15, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1.05])

        html2 = fig_to_html(fig2)

        print(f"✓ Training curves HTML generated")
        return html1, html2

    except Exception as e:
        print(f"Training plots error: {e}")
        err_html = f'<div style="padding:20px;text-align:center;">Error loading training plots: {str(e)}</div>'
        return err_html, err_html