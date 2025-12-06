import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from io import BytesIO

# Import getter functions instead of direct imports
from app.inference import get_model, get_tokenizer, get_pipeline

# SHAP explainer
_SHAP_EXPLAINER = None
_SHAP_FAILED = False

def get_shap_explainer():
    global _SHAP_EXPLAINER, _SHAP_FAILED
    
    if _SHAP_FAILED:
        raise Exception("SHAP initialization failed previously")
    
    if _SHAP_EXPLAINER is None:
        try:
            print("Creating SHAP explainer...")
            
            # Get the loaded model, tokenizer, and pipeline
            model = get_model()
            tokenizer = get_tokenizer()
            pipe = get_pipeline()
            
            # Prevent multiprocessing issues on macOS
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            def model_wrapper(texts):
                if isinstance(texts, str):
                    texts = [texts]
                results = []
                for text in texts:
                    with np.errstate(all='ignore'):
                        output = pipe(text)[0]
                        neg_p = pos_p = 0.0
                        for item in output:
                            if item["label"] in ["NEGATIVE", "LABEL_0"]:
                                neg_p = item["score"]
                            elif item["label"] in ["POSITIVE", "LABEL_1"]:
                                pos_p = item["score"]
                        results.append([neg_p, pos_p])
                return np.array(results)

            # Use a simpler masker that's more stable on macOS
            masker = shap.maskers.Text(tokenizer=tokenizer, mask_token="[MASK]", collapse_mask_token=True)
            _SHAP_EXPLAINER = shap.Explainer(
                model_wrapper, 
                masker=masker,
                algorithm="partition",
                silent=True
            )
            print("SHAP explainer ready!")
        except Exception as e:
            print(f"Failed to initialize SHAP: {e}")
            import traceback
            traceback.print_exc()
            _SHAP_FAILED = True
            raise
            
    return _SHAP_EXPLAINER

def fig_to_html(fig):
    """Convert matplotlib figure to HTML img tag with base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'

def create_shap_waterfall(text, class_index=1):
    """Create SHAP waterfall plot - Returns HTML"""
    try:
        explainer = get_shap_explainer()
        shap_values = explainer([text])

        vals = shap_values.values[0, :, class_index]
        base_val = shap_values.base_values[0][class_index]
        tokens = shap_values.data[0]

        expl = shap.Explanation(values=vals, base_values=base_val, data=tokens)

        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(expl, max_display=15, show=False)

        html = fig_to_html(fig)
        print(f"✓ Waterfall HTML generated ({len(html)} chars)")
        return html

    except Exception as e:
        print(f"Waterfall error: {e}")
        import traceback
        traceback.print_exc()
        return f'''<div style="padding:40px;text-align:center;font-size:14px;color:#e74c3c;">
            <strong>⚠️ SHAP Visualization Unavailable</strong><br><br>
            SHAP explainability is not working on your system.<br>
            This is a known issue with SHAP on macOS.<br><br>
            <em>The predictions still work perfectly!</em><br>
            Only the visual explanations are affected.
        </div>'''

def create_token_bar(text, class_index=1, top_k=15):
    """Create token contribution bar chart - Returns HTML"""
    try:
        explainer = get_shap_explainer()
        shap_values = explainer([text])

        vals = shap_values.values[0, :, class_index]
        tokens = shap_values.data[0]

        abs_vals = np.abs(vals)
        top_idx = np.argsort(abs_vals)[-top_k:][::-1]
        top_vals = vals[top_idx]
        top_tokens = [str(tokens[i]) for i in top_idx]

        fig, ax = plt.subplots(figsize=(12, max(8, 0.5 * len(top_tokens))))
        colors = ['#ff6b6b' if v > 0 else '#51cf66' for v in top_vals]

        y_pos = np.arange(len(top_tokens))
        ax.barh(y_pos, top_vals[::-1], color=colors[::-1], edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_tokens[::-1], fontsize=11)
        ax.axvline(0, color='black', linewidth=1.5)
        ax.set_xlabel('SHAP Value', fontsize=13, fontweight='bold')
        ax.set_title(f'Top {top_k} Token Contributions - {["NEGATIVE", "POSITIVE"][class_index]}',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        html = fig_to_html(fig)
        print(f"✓ Bar chart HTML generated ({len(html)} chars)")
        return html

    except Exception as e:
        print(f"Bar error: {e}")
        import traceback
        traceback.print_exc()
        return f'''<div style="padding:40px;text-align:center;font-size:14px;color:#e74c3c;">
            <strong>⚠️ SHAP Visualization Unavailable</strong><br><br>
            SHAP explainability is not working on your system.<br>
            This is a known issue with SHAP on macOS.<br><br>
            <em>The predictions still work perfectly!</em><br>
            Only the visual explanations are affected.
        </div>'''