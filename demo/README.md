# Running and hosting the Gradio demo

## Local (Windows, no Make)

From the project root with `.venv` activated:

```powershell
pip install -r requirements-demo.txt
python scripts\05_tune_models.py   # skip if results\models\final_model.joblib exists
python demo\app.py
```

Or use the helper script:

```powershell
.\scripts\run_demo.ps1
```

Open http://127.0.0.1:7860 in your browser.

## README screenshots and walkthrough video

With the demo server running in another terminal:

```powershell
pip install playwright
python -m playwright install chromium
python scripts\capture_demo_media.py
```

Writes `demo/assets/demo-*.png` and `demo/assets/demo-walkthrough.webm` (embedded in the root README).

## Local (macOS/Linux with Make)

```bash
pip install -r requirements-demo.txt
make tune   # if needed
make demo
```

## GitHub Pages vs a live demo

**GitHub Pages only serves static files** (your Quarto book, PDF copies, images). It cannot run Gradio or Streamlit backends. That is why the book works on Pages but the demo does not.

For a **clickable live demo link in your README** (like Streamlit Community Cloud), use **Hugging Face Spaces**:

1. Create a Space at https://huggingface.co/new-space (SDK: **Gradio**).
2. Upload or sync:
   - `demo/app.py` (rename to `app.py` at Space root, or set `app_file` in README frontmatter)
   - `src/adult_income_ml/` package
   - `configs/`
   - `requirements-demo.txt` as `requirements.txt` (add `lightgbm`, `xgboost`, main deps)
3. Upload `final_model.joblib` to the Space (or attach via [Git LFS](https://git-lfs.com/)) — the model is too large for GitHub `main` but HF Spaces supports it.
4. In Space `app.py`, set `ROOT` to the Space root and ensure `results/models/final_model.joblib` path exists after upload.
5. Add a README badge to your GitHub repo:

```markdown
[![HF Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/adult-income-demo)
```

Replace `YOUR_USERNAME/adult-income-demo` with your Space URL.

## Quick share (temporary, no deploy)

For a one-off link without hosting:

```python
demo.launch(share=True)
```

Gradio prints a public `*.gradio.live` URL valid for ~72 hours. Fine for demos, not for a permanent README link.

## Streamlit comparison

| Platform | Gradio | Streamlit |
|----------|--------|-----------|
| Free public host | Hugging Face Spaces | Streamlit Community Cloud |
| GitHub Pages | No (static only) | No |
| README link | HF Space URL | `*.streamlit.app` URL |

Your repo already uses **GitHub Pages for the Quarto book** and **Hugging Face Spaces** is the matching pattern for the interactive model demo.
