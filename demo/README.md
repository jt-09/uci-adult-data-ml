# Adult Income Prediction Demo

Interactive Gradio UI for the tuned classifier saved by the pipeline.

## Run

From the project root (with the main package already installed via `make setup` or `pip install -e .`):

```bash
pip install -r requirements-demo.txt && make tune && make demo
```

`make tune` trains models and writes `results/models/final_model.joblib`.  
`make demo` launches the Gradio app (default: http://127.0.0.1:7860).

If the model file is missing, the app shows: **Run make tune first**.
