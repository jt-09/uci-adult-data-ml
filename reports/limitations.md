# Limitations

- **Historical data:** UCI Adult reflects a specific U.S. survey era; patterns may not transfer to current populations or geographies.
- **Coarse labels:** Binary income bands hide within-class heterogeneity and inflation effects.
- **Fairness metrics:** Subgroup FPR/FNR/recall describe this model on this test split; they are not causal fairness guarantees and do not prescribe deployment policy.
- **Sensitive attributes:** Sex and race are used for auditing; removal (Section 17) reduces accuracy while proxy features remain.
- **Hyperparameter search:** Randomized search with bounded iterations is not exhaustive; reported CV scores have fold variance.
- **Interpretability:** SHAP and permutation importance operate on encoded features; raw categorical labels require additional mapping from the fitted pipeline.
- **Neural models:** PyTorch embedding MLP was not fully optimised; reported underperformance is not a definitive ceiling for neural tabular methods.

These limitations are reflected in Sections 3, 15, 17, and 19 of `reports/report.md`.
