# Decision log

| ID | Decision | Rationale |
|----|----------|-----------|
| D-001 | Macro F1 as selection metric | Class imbalance (~24% positive); accuracy alone hides minority-class errors |
| D-002 | 80/20 stratified split | Simple hold-out with stable class proportions in train and test |
| D-003 | One-hot + median/mode imputation | Works across linear, tree, and boosting models in one pipeline |
| D-004 | LightGBM as final model | Highest CV macro F1 among tuned candidates in the reference run |
| D-005 | Cap tuning iterations at 10 in script | `model_spaces.yaml` allows 20; script uses `min(n_iter, 10)` to keep runtime reasonable on CPU |
| D-006 | Minimum subgroup size 30 | Avoid unstable fairness rates in small intersectional cells |
| D-007 | TreeExplainer for SHAP | Matches final tree-based model; fails gracefully if SHAP cannot run |
