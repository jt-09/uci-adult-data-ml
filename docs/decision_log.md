# Decision Log

| ID | Decision | Rationale |
|----|----------|-----------|
| D-001 | Macro F1 as selection metric | Handles class imbalance |
| D-002 | 80/20 stratified split | Standard hold-out with balanced classes |
| D-003 | One-hot + median/mode imputation | sklearn baseline compatible with all model families |
