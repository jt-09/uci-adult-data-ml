# Fairness Constraints Explained

This note explains what the Adult Income project measured in subgroup fairness diagnostics, how that relates to common formal fairness definitions, why this study did not apply fairness constraints during training, and what a production deployment would need beyond observational metrics.

## What Was Measured

The final LightGBM classifier was evaluated once on the held-out test split (9,763 rows). Predictions were stratified by sex and race, and for each subgroup we computed false positive rate (FPR), false negative rate (FNR), precision, recall, and support. These are descriptive error rates for this model on this split; they do not establish causal mechanisms or legal compliance.

The most salient gaps appear in recall for the positive class (income above $50K). For sex, female recall is 0.546 compared with male recall of 0.685. Women who actually earn above the threshold are missed more often: the female false negative rate is 0.454 versus 0.315 for men. Female records show very low false positive rate (0.016) and high precision (0.816), while male records show higher false positive rate (0.086). The model is conservative about predicting high income for women but more willing to flag men, which depresses female recall despite strong overall accuracy.

For race, Black recall is 0.496 compared with White recall of 0.670, with Black false negative rate 0.504 versus White 0.330. Black individuals who exceed the income threshold are under-detected relative to White individuals. Intersectional cells in the appendix show further spread (for example Female|Black versus Male|White), reinforcing that single-axis summaries hide compound disparities.

## Demographic Parity

Demographic parity (statistical parity) requires that the fraction of positive predictions be similar across groups. Our observed disparities in FPR and recall imply that positive prediction rates likely differ by group even though we did not optimise for parity explicitly. Demographic parity is easy to state but often conflicts with base rates: if high-income prevalence differs by group in the data, forcing equal prediction rates can require predicting high income for some groups more often than the label distribution would suggest. Parity of predicted positives is a policy choice, not a statistical inevitability, and it says nothing about whether those predictions are correct.

## Equalized Odds and Equal Opportunity

Equalized odds requires equal true positive rates (recall for the positive class) and equal false positive rates across groups. Equal opportunity relaxes this to equal TPR only, which directly targets the recall gaps we observed. Post-processing can adjust decision thresholds per group; in-training constraints reweight or penalise violations during fitting. Both need a defined sensitive attribute, a chosen fairness notion, and data to estimate group-conditional rates without leaking test labels into selection.

Our project reports equal opportunity-relevant quantities (recall and FNR) but did not enforce them. The default 0.5 probability threshold was applied uniformly. Threshold tuning for equalized odds is future work because it requires a dedicated validation protocol rather than a single test evaluation.

## The Predictive Parity Conflict

Predictive parity asks that among predicted positives, precision be equal across groups. Our sex breakdown already shows unequal precision (0.816 female versus 0.774 male). Demographic parity, equalized odds, and predictive parity cannot all be satisfied simultaneously when group base rates differ unless the classifier is perfect. Improving one metric often worsens another. Practitioners must choose which error type is least acceptable in context rather than expecting one model to satisfy every definition.

## Why Constraints Were Not Applied Here

Three methodological constraints shaped this decision. First, the protocol reserves the test set for a single evaluation after model selection on cross-validation within the training split. Fairness constraints require iterative adjustment (threshold search, constrained retraining, or hyperparameter sweeps). Doing that on the test set would inflate reported fairness and accuracy. A proper workflow needs a validation fold for fairness tuning, with test touched only at the end.

Second, fairness constraints need explicit validation, not just reporting. Satisfying equal opportunity on one split does not guarantee transfer. Subgroup sizes vary (Black support 907 versus White 8369 on test), so estimated TPR gaps have different uncertainty. Constrained optimisation on sparse intersectional cells risks overfitting noise. This study prioritised auditable diagnostics (Tables 13-14) over claiming constrained fairness.

Third, removing sex and race from the feature set reduced macro F1 by about 0.035 but did not remove disparity. Occupation, marital status, education, and related fields act as proxies. Fairness through unawareness is insufficient when correlated attributes remain. Constraints applied while proxies persist can create superficial compliance: metrics may improve on monitored attributes while discrimination routes through other columns.

## What Production Would Add

A deployment pipeline would separate training, validation, and test, document the chosen fairness definition and acceptable trade-offs against accuracy, and implement per-group threshold policies or constrained training with monitoring on fresh data. Operations would add drift detection, periodic re-estimation of subgroup metrics, appeal workflows, and human review for high-stakes decisions. Legal review would map metrics to regulatory context. Model cards would record constraint versions and known proxy paths. Deployment would treat income prediction as assistive unless revalidated on contemporary, consent-governed data. This project establishes that disparities exist; it stops short of implementing constraints without the validation budget and governance structure production demands.
