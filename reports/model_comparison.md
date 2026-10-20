# Model comparison

Generated 2026-10-17T16:17:09+00:00 - champion **logreg-eng** (registry version 1).

Selection rule: highest mean 5-fold CV ROC-AUC on the training split. Test metrics come from a stratified 20% hold-out that was never used for tuning.

| Run | CV ROC-AUC | CV acc | CV prec | CV recall | CV F1 | Test ROC-AUC | Test acc | Test prec | Test recall | Overfit gap |
|---|---|---|---|---|---|---|---|---|---|---|
| logreg-eng | 0.915 +/- 0.013 | 0.831 | 0.834 | 0.792 | 0.809 | 0.948 | 0.869 | 0.812 | 0.929 | 0.010 |
| logreg-base | 0.908 +/- 0.014 | 0.843 | 0.863 | 0.792 | 0.823 | 0.953 | 0.885 | 0.839 | 0.929 | 0.025 |
| rf-eng | 0.903 +/- 0.021 | 0.810 | 0.817 | 0.764 | 0.782 | 0.945 | 0.885 | 0.862 | 0.893 | 0.067 |
| rf-base | 0.900 +/- 0.026 | 0.822 | 0.831 | 0.774 | 0.797 | 0.950 | 0.918 | 0.897 | 0.929 | 0.064 |
| gbm-eng | 0.894 +/- 0.024 | 0.826 | 0.845 | 0.774 | 0.802 | 0.956 | 0.902 | 0.867 | 0.929 | 0.073 |
| gbm-base | 0.883 +/- 0.028 | 0.822 | 0.851 | 0.756 | 0.795 | 0.960 | 0.902 | 0.867 | 0.929 | 0.078 |

## Champion hyper-parameters

```json
{
  "C": 0.1,
  "class_weight": null,
  "penalty": "l2"
}
```
