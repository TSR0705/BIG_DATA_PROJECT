# Baseline Model Benchmark Comparison

| model | validation_roc_auc | validation_pr_auc | validation_log_loss | validation_accuracy | validation_precision | validation_recall | validation_f1 | test_roc_auc | test_pr_auc | test_log_loss | test_accuracy | test_precision | test_recall | test_f1 | training_time_seconds | prediction_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7763 | 0.2692 | 0.2404 | 0.9200 | 0.5818 | 0.0344 | 0.0649 | 0.7773 | 0.2517 | 0.2417 | 0.9192 | 0.4870 | 0.0252 | 0.0480 | 6.7133 | 0.0298 |
| Random Forest | 0.7467 | 0.2437 | 0.2515 | 0.9194 | 0.5750 | 0.0062 | 0.0122 | 0.7454 | 0.2271 | 0.2545 | 0.9194 | 0.5833 | 0.0056 | 0.0112 | 93.4983 | 0.6120 |
| LightGBM | 0.7886 | 0.2880 | 0.2363 | 0.9203 | 0.5833 | 0.0432 | 0.0805 | 0.7877 | 0.2749 | 0.2376 | 0.9198 | 0.5490 | 0.0376 | 0.0704 | 10.7601 | 0.1710 |
| XGBoost | 0.7897 | 0.2896 | 0.2358 | 0.9205 | 0.5965 | 0.0457 | 0.0848 | 0.7893 | 0.2772 | 0.2369 | 0.9200 | 0.5571 | 0.0419 | 0.0779 | 15.6152 | 0.0870 |
