
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
)


p_values = [0.4, 0.5, 0.6]
n_values = [100, 300, 500]
d_values = [3, 10, 15]
g_values = [0.7, 0.8, 0.9]

results = []

for p in p_values:
    for n in n_values:
        for d in d_values:
            for g in g_values:
                metrics = {"Accuracy": [], "Precision": [], "Recall": [],
                           "F1": [], "Balanced_Accuracy": [], "AUC_ROC": [], "AUC_PR": []}

                for _ in range(num_repeats):
                    synthetic_X, synthetic_y = generate_synthetic_data(p, n, d, g)

                    synthetic_X_train, synthetic_X_valid, synthetic_y_train, synthetic_y_valid = train_test_split(
                        synthetic_X, synthetic_y, test_size=0.2, random_state=None, stratify=synthetic_y
                    )

                    logRegCCD = LogRegCCD().fit(synthetic_X_train, synthetic_y_train)


                    def evaluate_performance(model, X, y):
                        return {
                            "Accuracy": model.validate(X, y, accuracy_score),
                            "Precision": model.validate(X, y, precision_score),
                            "Recall": model.validate(X, y, recall_score),
                            "F1": model.validate(X, y, f1_score),
                            "Balanced_Accuracy": model.validate(X, y, balanced_accuracy_score),
                            "AUC_ROC": model.validate(X, y, roc_auc_score),
                            "AUC_PR": model.validate(X, y, average_precision_score)
                        }


                    scores = evaluate_performance(logRegCCD, synthetic_X_valid, synthetic_y_valid)

                    for key in scores:
                        metrics[key].append(scores[key])

                avg_scores = {key: np.mean(values) for key, values in metrics.items()}
                avg_scores.update({"p": p, "n": n, "d": d, "g": g})
                results.append(avg_scores)

results_df = pd.DataFrame(results)

results_df.to_csv("synthetic_experiment_results.csv", index=False)
