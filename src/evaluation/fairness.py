import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, f1_score


def evaluate_by_group(model, X_test, y_test, original_df, group_column="CS_RACA"):
    df_eval = X_test.copy()
    df_eval["y_true"] = y_test
    df_eval["y_pred"] = model.predict(X_test)

    if group_column not in original_df.columns:
        return {"erro": f"Coluna '{group_column}' não encontrada no dataframe original."}

    df_eval[group_column] = original_df.loc[df_eval.index, group_column]

    results = {}

    for group in df_eval[group_column].dropna().unique():
        subset = df_eval[df_eval[group_column] == group]

        if subset.empty:
            continue

        y_true_group = subset["y_true"]
        y_pred_group = subset["y_pred"]

        cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = precision_score(y_true_group, y_pred_group, zero_division=0)
        f1 = f1_score(y_true_group, y_pred_group, zero_division=0)

        results[str(group)] = {
            "size": int(len(subset)),
            "recall": float(recall),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    recalls = [v["recall"] for v in results.values()]
    if recalls:
        results["_summary"] = {
            "recall_gap": float(max(recalls) - min(recalls)),
            "best_group_recall": float(max(recalls)),
            "worst_group_recall": float(min(recalls)),
        }

    return results