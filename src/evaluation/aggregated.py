from __future__ import annotations
import numpy as np
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score

from src.config import N_CLASSES, INV_MAP


def evaluate_aggregated(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
) -> dict:
    """
    Compute and print overall metrics across the full test set.

    Returns a dict with scalar summary metrics for downstream use.
    """
    class_names = [INV_MAP[i] for i in range(N_CLASSES)]

    print("\n" + "=" * 60)
    print("AGGREGATED EVALUATION (full test set)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    f2_weighted = fbeta_score(y_test, y_pred, average='weighted', beta=2)
    f2_macro    = fbeta_score(y_test, y_pred, average='macro',    beta=2)
    print(f"Weighted F2 (beta=2): {f2_weighted:.4f}")
    print(f"Macro F2    (beta=2): {f2_macro:.4f}")

    for c in range(N_CLASSES):
        if np.sum(y_test == c) > 0:
            f2_c = fbeta_score(
                (y_test == c).astype(int), (y_pred == c).astype(int), beta=2
            )
            print(f"  {INV_MAP[c]}: F2 = {f2_c:.4f}")

    try:
        auc_macro    = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='weighted')
        print(f"\nROC-AUC (OvR) Macro:    {auc_macro:.4f}")
        print(f"ROC-AUC (OvR) Weighted: {auc_weighted:.4f}")

        for c in range(N_CLASSES):
            if np.sum(y_test == c) > 0:
                auc_c = roc_auc_score((y_test == c).astype(int), y_pred_probs[:, c])
                print(f"  {INV_MAP[c]}: AUC = {auc_c:.4f}")
    except ValueError as e:
        auc_macro = auc_weighted = float('nan')
        print(f"\nROC-AUC: could not compute — {e}")

    return {
        'f2_weighted': f2_weighted,
        'f2_macro': f2_macro,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
    }
