from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, roc_auc_score

from src.config import N_CLASSES, INV_MAP

_SHORT = {
    'Normal': 'Normal',
    'Harsh Acceleration': 'H.Accel',
    'Harsh Braking': 'H.Brake',
    'Harsh Turning': 'H.Turn',
}
_COL = [_SHORT[INV_MAP[c]] for c in range(N_CLASSES)]


def evaluate_by_driver(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    driver_ids: np.ndarray,
    save_csv: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-driver classification metrics on the test set.

    Parameters
    ----------
    y_test, y_pred, y_pred_probs : aligned prediction arrays
    driver_ids : string array aligned with the above (test-set only)
    save_csv   : optional path to write results CSV

    Returns
    -------
    pd.DataFrame with one row per driver
    """
    drivers = np.unique(driver_ids)
    rows = []

    for driver in sorted(drivers):
        mask = driver_ids == driver
        n    = int(mask.sum())
        if n == 0:
            continue

        yt   = y_test[mask]
        yp   = y_pred[mask]
        prob = y_pred_probs[mask]

        acc         = accuracy_score(yt, yp)
        f1_macro    = f1_score(yt, yp, average='macro',    zero_division=0)
        f1_weighted = f1_score(yt, yp, average='weighted', zero_division=0)
        f2_weighted = fbeta_score(yt, yp, beta=2, average='weighted', zero_division=0)
        f2_macro    = fbeta_score(yt, yp, beta=2, average='macro',    zero_division=0)

        per_cls_f1 = f1_score(yt, yp, average=None, zero_division=0, labels=range(N_CLASSES))

        try:
            auc = roc_auc_score(yt, prob, multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')

        row = {
            'driver':       driver,
            'n_windows':    n,
            'accuracy':     round(acc, 4),
            'f1_macro':     round(f1_macro, 4),
            'f1_weighted':  round(f1_weighted, 4),
            'f2_weighted':  round(f2_weighted, 4),
            'f2_macro':     round(f2_macro, 4),
            'auc_macro':    round(auc, 4),
        }
        for i, col in enumerate(_COL):
            row[f'f1_{col}'] = round(float(per_cls_f1[i]), 4)

        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("EVALUATION BY DRIVER")
    print("=" * 60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    print(df.to_string(index=False))

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"Driver results saved to {save_csv}")

    return df
