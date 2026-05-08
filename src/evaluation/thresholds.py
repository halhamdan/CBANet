from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.config import N_CLASSES, INV_MAP


def optimize_class_thresholds(model, X_val: np.ndarray, y_val: np.ndarray) -> dict[int, float]:
    """Find per-class probability thresholds that maximise F1 on the validation set."""
    y_pred_probs  = model.predict(X_val, batch_size=128, verbose=0)
    thresholds_out = {}

    for c in range(N_CLASSES):
        y_true_bin           = (y_val == c).astype(int)
        precision, recall, t = precision_recall_curve(y_true_bin, y_pred_probs[:, c])

        if len(t) == 0:
            thresholds_out[c] = 0.5
            continue

        f1 = 2 * (precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-9)
        best_t = float(t[int(np.argmax(f1))])

        # Recall-friendly adjustment for minority classes
        if c in (2, 3):
            best_t = max(0.10, best_t * 0.85)

        thresholds_out[c] = best_t

    print("Optimal thresholds:")
    for c in range(N_CLASSES):
        print(f"  {INV_MAP[c]}: {thresholds_out[c]:.4f}")
    return thresholds_out


def predict_with_thresholds(
    model, X: np.ndarray, thresholds: dict[int, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with per-class thresholds and a slight boost for braking/turning recall."""
    probs = model.predict(X, batch_size=128, verbose=0)
    pred  = np.zeros(len(X), dtype=int)

    for i in range(len(X)):
        p    = probs[i].copy()
        p[2] *= 1.25   # Harsh Braking recall boost
        p[3] *= 1.25   # Harsh Turning recall boost

        candidates = [c for c in range(N_CLASSES) if p[c] >= thresholds[c]]
        pred[i]    = max(candidates, key=lambda c: p[c]) if candidates else int(np.argmax(p))

    return pred, probs
