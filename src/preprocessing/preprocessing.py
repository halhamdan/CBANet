from __future__ import annotations
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from src.config import N_CLASSES, INV_MAP


def apply_smote_augmentation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_ratio_to_majority: float = 0.80,
    k_neighbors: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes using SMOTE up to target_ratio_to_majority of the majority count."""
    original_shape = X_train.shape
    X_flat = X_train.reshape(original_shape[0], -1)

    class_counts   = np.bincount(y_train, minlength=N_CLASSES)
    majority_class = int(np.argmax(class_counts))
    majority_count = int(class_counts[majority_class])
    target_cap     = int(np.floor(majority_count * target_ratio_to_majority))

    target_samples = {
        c: max(int(class_counts[c]), target_cap)
        for c in range(N_CLASSES)
        if c != majority_class
    }

    if all(target_samples[c] == int(class_counts[c]) for c in target_samples):
        print("SMOTE skipped (targets equal to current counts).")
        return X_train, y_train

    print(f"Applying SMOTE: majority={INV_MAP[majority_class]} (n={majority_count}), targets={target_samples}")
    smote = SMOTE(sampling_strategy=target_samples, random_state=42, k_neighbors=k_neighbors)
    X_res_flat, y_res = smote.fit_resample(X_flat, y_train)
    X_res = X_res_flat.reshape(-1, original_shape[1], original_shape[2])

    print(f"After SMOTE: {X_res.shape}, class dist={np.bincount(y_res, minlength=N_CLASSES)}")
    return X_res, y_res


def normalize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on training windows and transform all splits."""
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_val, X_test, scaler


def compute_enhanced_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights with a 1.3x boost for braking and turning."""
    class_counts = np.bincount(y_train, minlength=N_CLASSES)
    total        = len(y_train)
    balanced     = total / (N_CLASSES * np.maximum(class_counts, 1))

    enhanced     = balanced.copy()
    enhanced[2] *= 1.3   # Harsh Braking
    enhanced[3] *= 1.3   # Harsh Turning
    enhanced     = enhanced / enhanced.mean()

    cw = {i: float(enhanced[i]) for i in range(N_CLASSES)}
    print(f"Class counts: {class_counts} | Weights: {cw}")
    return cw
