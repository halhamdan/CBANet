from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_confusion_matrix(
    cm: np.ndarray, class_names, out_path: str, title: str, color_map: str = "Blues"
) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        cm_pct[np.isnan(cm_pct)] = 0.0

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap=color_map,
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss_curve(train: list, val: list, out_path: str, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(train, label="Train Loss")
    plt.plot(val,   label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_accuracy_curve(acc_values: list, out_path: str, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(acc_values, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_weighted_ce_loss(
    loss_values: list, out_path: str, title: str = "Weighted Cross-Entropy Loss"
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(loss_values, label="Weighted CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_unweighted_ce_loss(
    loss_values: list, out_path: str, title: str = "Unweighted Cross-Entropy Loss"
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(loss_values, label="Unweighted CE Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_multiclass_roc_curve(
    y_true, y_score: np.ndarray, class_names, out_path: str,
    title: str = "Multiclass ROC Curve",
) -> None:
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    fpr, tpr, roc_auc_vals = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc_vals[i]   = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(['darkorange', 'cornflowerblue', 'seagreen', 'red'])

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], linestyle='--', color=color,
                 label=f"{class_names[i]} (AUC = {roc_auc_vals[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("AUC per class:")
    for i in range(len(class_names)):
        print(f"  {class_names[i]}: {roc_auc_vals[i]:.3f}")
