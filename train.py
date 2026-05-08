from __future__ import annotations
import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import load_config, INV_MAP, N_CLASSES
from src.data_loading import load_data
from src.preprocessing import (
    add_physics_features,
    make_windows_with_metadata,
    apply_smote_augmentation,
    normalize,
    compute_enhanced_class_weights,
)
from src.model import build_enhanced_model, sparse_categorical_focal_loss_enhanced
from src.evaluation import (
    optimize_class_thresholds,
    predict_with_thresholds,
    evaluate_aggregated,
    evaluate_by_session,
    evaluate_by_driver,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_accuracy_curve,
    plot_weighted_ce_loss,
    plot_unweighted_ce_loss,
    plot_multiclass_roc_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CBANet driving behaviour classifier")
    p.add_argument("--config",      default="config.yaml", help="Path to config YAML")
    p.add_argument("--data",                               help="Override data glob path")
    p.add_argument("--epochs",      type=int,              help="Override training epochs")
    p.add_argument("--batch-size",  type=int,              help="Override batch size")
    p.add_argument("--seed",        type=int,              help="Override random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    if args.data:       cfg['data']['path']           = args.data
    if args.epochs:     cfg['training']['epochs']     = args.epochs
    if args.batch_size: cfg['training']['batch_size'] = args.batch_size
    if args.seed:       cfg['training']['random_seed'] = args.seed

    seed = cfg['training']['random_seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ── 1. Load & engineer features ──────────────────────────────────────────
    print("Loading files...")
    df = load_data(cfg['data']['path'])
    df, final_features = add_physics_features(df)
    print(f"Total rows: {len(df):,} | Features: {len(final_features)}")
    print(f"Sessions: {df['session_id'].nunique()} | Drivers: {df['driver_id'].nunique()}")
    print("Label distribution:")
    print(df['Label'].value_counts())

    # ── 2. Sliding-window segmentation (per session) ─────────────────────────
    pp = cfg['preprocessing']
    print("\nCreating windows (per session)...")
    X, y, session_ids, driver_ids = make_windows_with_metadata(
        df, final_features,
        wsize=pp['window_size'],
        step=pp['step'],
        harsh_fraction=pp['harsh_fraction'],
        enable_physics_override=pp['enable_physics_override'],
    )
    print(f"Windows: {X.shape} | Label dist: {np.bincount(y, minlength=N_CLASSES)}")

    # ── 3. Train / val / test split (index-based to keep metadata aligned) ───
    tr      = cfg['training']
    indices = np.arange(len(X))

    idx_trainval, idx_test = train_test_split(
        indices, test_size=tr['test_size'], stratify=y, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=tr['val_size'], stratify=y[idx_trainval], random_state=seed
    )

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]
    session_ids_test = session_ids[idx_test]
    driver_ids_test  = driver_ids[idx_test]

    print(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # ── 4. SMOTE augmentation ─────────────────────────────────────────────────
    smote_cfg = pp['smote']
    X_train, y_train = apply_smote_augmentation(
        X_train, y_train,
        target_ratio_to_majority=smote_cfg['target_ratio_to_majority'],
        k_neighbors=smote_cfg['k_neighbors'],
    )

    # ── 5. Normalisation ──────────────────────────────────────────────────────
    X_train, X_val, X_test, scaler = normalize(X_train, X_val, X_test)
    print(f"\nFinal sizes: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

    # ── 6. Class weights ──────────────────────────────────────────────────────
    class_weight_dict = compute_enhanced_class_weights(y_train)

    # ── 7. Build & compile ────────────────────────────────────────────────────
    model = build_enhanced_model((pp['window_size'], len(final_features)))
    model.summary()

    opt_cfg = tr['optimizer']
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=opt_cfg['learning_rate'],
            weight_decay=opt_cfg['weight_decay'],
        ),
        loss=sparse_categorical_focal_loss_enhanced(),
        metrics=['accuracy'],
    )

    # ── 8. Callbacks ──────────────────────────────────────────────────────────
    es_cfg = tr['early_stopping']
    lr_cfg = tr['reduce_lr']
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=es_cfg['patience'],
            restore_best_weights=True, verbose=1, min_delta=es_cfg['min_delta'],
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=lr_cfg['factor'], patience=lr_cfg['patience'],
            min_lr=lr_cfg['min_lr'], verbose=1, min_delta=lr_cfg['min_delta'],
        ),
    ]

    # ── 9. Train ──────────────────────────────────────────────────────────────
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=tr['epochs'],
        batch_size=tr['batch_size'],
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 10. Threshold optimisation ────────────────────────────────────────────
    print("\nOptimising thresholds...")
    optimal_thresholds = optimize_class_thresholds(model, X_val, y_val)

    # ── 11. Predict ───────────────────────────────────────────────────────────
    y_pred, y_pred_probs = predict_with_thresholds(model, X_test, optimal_thresholds)

    # ── 12. Evaluation (three segments) ──────────────────────────────────────
    out_cfg  = cfg['output']
    results_dir = os.path.join(out_cfg.get('results_dir', './results'))
    os.makedirs(results_dir, exist_ok=True)

    evaluate_aggregated(y_test, y_pred, y_pred_probs)

    evaluate_by_session(
        y_test, y_pred, y_pred_probs, session_ids_test,
        save_csv=os.path.join(results_dir, 'eval_by_session.csv'),
    )

    evaluate_by_driver(
        y_test, y_pred, y_pred_probs, driver_ids_test,
        save_csv=os.path.join(results_dir, 'eval_by_driver.csv'),
    )

    # ── 13. Save model artifacts ──────────────────────────────────────────────
    models_dir = out_cfg['models_dir']
    os.makedirs(models_dir, exist_ok=True)

    model.save(os.path.join(models_dir, 'cbanet_model.keras'))
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, 'optimal_thresholds.json'), 'w') as f:
        json.dump({str(k): v for k, v in optimal_thresholds.items()}, f, indent=2)
    with open(os.path.join(models_dir, 'feature_list.json'), 'w') as f:
        json.dump(final_features, f, indent=2)
    print(f"\nModel artifacts saved to {models_dir}/")

    # ── 14. Save plots ────────────────────────────────────────────────────────
    plot_dir   = out_cfg['plots_dir']
    model_name = out_cfg['model_name']
    os.makedirs(plot_dir, exist_ok=True)

    class_names = [INV_MAP[i] for i in range(N_CLASSES)]
    train_loss  = history.history['loss']
    val_loss    = history.history['val_loss']
    val_acc     = [v * 100 for v in history.history['val_accuracy']]
    cm          = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cm, class_names, f"{plot_dir}/conf_matrix.png",
                          f"{model_name} - Confusion Matrix (%)")
    plot_loss_curve(train_loss, val_loss, f"{plot_dir}/loss_curve.png",
                    f"{model_name} - Loss Curve")
    plot_weighted_ce_loss(train_loss, f"{plot_dir}/weighted_ce.png")
    plot_unweighted_ce_loss(val_loss, f"{plot_dir}/unweighted_ce.png")
    plot_accuracy_curve(val_acc, f"{plot_dir}/acc_curve.png",
                        f"{model_name} - Accuracy Curve")
    plot_multiclass_roc_curve(y_test, y_pred_probs, class_names,
                              f"{plot_dir}/multiclass_roc.png",
                              f"{model_name} - Multiclass ROC Curve")

    print(f"Plots saved to {plot_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
