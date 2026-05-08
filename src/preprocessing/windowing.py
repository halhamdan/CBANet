from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import LABEL_MAP, N_CLASSES


def _windows_for_session(
    df: pd.DataFrame,
    features: list[str],
    wsize: int,
    step: int,
    harsh_fraction: float,
    enable_physics_override: bool,
) -> tuple[list[np.ndarray], list[int]]:
    """Create windows for a single session. Returns lists (may be empty if session is too short)."""
    data   = df[features].values.astype(np.float32)
    labels = df['Label'].map(LABEL_MAP).values.astype(np.int32)
    n      = len(df)

    if n < wsize:
        return [], []

    idx_long_acc    = features.index('Longitudinal acceleration (g)')
    idx_lat_acc     = features.index('Lateral acceleration (g)')
    idx_brake       = features.index('Brake_Pedal_Position (%)')
    idx_throttle    = features.index('Accelerator_Pedal_Position (%)')
    idx_speed       = features.index('Speed (km/h)')
    idx_neg_acc     = features.index('Neg_Long_Acc')
    idx_brake_score = features.index('Harsh_Braking_Score')
    idx_turn_score  = features.index('Harsh_Turning_Score')

    X_list, y_list = [], []

    for i in range(0, n - wsize, step):
        window      = data[i:i + wsize]
        lbls_in_win = labels[i:i + wsize]

        counts = np.bincount(lbls_in_win, minlength=N_CLASSES)

        harsh_counts    = counts.copy()
        harsh_counts[0] = 0
        if harsh_counts.sum() > 0 and (harsh_counts.max() / wsize) >= harsh_fraction:
            win_label = int(np.argmax(harsh_counts))
        else:
            win_label = int(np.argmax(counts))

        if enable_physics_override:
            long_acc_vals    = window[:, idx_long_acc]
            lat_acc_vals     = window[:, idx_lat_acc]
            brake_vals       = window[:, idx_brake]
            throttle_vals    = window[:, idx_throttle]
            speed_vals       = window[:, idx_speed]
            neg_acc_vals     = window[:, idx_neg_acc]
            brake_score_vals = window[:, idx_brake_score]
            turn_score_vals  = window[:, idx_turn_score]

            braking = [
                np.mean(neg_acc_vals) > 0.14,
                np.mean(brake_vals) > 30,
                np.sum(neg_acc_vals > 0.18) >= 5,
                np.mean(brake_score_vals) > 0.05,
                np.mean(speed_vals) > 15,
            ]
            if sum(braking) >= 4:
                win_label = 2

            turning = [
                np.max(np.abs(lat_acc_vals)) > 0.30,
                np.sum(np.abs(lat_acc_vals) > 0.18) >= int(wsize * 0.35),
                np.mean(turn_score_vals) > 0.9,
                np.mean(speed_vals) > 20,
            ]
            if sum(turning) >= 3:
                win_label = 3

            accel = [
                np.mean(long_acc_vals) > 0.14,
                np.mean(throttle_vals) > 45,
                np.sum(long_acc_vals > 0.18) >= 5,
                np.max(long_acc_vals) > 0.30,
            ]
            if sum(accel) >= 3:
                win_label = 1

        X_list.append(window)
        y_list.append(win_label)

    return X_list, y_list


def make_windows_with_metadata(
    df: pd.DataFrame,
    features: list[str],
    wsize: int,
    step: int,
    harsh_fraction: float = 0.10,
    enable_physics_override: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment each session independently into fixed-size windows.

    Windowing per session prevents windows from straddling two different
    recording files. Returns parallel arrays of windows, labels, session IDs,
    and driver IDs — all aligned by index.

    Returns
    -------
    X            : (N, window_size, n_features)
    y            : (N,)  integer class labels
    session_ids  : (N,)  str — source CSV stem for each window
    driver_ids   : (N,)  str — driver identifier extracted from filename
    """
    X_all, y_all       = [], []
    session_ids_all    = []
    driver_ids_all     = []

    for session_id, session_df in df.groupby('session_id', sort=False):
        driver_id = session_df['driver_id'].iloc[0]
        X_sess, y_sess = _windows_for_session(
            session_df.reset_index(drop=True), features,
            wsize, step, harsh_fraction, enable_physics_override,
        )
        if not y_sess:
            print(f"  Warning: session '{session_id}' too short for even one window — skipped.")
            continue

        X_all.append(np.asarray(X_sess, dtype=np.float32))
        y_all.append(np.asarray(y_sess, dtype=np.int32))
        session_ids_all.extend([session_id] * len(y_sess))
        driver_ids_all.extend([driver_id] * len(y_sess))

    return (
        np.concatenate(X_all,  axis=0),
        np.concatenate(y_all,  axis=0),
        np.array(session_ids_all),
        np.array(driver_ids_all),
    )
