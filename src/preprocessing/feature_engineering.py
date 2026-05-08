from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import BASE_FEATURES


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float32")


def add_physics_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Derive 25 physics-inspired features from 7 raw signals."""
    for col in BASE_FEATURES:
        df[col] = to_float(df[col])

    speed_kmh = df['Speed (km/h)']
    df['Speed_ms'] = speed_kmh * (1000.0 / 3600.0)

    long_acc = df['Longitudinal acceleration (g)']
    lat_acc  = df['Lateral acceleration (g)']

    df['Total_G']       = np.sqrt(long_acc**2 + lat_acc**2)
    df['Long_Jerk']     = long_acc.diff().fillna(0.0)
    df['Lat_Jerk']      = lat_acc.diff().fillna(0.0)
    df['Long_Jerk_Abs'] = df['Long_Jerk'].abs()

    df['Kinetic_Turn_Energy'] = lat_acc * speed_kmh

    df['Pos_Long_Acc'] = long_acc.clip(lower=0.0)
    df['Neg_Long_Acc'] = long_acc.clip(upper=0.0).abs()
    df['Delta_Speed']  = speed_kmh.diff().fillna(0.0)

    df['Throttle_Jerk'] = to_float(df['Accelerator_Pedal_Position (%)']).diff().fillna(0.0)
    df['Brake_Jerk']    = to_float(df['Brake_Pedal_Position (%)']).diff().abs().fillna(0.0)
    df['Long_Power']    = df['Speed_ms'] * df['Pos_Long_Acc']

    df['Turn_Sharpness']  = lat_acc.diff().abs().fillna(0.0)
    df['Lat_Acc_Smooth']  = lat_acc.rolling(window=5, min_periods=1).mean()

    df['Deceleration_Power'] = df['Speed_ms'] * df['Neg_Long_Acc']
    df['Brake_Engagement']   = to_float(df['Brake_Pedal_Position (%)']) * df['Neg_Long_Acc']

    df['Harsh_Braking_Score'] = df['Neg_Long_Acc'] * (to_float(df['Brake_Pedal_Position (%)']) / 100.0)
    df['Harsh_Turning_Score'] = lat_acc.abs() * df['Speed_ms']

    final_features = BASE_FEATURES + [
        'Speed_ms',
        'Total_G', 'Long_Jerk', 'Lat_Jerk', 'Long_Jerk_Abs',
        'Kinetic_Turn_Energy',
        'Pos_Long_Acc', 'Neg_Long_Acc', 'Delta_Speed',
        'Throttle_Jerk', 'Brake_Jerk', 'Long_Power',
        'Turn_Sharpness', 'Lat_Acc_Smooth',
        'Deceleration_Power', 'Brake_Engagement',
        'Harsh_Braking_Score', 'Harsh_Turning_Score'
    ]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[final_features] = df[final_features].fillna(0.0).astype("float32")

    return df, final_features
