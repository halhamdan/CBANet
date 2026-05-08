from __future__ import annotations
import yaml

LABEL_MAP = {'Normal': 0, 'Harsh Acceleration': 1, 'Harsh Braking': 2, 'Harsh Turning': 3}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}
N_CLASSES = 4

BASE_FEATURES = [
    'Longitudinal acceleration (g)',
    'Lateral acceleration (g)',
    'Speed (km/h)',
    'Accelerator_Pedal_Position (%)',
    'Brake_Pedal_Position (%)',
    'Engine_Speed (rpm)',
    'Gradient (%)'
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
