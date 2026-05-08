from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

from src.config import N_CLASSES


def sparse_categorical_focal_loss_enhanced(alpha_weights=None):
    """Class-weighted cross-entropy loss (gamma=0 per ablation Table IV in the paper)."""
    if alpha_weights is None:
        alpha_weights = np.ones(N_CLASSES, dtype=np.float32)
    alpha = tf.constant(alpha_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true    = tf.cast(tf.reshape(y_true, [-1]), tf.int32)   # ensure shape [batch]
        y_pred    = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_oh = tf.one_hot(y_true, depth=N_CLASSES)

        # gamma=0 reduces focal loss to weighted cross-entropy
        class_gamma = tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        gamma_t = tf.reduce_sum(y_true_oh * class_gamma, axis=-1)
        a_t     = tf.reduce_sum(y_true_oh * alpha, axis=-1)

        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)
        return tf.reduce_mean(a_t * tf.pow(1.0 - p_t, gamma_t) * (-tf.math.log(p_t)))

    return loss_fn


def build_enhanced_model(input_shape: tuple, n_classes: int = N_CLASSES) -> Model:
    """CBANet: Conv1D + Bidirectional LSTM with temporal self-attention."""
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 5, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # Temporal self-attention
    attn = layers.Dense(1, activation='tanh')(lstm_out)   # (B, T, 1)
    attn = layers.Flatten()(attn)                          # (B, T)
    attn = layers.Activation('softmax')(attn)              # (B, T)
    attn = layers.RepeatVector(128)(attn)                  # (B, 128, T)
    attn = layers.Permute([2, 1])(attn)                    # (B, T, 128)

    weighted = layers.Multiply()([lstm_out, attn])
    weighted = layers.Bidirectional(layers.LSTM(32))(weighted)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(weighted)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation='softmax')(x)
    return Model(inp, out)
