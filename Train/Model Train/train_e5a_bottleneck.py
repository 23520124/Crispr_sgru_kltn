#!/usr/bin/env python3
"""
E5-A: Bottleneck Dense Layer AFTER Flatten

Compares two modes on winning E3v2/E4 baseline architecture:
1. control: Winning architecture, no bottleneck
   (batch, 24, 120) → Flatten → 2880 → Dense(128) → Dense(64) → output

2. bottleneck: Insert Dense bottleneck (64 units) AFTER Flatten
   (batch, 24, 120) → Flatten → 2880 → Dense(64, relu) → Dense(128) → Dense(64) → output

Rationale for 64-unit bottleneck AFTER Flatten:
- Control path: 2880 → Dense(128) requires 2880×128 + 128 = 368,768 params
- Bottleneck path:
    2880 → Dense(64) requires 2880×64 + 64 = 184,384 params
    64 → Dense(128) requires 64×128 + 128 = 8,320 params
    Total before Dense(64): 192,704 params (saves ~176K params)
- Hypothesis: aggressive compression forces more efficient feature representation
- Goal: reduce parameters and overfitting risk

Shared weights:
- CNN encoder + GRU backbone (same input→output shapes)
- Biofusion branch (if stage=='biofusion')
- Head NOT shared: different input shapes (2880 vs 64)

Single factor change:
- Bottleneck layer AFTER Flatten only
- Same CNN encoder, GRU, loss, optimizer, sampling, checkpoint, threshold
- Same splits, feature_set, seed for shared layers

Baseline sources (from E3v2 winners):
- K562: focal_bce + lower_positive_hard, checkpoint val_prauc
- HEK293t: control + lower_positive_hard, checkpoint val_loss
- II5: control + lower_positive, checkpoint val_prauc
- II6: control + lower_positive, checkpoint val_loss

Usage:
    python train_e5a_bottleneck.py --dataset K562 --stage p12 [--feature-set e3a]
"""

import os
import sys
import argparse
import json
import hashlib
import random
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    confusion_matrix
)

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
possible_roots = [
    CURRENT_DIR,
    os.path.abspath(os.path.join(CURRENT_DIR, "..")),
    os.path.abspath(os.path.join(CURRENT_DIR, "../..")),
    os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
]

BASE_DIR = None
for root_candidate in possible_roots:
    if os.path.exists(os.path.join(root_candidate, "Datasets")):
        BASE_DIR = root_candidate
        break

if BASE_DIR is None:
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

print(f"-> BASE_DIR: {BASE_DIR}")
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(BASE_DIR, "Train"))
sys.path.append(os.path.join(BASE_DIR, "Train", "Model Train"))

try:
    import Encoder_sgRNA_off
except ImportError:
    sys.path.append(os.path.abspath("."))
    import Encoder_sgRNA_off

from biological_features import extract_features_batch
from MODEL_E5A_Bottleneck import (
    build_e5a_model, count_model_parameters,
    capture_shared_weights, apply_shared_weights,
    hash_shared_weights, verify_shared_weights_match,
    get_shared_layer_names, get_shared_weights_hash_from_model
)

# Constants
RANDOM_SEED = 42
EPOCHS = 35
STEPS_PER_EPOCH = 120
BATCH_SIZE = 256
N_FOLDS = 5

# E5-A modes to compare
E5A_MODES = ['control', 'bottleneck']
BOTTLENECK_UNITS = 64  # Reduced from 256 to actually decrease parameters

# E5-A baseline configurations (inherited from E3v2 winners + E4 encoder winners)
# CRITICAL: CNN encoder selection based ONLY on E4 validation metrics, NEVER test metrics
E5A_BASELINES = {
    'K562': {
        'p12': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'focal_bce',
            'sampling_mode': 'lower_positive_hard',
            'checkpoint_strategy': 'val_prauc',
            'baseline_source': 'E3v2 focal_bce + lower_positive_hard + val_prauc'
        },
        'biofusion': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'focal_bce',
            'sampling_mode': 'lower_positive_hard',
            'checkpoint_strategy': 'val_prauc',
            'baseline_source': 'E3v2 focal_bce + lower_positive_hard (E4 encoder selection pending)'
        }
    },
    'HEK293t': {
        'p12': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'control',
            'sampling_mode': 'lower_positive_hard',
            'checkpoint_strategy': 'val_loss',
            'baseline_source': 'E3v2 control + lower_positive_hard + val_loss'
        },
        'biofusion': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'control',
            'sampling_mode': 'lower_positive_hard',
            'checkpoint_strategy': 'val_loss',
            'baseline_source': 'E3v2 control + lower_positive_hard (E4 encoder selection pending)'
        }
    },
    'II5': {
        'p12': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'control',
            'sampling_mode': 'lower_positive',
            'checkpoint_strategy': 'val_prauc',
            'baseline_source': 'E3v2 control + lower_positive + val_prauc'
        },
        'biofusion': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'control',
            'sampling_mode': 'lower_positive',
            'checkpoint_strategy': 'val_prauc',
            'baseline_source': 'E3v2 control + lower_positive (E4 encoder selection pending)'
        }
    },
    'II6': {
        'p12': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'control',
            'sampling_mode': 'lower_positive',
            'checkpoint_strategy': 'val_loss',
            'baseline_source': 'E3v2 control + lower_positive + val_loss'
        },
        'biofusion': {
            'cnn_variant': 'p12_control',
            'encoder_winner_source': 'E4 winner not predeclared by validation',
            'whether_encoder_is_e4_winner': False,
            'encoder_selection_metric': None,
            'encoder_selection_value': None,
            'encoder_selection_source': 'Default p12_control (E4 validation-based winner not available)',
            'test_metrics_used_for_selection': False,
            'loss_mode': 'control',
            'sampling_mode': 'lower_positive',
            'checkpoint_strategy': 'val_loss',
            'baseline_source': 'E3v2 control + lower_positive (E4 encoder selection pending)'
        }
    }
}

# Sampling configurations (from E3v2)
SAMPLING_CONFIGS = {
    'p12_control': {'n_pos': 64, 'n_hard': 64, 'n_normal': 128},
    'lower_positive': {'n_pos': 32, 'n_hard': 64, 'n_normal': 160},
    'lower_positive_hard': {'n_pos': 32, 'n_hard': 32, 'n_normal': 192}
}

# Feature set defaults
DATASET_FEATURE_MAP = {
    'II6': 'e3a',
    'HEK293t': 'e3c',
    'II5': 'e3b',
    'K562': 'e3a'
}

# Paper reference results (S1 table)
PAPER_REFERENCE = {
    'K562': {'auroc': 1.000, 'prauc': 0.941},
    'HEK293t': {'auroc': 0.998, 'prauc': 0.801},
    'II5': {'auroc': 0.998, 'prauc': 0.521},
    'II6': {'auroc': 0.996, 'prauc': 0.442},
}

# RNG seed offsets
SEED_OFFSETS = {
    'positive': 0,
    'hard': 1000,
    'normal': 2000,
    'shuffle': 3000
}


def load_data(file_path):
    """Load data with source row IDs."""
    Negative, Positive = [], []
    with open(file_path, 'r') as f:
        for row_id, line in enumerate(f):
            ll = line.strip().split(',')
            try:
                label_item = np.float64(ll[2])
            except (ValueError, IndexError):
                continue
            row_with_id = (ll[0], ll[1], ll[2], row_id)
            if label_item == 0.0:
                Negative.append(row_with_id)
            else:
                Positive.append(row_with_id)
    return Negative, Positive


def encoding(data):
    """Encode sequences with explicit shape (N, 1, 24, 7)."""
    encode = []
    for _, row in data.iterrows():
        en = Encoder_sgRNA_off.Encoder(on_seq=row[0], off_seq=row[1], with_category=True, label=row[2])
        encode.append(en.on_off_code)
    encoded = np.array(encode, dtype=np.float32)
    # Ensure shape (N, 1, 24, 7)
    if encoded.ndim == 3:
        encoded = encoded[:, np.newaxis, :, :]
    assert encoded.shape[1:] == (1, 24, 7), f"Encoding shape mismatch: {encoded.shape}"
    return encoded


def count_mismatches(seq1, seq2):
    """Count mismatches in first 20 positions."""
    limit = min(20, len(seq1), len(seq2))
    return sum(1 for a, b in zip(seq1[:limit], seq2[:limit]) if a != b)


def compute_file_hash(file_path):
    """Hash file for reproducibility."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_split_hash(data_array):
    """Hash split IDs."""
    source_ids = [row[3] for row in data_array]
    return hashlib.md5(str(sorted(source_ids)).encode()).hexdigest()[:16]


def get_shared_weights_hash(model, stage):
    """
    Get hash of shared weights (GRU + head, excluding bottleneck).

    Args:
        model: Keras model
        stage: 'p12' or 'biofusion'

    Returns:
        hash string (16 chars)
    """
    return get_shared_weights_hash_from_model(model, stage)


class E3v2StatefulGenerator:
    """
    Stateful batch generator with hard negative mining (E3v2 protocol).

    Generates 4200 total batches (35 epochs × 120 steps) with consistent RNG sequence.
    Uses separate pool arrays to avoid indexing errors.
    """

    def __init__(self, train_positive, train_negative, train_enc, train_bio, train_df,
                 n_pos, n_hard, n_normal, total_batches, random_seed, stage='p12'):
        """
        Args:
            train_positive: list of (sgRNA, off, label, source_id)
            train_negative: list of (sgRNA, off, label, source_id)
            train_enc: encoded sequences (N, 1, 24, 7) in train_df order
            train_bio: biological features (N, bio_dim) in train_df order
            train_df: full training DataFrame with source_id column
            n_pos, n_hard, n_normal: samples per pool per batch
            total_batches: 4200 (35 epochs × 120 steps)
            random_seed: base seed
            stage: 'p12' or 'biofusion'
        """
        self.n_pos = n_pos
        self.n_hard = n_hard
        self.n_normal = n_normal
        self.total_batches = total_batches
        self.seed = random_seed
        self.stage = stage
        self.batch_counter = 0
        self.actual_draws = {'positive': 0, 'hard': 0, 'normal': 0}

        # Build source_id to train_df index mapping
        source_id_to_train_idx = {row['source_id']: idx for idx, row in train_df.iterrows()}

        # Extract positive pool using actual train_df indices
        positive_train_indices = [source_id_to_train_idx[row[3]] for row in train_positive]
        self.positive_enc = train_enc[positive_train_indices]
        self.positive_bio = train_bio[positive_train_indices] if train_bio is not None else None

        # Separate negatives by mismatch count
        hard_train_indices = []
        normal_train_indices = []
        for row in train_negative:
            train_idx = source_id_to_train_idx[row[3]]
            sgRNA, off = row[0], row[1]
            mm = count_mismatches(sgRNA, off)
            if mm <= 3:
                hard_train_indices.append(train_idx)
            else:
                normal_train_indices.append(train_idx)

        # Fallback if pools are empty
        self.hard_fallback = False
        self.normal_fallback = False

        if len(hard_train_indices) == 0:
            print(f"  WARNING: Hard pool empty, using all negatives for both hard and normal")
            all_neg_indices = [source_id_to_train_idx[row[3]] for row in train_negative]
            hard_train_indices = all_neg_indices
            normal_train_indices = all_neg_indices
            self.hard_fallback = True
            self.normal_fallback = True
        elif len(normal_train_indices) == 0:
            print(f"  WARNING: Normal pool empty, using all negatives")
            all_neg_indices = [source_id_to_train_idx[row[3]] for row in train_negative]
            normal_train_indices = all_neg_indices
            self.normal_fallback = True

        if len(positive_train_indices) == 0:
            raise ValueError("No positive samples in training set")

        # Extract separate arrays for hard and normal pools using actual train_df indices
        self.hard_enc = train_enc[hard_train_indices]
        self.hard_bio = train_bio[hard_train_indices] if train_bio is not None else None

        self.normal_enc = train_enc[normal_train_indices]
        self.normal_bio = train_bio[normal_train_indices] if train_bio is not None else None

        self.pool_sizes = {
            'positive': len(self.positive_enc),
            'hard': len(self.hard_enc),
            'normal': len(self.normal_enc)
        }

        print(f"  Pool sizes: pos={self.pool_sizes['positive']}, "
              f"hard={self.pool_sizes['hard']}, normal={self.pool_sizes['normal']}")

    def __call__(self):
        """Generator function for tf.data.Dataset.from_generator."""
        while self.batch_counter < self.total_batches:
            batch_idx = self.batch_counter

            # Independent RNGs with seed offsets
            pos_seed = self.seed + SEED_OFFSETS['positive'] + batch_idx
            hard_seed = self.seed + SEED_OFFSETS['hard'] + batch_idx
            normal_seed = self.seed + SEED_OFFSETS['normal'] + batch_idx
            shuffle_seed = self.seed + SEED_OFFSETS['shuffle'] + batch_idx

            pos_rng = np.random.RandomState(pos_seed)
            hard_rng = np.random.RandomState(hard_seed)
            normal_rng = np.random.RandomState(normal_seed)
            shuffle_rng = np.random.RandomState(shuffle_seed)

            # Sample with replacement from each pool
            pos_pool_indices = pos_rng.choice(len(self.positive_enc), size=self.n_pos, replace=True)
            hard_pool_indices = hard_rng.choice(len(self.hard_enc), size=self.n_hard, replace=True)
            normal_pool_indices = normal_rng.choice(len(self.normal_enc), size=self.n_normal, replace=True)

            # Extract from respective pools
            batch_enc_parts = [
                self.positive_enc[pos_pool_indices],
                self.hard_enc[hard_pool_indices],
                self.normal_enc[normal_pool_indices]
            ]
            batch_enc = np.concatenate(batch_enc_parts, axis=0)

            if self.stage == 'biofusion':
                batch_bio_parts = [
                    self.positive_bio[pos_pool_indices],
                    self.hard_bio[hard_pool_indices],
                    self.normal_bio[normal_pool_indices]
                ]
                batch_bio = np.concatenate(batch_bio_parts, axis=0)

            labels = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_normal), dtype=np.float32)

            # Shuffle
            perm = shuffle_rng.permutation(len(batch_enc))
            batch_enc = batch_enc[perm]
            labels = labels[perm]

            labels = labels.reshape(-1, 1)  # Shape (batch, 1)

            # Track actual draws
            self.actual_draws['positive'] += self.n_pos
            self.actual_draws['hard'] += self.n_hard
            self.actual_draws['normal'] += self.n_normal

            # Increment counter before yield
            self.batch_counter += 1

            if self.stage == 'p12':
                yield batch_enc, labels
            else:  # biofusion
                batch_bio = batch_bio[perm]
                yield (batch_enc, batch_bio), labels

    def verify_batch_count(self):
        """Verify that exactly total_batches were generated."""
        assert self.batch_counter == self.total_batches, \
            f"Batch count mismatch: generated {self.batch_counter}, expected {self.total_batches}"

        # Verify actual draws match expected
        expected_pos = self.n_pos * self.total_batches
        expected_hard = self.n_hard * self.total_batches
        expected_normal = self.n_normal * self.total_batches

        assert self.actual_draws['positive'] == expected_pos, \
            f"Positive draws mismatch: {self.actual_draws['positive']} != {expected_pos}"
        assert self.actual_draws['hard'] == expected_hard, \
            f"Hard draws mismatch: {self.actual_draws['hard']} != {expected_hard}"
        assert self.actual_draws['normal'] == expected_normal, \
            f"Normal draws mismatch: {self.actual_draws['normal']} != {expected_normal}"

    def get_pool_info(self):
        """Return pool metadata for config."""
        return {
            'pool_sizes': self.pool_sizes,
            'hard_fallback': self.hard_fallback,
            'normal_fallback': self.normal_fallback,
            'expected_total_batches': self.total_batches,
            'actual_batch_counter': self.batch_counter,
            'actual_draws': self.actual_draws,
            'expected_draws_per_pool': {
                'positive': self.n_pos * self.total_batches,
                'hard': self.n_hard * self.total_batches,
                'normal': self.n_normal * self.total_batches
            }
        }


class DualCheckpointCallback(Callback):
    """
    Save two checkpoints per trajectory: best val_loss and best val_prauc.

    PRAUC computed via explicit prediction on validation set (not Keras metric).
    """

    def __init__(self, fold, checkpoint_dir, val_data, val_labels, stage, tolerance=1e-12):
        super().__init__()
        self.fold = fold
        self.checkpoint_dir = checkpoint_dir
        self.val_data = val_data
        self.val_labels = val_labels
        self.stage = stage
        self.tolerance = tolerance

        # Best trackers
        self.best_val_loss = np.inf
        self.best_val_prauc = -np.inf
        self.best_epoch_loss = 0
        self.best_epoch_prauc = 0

        # Create subdirectories
        self.loss_dir = os.path.join(checkpoint_dir, 'val_loss')
        self.prauc_dir = os.path.join(checkpoint_dir, 'val_prauc')
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.prauc_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', np.inf)

        # Compute PRAUC explicitly
        val_proba = self.model.predict(self.val_data, verbose=0).ravel()
        val_prauc = average_precision_score(self.val_labels, val_proba)

        # Save best val_loss checkpoint
        if val_loss < self.best_val_loss - self.tolerance:
            self.best_val_loss = val_loss
            self.best_epoch_loss = epoch + 1
            self.model.save_weights(os.path.join(self.loss_dir, f'fold{self.fold}_best.weights.h5'))

        # Save best val_prauc checkpoint
        if val_prauc > self.best_val_prauc + self.tolerance:
            self.best_val_prauc = val_prauc
            self.best_epoch_prauc = epoch + 1
            self.model.save_weights(os.path.join(self.prauc_dir, f'fold{self.fold}_best.weights.h5'))


def threshold_curve(y_true, y_proba):
    """
    Compute F1 and MCC for all unique probability thresholds, including 0.0 and inf.

    Returns:
        best_f1_threshold, best_f1_value, best_mcc_threshold, best_mcc_value
    """
    unique_probs = np.unique(y_proba)
    candidates = [np.inf] + list(unique_probs) + [0.0]

    best_f1 = -1.0
    best_f1_thresh = 0.5
    best_mcc = -2.0
    best_mcc_thresh = 0.5

    for thresh in candidates:
        if thresh == np.inf:
            predicted = np.zeros(len(y_true), dtype=np.int32)
        else:
            predicted = (y_proba >= thresh).astype(np.int32)

        if len(np.unique(predicted)) < 2:
            continue

        f1 = f1_score(y_true, predicted, zero_division=0.0)
        mcc = matthews_corrcoef(y_true, predicted)

        # F1: higher is better, ties → higher threshold
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-12 and thresh > best_f1_thresh):
            best_f1 = f1
            best_f1_thresh = thresh

        # MCC: higher is better, ties → higher threshold
        if mcc > best_mcc or (abs(mcc - best_mcc) < 1e-12 and thresh > best_mcc_thresh):
            best_mcc = mcc
            best_mcc_thresh = thresh

    return best_f1_thresh, best_f1, best_mcc_thresh, best_mcc


def evaluate_with_threshold(y_true, y_proba, threshold):
    """Evaluate metrics at a fixed threshold."""
    y_pred = (y_proba >= threshold).astype(np.int32)

    auroc = roc_auc_score(y_true, y_proba)
    prauc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'threshold': float(threshold),
        'AUROC': float(auroc),
        'PRAUC': float(prauc),
        'F1': float(f1),
        'Precision': float(precision),
        'Recall': float(recall),
        'MCC': float(mcc),
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn)
    }


def train_e5a_bottleneck(dataset, stage, feature_set=None):
    """
    E5-A Bottleneck experiment: control vs bottleneck.

    Args:
        dataset: 'K562', 'HEK293t', 'II5', 'II6'
        stage: 'p12' or 'biofusion'
        feature_set: biological feature set ('e3a', 'e3b', 'e3c') or None (auto-detect)
    """
    print(f"\n{'='*80}")
    print(f"E5-A BOTTLENECK EXPERIMENT: {dataset} - {stage}")
    print(f"{'='*80}\n")

    # Get baseline configuration for dataset and stage
    if dataset not in E5A_BASELINES:
        raise ValueError(f"Dataset {dataset} not supported. Use: {list(E5A_BASELINES.keys())}")

    if stage not in E5A_BASELINES[dataset]:
        raise ValueError(f"Stage {stage} not configured for {dataset}. Use: {list(E5A_BASELINES[dataset].keys())}")

    baseline_config = E5A_BASELINES[dataset][stage]
    cnn_variant = baseline_config['cnn_variant']
    encoder_winner_source = baseline_config['encoder_winner_source']
    whether_encoder_is_e4_winner = baseline_config['whether_encoder_is_e4_winner']
    loss_mode = baseline_config['loss_mode']
    sampling_mode = baseline_config['sampling_mode']
    checkpoint_strategy = baseline_config['checkpoint_strategy']

    print(f"Baseline configuration:")
    print(f"  CNN variant: {cnn_variant}")
    print(f"  Encoder source: {encoder_winner_source}")
    print(f"  Is E4 winner: {whether_encoder_is_e4_winner}")
    print(f"  Loss mode: {loss_mode}")
    print(f"  Sampling mode: {sampling_mode}")
    print(f"  Checkpoint strategy: {checkpoint_strategy}")
    print(f"  Baseline source: {baseline_config['baseline_source']}\n")

    # Feature set
    if feature_set is None:
        feature_set = DATASET_FEATURE_MAP.get(dataset, 'e3a')
    print(f"Feature set: {feature_set}")

    # Sampling configuration
    n_pos = SAMPLING_CONFIGS[sampling_mode]['n_pos']
    n_hard = SAMPLING_CONFIGS[sampling_mode]['n_hard']
    n_normal = SAMPLING_CONFIGS[sampling_mode]['n_normal']

    print(f"Sampling: {n_pos} pos + {n_hard} hard + {n_normal} normal = {BATCH_SIZE} per batch\n")

    # Load data
    data_file = os.path.join(BASE_DIR, "Datasets", f"{dataset}.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    print(f"Loading data from: {data_file}")
    file_hash = compute_file_hash(data_file)
    print(f"File hash: {file_hash}\n")

    Negative, Positive = load_data(data_file)
    all_data = Positive + Negative
    print(f"Loaded: {len(Positive)} positive, {len(Negative)} negative\n")

    # Stratified train/test split (85/15)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    all_df = pd.DataFrame(all_data, columns=['sgRNA', 'off', 'label', 'source_id'])
    all_df['label'] = all_df['label'].astype(float)

    from sklearn.model_selection import train_test_split
    train_val_data, test_data = train_test_split(
        all_data,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=[row[2] for row in all_data]
    )

    test_hash = get_split_hash(test_data)
    print(f"Train+Val: {len(train_val_data)}, Test: {len(test_data)}")
    print(f"Test split hash: {test_hash}\n")

    # Encode test set once
    test_df = pd.DataFrame(test_data, columns=['sgRNA', 'off', 'label', 'source_id'])
    test_enc = encoding(test_df)
    test_labels = test_df['label'].astype('float32').to_numpy()

    test_tuples = list(test_df[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))
    test_bio = extract_features_batch(test_tuples, feature_set=feature_set)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "weights", f"{dataset}_E5A_bottleneck_{stage}_{feature_set}_{timestamp}")
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    splits_dir = os.path.join(output_dir, 'splits')
    histories_dir = os.path.join(output_dir, 'histories')
    predictions_dir = os.path.join(output_dir, 'predictions')

    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(histories_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Save test split
    with open(os.path.join(splits_dir, 'test_source_ids.txt'), 'w') as f:
        f.write('\n'.join([str(row[3]) for row in test_data]))

    # Count parameters for both modes
    param_counts = {}
    for mode in E5A_MODES:
        params = count_model_parameters(
            stage=stage,
            mode=mode,
            cnn_variant=cnn_variant,
            bottleneck_units=BOTTLENECK_UNITS,
            bio_feature_dim=44  # placeholder, will update after actual feature extraction
        )
        param_counts[mode] = params
        print(f"\n{mode} mode:")
        print(f"  CNN variant: {params['cnn_variant']}")
        print(f"  Total params: {params['total_params']}")
        print(f"  Trainable params: {params['trainable_params']}")
        if mode == 'bottleneck':
            print(f"  Bottleneck units: {params['bottleneck_units']}")

    # 5-fold cross-validation
    train_val_df = pd.DataFrame(train_val_data, columns=['sgRNA', 'off', 'label', 'source_id'])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    labels = train_val_df['label'].astype(float)

    # Storage for results
    fold_results = []
    fold_sampling_fallbacks = {}
    e5a_split_hashes = {'test': test_hash}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, labels), start=1):
        print(f"\n{'='*80}")
        print(f"Fold {fold}/{N_FOLDS}")
        print(f"{'='*80}")

        # Split data
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        train_pos = train_df[train_df['label'].astype(float) == 1.0].reset_index(drop=True)
        train_neg = train_df[train_df['label'].astype(float) == 0.0].reset_index(drop=True)

        print(f"Train: {len(train_df)} (pos={len(train_pos)}, neg={len(train_neg)})")
        print(f"Val: {len(val_df)} (pos={len(val_df[val_df['label'].astype(float)==1.0])}, neg={len(val_df[val_df['label'].astype(float)==0.0])})")

        # Save split
        train_tuples = list(train_df[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))
        val_tuples = list(val_df[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))

        train_hash = get_split_hash(train_tuples)
        val_hash = get_split_hash(val_tuples)

        # Store hashes
        e5a_split_hashes[f'fold_{fold}_train'] = train_hash
        e5a_split_hashes[f'fold_{fold}_val'] = val_hash

        with open(os.path.join(splits_dir, f'fold{fold}_train_source_ids.txt'), 'w') as f:
            f.write('\n'.join([str(row[3]) for row in train_tuples]))
        with open(os.path.join(splits_dir, f'fold{fold}_val_source_ids.txt'), 'w') as f:
            f.write('\n'.join([str(row[3]) for row in val_tuples]))

        # Encode and prepare features
        train_enc = encoding(train_df)
        val_enc = encoding(val_df)
        val_labels = val_df['label'].astype('float32').to_numpy()

        train_bio = extract_features_batch(train_tuples, feature_set=feature_set)
        val_bio = extract_features_batch(val_tuples, feature_set=feature_set)

        # Fit scaler on train, transform val and test
        scaler = StandardScaler()
        train_bio = scaler.fit_transform(train_bio).astype(np.float32)
        val_bio = scaler.transform(val_bio).astype(np.float32)
        test_bio_scaled = scaler.transform(test_bio).astype(np.float32)
        bio_dim = train_bio.shape[1]

        print(f"Biological feature dim: {bio_dim}")

        # Separate positive/negative for mining
        train_positive = list(train_pos[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))
        train_negative = list(train_neg[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))

        # Fold seed
        fold_seed = RANDOM_SEED + fold * 100

        # Train both modes with shared initial weights
        fold_mode_results = []
        shared_weights_reference = None
        shared_init_hash_reference = None

        for mode in E5A_MODES:
            print(f"\n{'-'*80}")
            print(f"Fold {fold} - Mode: {mode}")
            print(f"{'-'*80}")

            # Set seeds
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            tf.random.set_seed(fold_seed)

            # Build model
            model = build_e5a_model(
                stage=stage,
                mode=mode,
                cnn_variant=cnn_variant,
                bottleneck_units=BOTTLENECK_UNITS,
                bio_feature_dim=bio_dim,
                loss_mode=loss_mode,
                learning_rate=0.0001
            )

            # Capture or apply shared initial weights
            if mode == E5A_MODES[0]:  # control mode
                # Capture shared weights (CNN + GRU, excluding head)
                shared_weights_reference = capture_shared_weights(model, stage, cnn_variant=cnn_variant)
                shared_init_hash_reference = hash_shared_weights(shared_weights_reference)
                print(f"Captured shared initial weights from control mode")
                print(f"Shared init hash (reference): {shared_init_hash_reference}")
            else:
                # Apply shared weights to bottleneck mode
                apply_shared_weights(model, shared_weights_reference, stage, cnn_variant=cnn_variant)
                print(f"Applied shared initial weights to {mode} mode")
                verify_shared_weights_match(model, shared_weights_reference, stage, cnn_variant=cnn_variant)
                print(f"  ✓ Shared weights verified")

            # Get init hashes
            shared_hash = get_shared_weights_hash_from_model(model, stage, cnn_variant=cnn_variant)
            print(f"Shared init hash: {shared_hash}")

            # Verify hash matches reference
            if shared_hash != shared_init_hash_reference:
                raise RuntimeError(f"Fold {fold}, mode {mode}: Shared init hash mismatch!\n"
                                   f"Expected (reference): {shared_init_hash_reference}\n"
                                   f"Got: {shared_hash}\n"
                                   f"Shared weights must be identical across all modes before training.")

            # Setup checkpoint callback
            mode_checkpoint_dir = os.path.join(checkpoints_dir, mode)

            if stage == 'p12':
                val_data_for_callback = val_enc
            else:
                val_data_for_callback = [val_enc, val_bio]

            checkpoint_callback = DualCheckpointCallback(
                fold,
                mode_checkpoint_dir,
                val_data_for_callback,
                val_labels,
                stage
            )

            # Create stateful generator
            total_batches = EPOCHS * STEPS_PER_EPOCH  # 4200
            generator_obj = E3v2StatefulGenerator(
                train_positive=train_positive,
                train_negative=train_negative,
                train_enc=train_enc,
                train_bio=train_bio,
                train_df=train_df,
                n_pos=n_pos,
                n_hard=n_hard,
                n_normal=n_normal,
                total_batches=total_batches,
                random_seed=fold_seed,
                stage=stage
            )

            # Create tf.data.Dataset from generator
            if stage == 'p12':
                output_signature = (
                    tf.TensorSpec(shape=(BATCH_SIZE, 1, 24, 7), dtype=tf.float32),
                    tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
                )
            else:
                output_signature = (
                    (tf.TensorSpec(shape=(BATCH_SIZE, 1, 24, 7), dtype=tf.float32),
                     tf.TensorSpec(shape=(BATCH_SIZE, bio_dim), dtype=tf.float32)),
                    tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
                )

            train_dataset = tf.data.Dataset.from_generator(
                generator_obj,
                output_signature=output_signature
            )

            # Prepare validation data
            if stage == 'p12':
                val_data = (val_enc, val_labels.reshape(-1, 1))
            else:
                val_data = ([val_enc, val_bio], val_labels.reshape(-1, 1))

            # Shape assertions
            assert train_enc.shape[1:] == (1, 24, 7), f"Train enc shape: {train_enc.shape}"
            assert val_enc.shape[1:] == (1, 24, 7), f"Val enc shape: {val_enc.shape}"
            print(f"Input shapes verified: train={train_enc.shape}, val={val_enc.shape}")

            # Train
            print(f"Training for {EPOCHS} epochs (total {total_batches} batches)...")
            history = model.fit(
                train_dataset,
                epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_data=val_data,
                callbacks=[checkpoint_callback],
                verbose=1
            )

            # Verify batch count
            generator_obj.verify_batch_count()
            print(f"  ✓ Training complete: generated {generator_obj.batch_counter} batches (expected {total_batches})")

            # Record sampling fallbacks and pool info after training
            if mode == E5A_MODES[0]:  # Record once per fold
                pool_info = generator_obj.get_pool_info()
                fold_sampling_fallbacks[fold] = pool_info

            # Save history
            history_path = os.path.join(histories_dir, f'fold{fold}_{mode}_history.json')
            with open(history_path, 'w') as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

            # Load best checkpoint based on checkpoint_strategy
            best_checkpoint_path = os.path.join(mode_checkpoint_dir, checkpoint_strategy, f'fold{fold}_best.weights.h5')
            if not os.path.exists(best_checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint_path}")

            model.load_weights(best_checkpoint_path)
            print(f"\nLoaded checkpoint: {checkpoint_strategy}, epoch {checkpoint_callback.best_epoch_loss if checkpoint_strategy == 'val_loss' else checkpoint_callback.best_epoch_prauc}")

            # Validation predictions and threshold selection
            if stage == 'p12':
                val_proba = model.predict(val_enc, verbose=0).ravel()
                test_proba = model.predict(test_enc, verbose=0).ravel()
            else:
                val_proba = model.predict([val_enc, val_bio], verbose=0).ravel()
                test_proba = model.predict([test_enc, test_bio_scaled], verbose=0).ravel()

            # Select F1-optimal threshold on validation
            f1_thresh, f1_val, mcc_thresh, mcc_val = threshold_curve(val_labels, val_proba)
            selected_threshold = f1_thresh
            print(f"Validation threshold (F1-optimal): {selected_threshold:.6f}, F1={f1_val:.6f}")

            # Evaluate on validation and test with selected threshold
            val_metrics = evaluate_with_threshold(val_labels, val_proba, selected_threshold)
            test_metrics = evaluate_with_threshold(test_labels, test_proba, selected_threshold)

            print(f"Validation metrics: AUROC={val_metrics['AUROC']:.6f}, PRAUC={val_metrics['PRAUC']:.6f}, F1={val_metrics['F1']:.6f}")
            print(f"Test metrics: AUROC={test_metrics['AUROC']:.6f}, PRAUC={test_metrics['PRAUC']:.6f}, F1={test_metrics['F1']:.6f}")

            # Save predictions
            pred_df = pd.DataFrame({
                'source_id': test_df['source_id'],
                'sgRNA': test_df['sgRNA'],
                'off': test_df['off'],
                'true_label': test_labels,
                'probability': test_proba,
                'predicted_label': (test_proba >= selected_threshold).astype(int)
            })
            pred_df.to_csv(os.path.join(predictions_dir, f'fold{fold}_{mode}_predictions.csv'), index=False)

            # Record results
            result = {
                'fold': fold,
                'mode': mode,
                'shared_init_hash': shared_hash,
                'shared_init_hash_reference': shared_init_hash_reference,
                'train_hash': train_hash,
                'val_hash': val_hash,
                'checkpoint_strategy': checkpoint_strategy,
                'selected_epoch': checkpoint_callback.best_epoch_loss if checkpoint_strategy == 'val_loss' else checkpoint_callback.best_epoch_prauc,
            }

            # Add validation metrics
            for k, v in val_metrics.items():
                result[f'val_{k}'] = v

            # Add test metrics
            for k, v in test_metrics.items():
                result[f'test_{k}'] = v

            fold_mode_results.append(result)

            # Clear session
            tf.keras.backend.clear_session()

        fold_results.extend(fold_mode_results)

    # Save per-fold results
    per_fold_df = pd.DataFrame(fold_results)

    # Debug: print columns before adding dataset/feature_set/stage
    print(f"\nDataFrame columns after creation: {list(per_fold_df.columns)}")
    print(f"DataFrame shape: {per_fold_df.shape}")
    if len(per_fold_df) > 0:
        print(f"First row sample: {per_fold_df.iloc[0].to_dict()}")

    per_fold_df['dataset'] = dataset
    per_fold_df['feature_set'] = feature_set
    per_fold_df['stage'] = stage

    # Reorder columns
    cols_order = ['dataset', 'feature_set', 'stage', 'fold', 'mode',
                  'shared_init_hash', 'shared_init_hash_reference',
                  'train_hash', 'val_hash',
                  'checkpoint_strategy', 'selected_epoch', 'threshold'] + \
                 [c for c in per_fold_df.columns if c.startswith('val_') or c.startswith('test_')]
    per_fold_df = per_fold_df[[c for c in cols_order if c in per_fold_df.columns]]

    print(f"\nDataFrame columns after reorder: {list(per_fold_df.columns)}")
    print(f"Has 'fold' column: {'fold' in per_fold_df.columns}")
    print(f"Has 'mode' column: {'mode' in per_fold_df.columns}")

    per_fold_df.to_csv(os.path.join(output_dir, 'E5A_bottleneck_per_fold.csv'), index=False)

    # Debug: check data completeness
    print(f"\nPer-fold results collected: {len(per_fold_df)} rows")
    print(f"Expected: {N_FOLDS * len(E5A_MODES)} rows ({N_FOLDS} folds × {len(E5A_MODES)} modes)")

    # Check if fold and mode columns exist
    if 'fold' not in per_fold_df.columns:
        raise ValueError("Column 'fold' not found in per_fold_df")
    if 'mode' not in per_fold_df.columns:
        raise ValueError("Column 'mode' not found in per_fold_df")

    for fold_num in range(1, N_FOLDS + 1):
        for mode_name in E5A_MODES:
            fold_mask = per_fold_df['fold'] == fold_num
            mode_mask = per_fold_df['mode'] == mode_name
            count = (fold_mask & mode_mask).sum()
            if count != 1:
                print(f"  WARNING: fold {fold_num}, mode {mode_name}: {count} rows (expected 1)")

    # Compute paired differences (bottleneck - control)
    paired_rows = []
    for fold_num in range(1, N_FOLDS + 1):
        fold_mask = per_fold_df['fold'] == fold_num
        control_mask = per_fold_df['mode'] == 'control'
        bottleneck_mask = per_fold_df['mode'] == 'bottleneck'

        control_rows = per_fold_df[fold_mask & control_mask]
        bottleneck_rows = per_fold_df[fold_mask & bottleneck_mask]

        if len(control_rows) == 0 or len(bottleneck_rows) == 0:
            print(f"  WARNING: Skipping fold {fold_num} - missing data (control: {len(control_rows)}, bottleneck: {len(bottleneck_rows)})")
            continue

        control = control_rows.iloc[0]
        bottleneck = bottleneck_rows.iloc[0]

        paired = {
            'fold': fold_num,
            'comparison': 'bottleneck - control'
        }

        for metric in ['threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC', 'TP', 'FP', 'FN', 'TN']:
            for split in ['val', 'test']:
                col = f'{split}_{metric}'
                if col in bottleneck and col in control:
                    paired[f'diff_{split}_{metric}'] = float(bottleneck[col]) - float(control[col])

        paired_rows.append(paired)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(output_dir, 'E5A_bottleneck_paired_differences.csv'), index=False)

    # Compute summary statistics (mean ± std with ddof=0)
    summary_rows = []
    paper = PAPER_REFERENCE.get(dataset, {})

    for mode_name in E5A_MODES:
        mode_rows = per_fold_df[per_fold_df['mode'] == mode_name]
        row = {
            'dataset': dataset,
            'feature_set': feature_set,
            'stage': stage,
            'mode': mode_name,
            'n_folds': N_FOLDS,
            'bottleneck_units': BOTTLENECK_UNITS if mode_name == 'bottleneck' else None,
            'baseline_source': baseline_config['baseline_source']
        }

        # Aggregate metrics
        for metric in ['threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC']:
            for split in ['val', 'test']:
                col = f'{split}_{metric}'
                if col in mode_rows.columns:
                    values = mode_rows[col].astype(float)
                    row[f'{split}_{metric}_Mean'] = float(values.mean())
                    row[f'{split}_{metric}_Std'] = float(values.std(ddof=0))

        # Paper reference
        if mode_name == 'control':
            row['paper_auroc'] = paper.get('auroc', None)
            row['paper_prauc'] = paper.get('prauc', None)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'E5A_bottleneck_summary.csv'), index=False)

    # Save config
    config = {
        'experiment': 'E5A_bottleneck',
        'description': 'Isolate bottleneck Dense layer effect AFTER Flatten',
        'single_factor': 'bottleneck_layer_after_flatten',
        'architecture': {
            'control': '(24, 120) → Flatten → 2880 → Dense(128) → Dense(64) → output',
            'bottleneck': '(24, 120) → Flatten → 2880 → Dense(64, relu) → Dense(128) → Dense(64) → output'
        },
        'parameter_analysis': {
            'control_first_dense': '2880 × 128 + 128 = 368,768 params',
            'bottleneck_first_dense': '2880 × 64 + 64 = 184,384 params',
            'bottleneck_second_dense': '64 × 128 + 128 = 8,320 params',
            'bottleneck_total_before_dense64': '192,704 params',
            'parameter_savings': 'saves ~176K params before final Dense(64)',
            'note': 'Actual total params computed by model.count_params() in parameter_counts field'
        },
        'shared_layers': {
            'cnn_encoder': get_shared_layer_names(stage=stage, cnn_variant=cnn_variant)[:6],  # CNN encoder layers
            'gru_backbone': ['bigru_30', 'bigru_20', 'bigru_10', 'input_reshape', 'concat_input_gru1', 'concat_gru1_gru2', 'concat_all_gru'],
            'biofusion_branch': ['bio_dense', 'bio_norm', 'bio_dropout'] if stage == 'biofusion' else [],
            'not_shared': ['flatten', 'bottleneck_dense', 'dense_128', 'dense_64', 'dropout_head', 'prob_output'],
            'not_shared_reason': 'Head layers have different input shapes (2880 vs 64) between control and bottleneck'
        },
        'dataset': dataset,
        'stage': stage,
        'feature_set': feature_set,
        'cnn_variant': cnn_variant,
        'encoder_winner_source': encoder_winner_source,
        'whether_encoder_is_e4_winner': whether_encoder_is_e4_winner,
        'encoder_selection_metric': baseline_config['encoder_selection_metric'],
        'encoder_selection_value': baseline_config['encoder_selection_value'],
        'encoder_selection_source': baseline_config['encoder_selection_source'],
        'test_metrics_used_for_selection': baseline_config['test_metrics_used_for_selection'],
        'timestamp': timestamp,
        'random_seed': RANDOM_SEED,
        'n_folds': N_FOLDS,
        'epochs': EPOCHS,
        'steps_per_epoch': STEPS_PER_EPOCH,
        'batch_size': BATCH_SIZE,
        'total_batches_per_fold': EPOCHS * STEPS_PER_EPOCH,
        'modes': E5A_MODES,
        'bottleneck_units': BOTTLENECK_UNITS,
        'bottleneck_position': 'AFTER Flatten, before Dense(128)',
        'bottleneck_rationale': 'Aggressive compression 2880→64 to reduce parameters and overfitting risk; forces efficient feature representation',
        'baseline_config': baseline_config,
        'baseline_architecture_source': f'{cnn_variant} encoder + GRU backbone',
        'loss_mode': loss_mode,
        'sampling_mode': sampling_mode,
        'sampling_config': SAMPLING_CONFIGS[sampling_mode],
        'checkpoint_strategy': checkpoint_strategy,
        'threshold_selection': 'F1-optimal on validation',
        'file_hash': file_hash,
        'split_hashes': e5a_split_hashes,
        'fold_sampling_fallbacks': fold_sampling_fallbacks,
        'parameter_counts': param_counts,
        'paper_reference': PAPER_REFERENCE.get(dataset, {}),
        'shared_init_hashes': {
            f'fold_{fold}': shared_init_hash_reference
            for fold in range(1, N_FOLDS + 1)
        }
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"E5-A BOTTLENECK EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - config.json")
    print(f"  - E5A_bottleneck_per_fold.csv")
    print(f"  - E5A_bottleneck_paired_differences.csv")
    print(f"  - E5A_bottleneck_summary.csv")
    print(f"  - splits/")
    print(f"  - checkpoints/{{control,bottleneck}}/")
    print(f"  - histories/")
    print(f"  - predictions/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E5-A: Bottleneck experiment')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['K562', 'HEK293t', 'II5', 'II6'],
                        help='Dataset name')
    parser.add_argument('--stage', type=str, required=True,
                        choices=['p12', 'biofusion'],
                        help='Model stage')
    parser.add_argument('--feature-set', type=str, default=None,
                        choices=['e3a', 'e3b', 'e3c'],
                        help='Biological feature set (auto-detect if not specified)')

    args = parser.parse_args()
    train_e5a_bottleneck(args.dataset, args.stage, args.feature_set)
