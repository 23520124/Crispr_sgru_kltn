#!/usr/bin/env python3
"""
E4: CNN Encoder Comparison

Compares three CNN encoder variants on MODEL12 P12 backbone:
1. p12_control: Original MODEL12 Inception CNN (baseline)
2. residual_cnn: Residual CNN with comparable parameter budget
3. convnext_1d: ConvNeXt-style 1D block (depthwise conv + LayerNorm + GELU)

Two stages:
- p12: sequence-only (single input)
- biofusion: sequence + biological features (two inputs)

All variants use:
- Same initial weights per fold (verified via shared layer hashes)
- Same GRU stack (30/20/10), head, loss, optimizer
- Same sampling, epochs, steps_per_epoch, batch_size
- Same checkpoint strategy, threshold selection on validation
- Blind test prediction only after all selections fixed

Baseline sources:
- K562: E3v2 focal_bce + lower_positive_hard, checkpoint val_prauc (config_only)
- HEK293t: E3v2 control + p12_control sampling, checkpoint val_prauc (config_only)
- II5: E3v2 control + lower_positive, checkpoint val_prauc (config_only)
- II6: E3v2 control + lower_positive, checkpoint val_loss (config_only)

Usage:
    python train_e4_cnn.py --dataset K562 --stage p12 [--feature-set e3a]
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
from MODEL_E4_CNN import (build_e4_cnn_model, count_cnn_parameters,
                          capture_shared_weights, apply_shared_weights,
                          hash_shared_weights, verify_shared_weights_match,
                          get_shared_layer_names, get_shared_weights_hash_from_model)

# Constants
RANDOM_SEED = 42
EPOCHS = 35
STEPS_PER_EPOCH = 120
BATCH_SIZE = 256
N_FOLDS = 5

# CNN variants to compare
CNN_VARIANTS = ['p12_control', 'residual_cnn', 'convnext_1d']
CONVNEXT_KERNEL_SIZE = 7  # 5 or 7

# E4 baseline configurations (config_only - no checkpoint reuse)
# Based on E3v2 winners: loss mode, sampling mode, checkpoint strategy
E4_BASELINES = {
    'K562': {
        'loss_mode': 'focal_bce',
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'sampling_mode': 'lower_positive_hard',
        'checkpoint_strategy': 'val_prauc',
        'baseline_source': 'E3v2-A focal_bce + E3v2-B lower_positive_hard + E3v2-C val_prauc',
        'baseline_reproduction_mode': 'config_only',
        'baseline_note': 'Focal loss (gamma=2.0, alpha=0.25) from E3v2-A winner',
        'reference_config': 'weights/K562_E3V2_checkpoint_e3a_20260925_131725/config.json'
    },
    'HEK293t': {
        'loss_mode': 'control',
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'sampling_mode': 'lower_positive_hard',
        'checkpoint_strategy': 'val_loss',
        'baseline_source': 'E3v2-A control + E3v2-B lower_positive_hard + E3v2-C val_loss',
        'baseline_reproduction_mode': 'config_only',
        'baseline_note': 'BCE-Dice loss, lower_positive_hard sampling, val_loss checkpoint',
        'reference_config': 'weights/HEK293t_E3V2_checkpoint_e3c_20260925_160337/config.json'
    },
    'II5': {
        'loss_mode': 'control',
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'sampling_mode': 'lower_positive',
        'checkpoint_strategy': 'val_prauc',
        'baseline_source': 'E3v2-A control + E3v2-B lower_positive + E3v2-C val_prauc',
        'baseline_reproduction_mode': 'config_only',
        'baseline_note': 'BCE-Dice loss, lower_positive sampling, val_prauc checkpoint',
        'reference_config': 'weights/II5_E3V2_checkpoint_e3b_20260925_133909/config.json'
    },
    'II6': {
        'loss_mode': 'control',
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'sampling_mode': 'lower_positive',
        'checkpoint_strategy': 'val_loss',
        'baseline_source': 'E3v2-A control + E3v2-B lower_positive + E3v2-C val_loss',
        'baseline_reproduction_mode': 'config_only',
        'baseline_note': 'BCE-Dice loss, lower_positive sampling, val_loss checkpoint from E3v2-C winner',
        'reference_config': 'weights/II6_E3V2_checkpoint_e3a_20260925_141116/config.json'
    }
}

# Sampling configurations
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
    Get hash of shared weights (GRU + head, excluding encoder).

    Calls get_shared_weights_hash_from_model for consistency.

    Args:
        model: Keras model
        stage: 'p12' or 'biofusion'

    Returns:
        hash string (16 chars)
    """
    return get_shared_weights_hash_from_model(model, stage)


def get_encoder_weights_hash(model, cnn_variant):
    """Get hash of encoder weights only."""
    encoder_prefixes = {
        'p12_control': 'p12_',
        'residual_cnn': 'res_',
        'convnext_1d': 'cnext_'
    }
    prefix = encoder_prefixes.get(cnn_variant, '')

    encoder_weights = []
    for layer in model.layers:
        if layer.name.startswith(prefix):
            encoder_weights.extend(layer.get_weights())

    if len(encoder_weights) == 0:
        return "no_encoder_weights"

    weights_bytes = b''.join([w.tobytes() for w in encoder_weights])
    return hashlib.md5(weights_bytes).hexdigest()[:16]


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

        self.best_val_loss = float('inf')
        self.best_val_prauc = -float('inf')
        self.best_val_loss_epoch = -1
        self.best_val_prauc_epoch = -1

        self.val_loss_path = os.path.join(checkpoint_dir, 'val_loss', f'fold{fold}_best.weights.h5')
        self.val_prauc_path = os.path.join(checkpoint_dir, 'val_prauc', f'fold{fold}_best.weights.h5')

        os.makedirs(os.path.dirname(self.val_loss_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.val_prauc_path), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')

        # Explicit PRAUC computation
        val_proba = self.model.predict(self.val_data, verbose=0).ravel()
        if np.unique(self.val_labels).size == 2:
            val_prauc = float(average_precision_score(self.val_labels, val_proba))
        else:
            val_prauc = float('nan')

        # Update val_loss checkpoint
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_loss_epoch = epoch + 1
            self.model.save_weights(self.val_loss_path)
            print(f"\n -> Fold {self.fold} epoch {epoch+1}: val_loss checkpoint saved (val_loss={val_loss:.6f})")

        # Update val_prauc checkpoint (tie: keep earlier epoch)
        if not np.isnan(val_prauc):
            if val_prauc > self.best_val_prauc + self.tolerance:
                self.best_val_prauc = val_prauc
                self.best_val_prauc_epoch = epoch + 1
                self.model.save_weights(self.val_prauc_path)
                print(f"\n -> Fold {self.fold} epoch {epoch+1}: val_prauc checkpoint saved (val_prauc={val_prauc:.6f})")


def threshold_curve(y_true, y_proba, tolerance=1e-12):
    """
    E0/E3v2-D cumulative threshold selection with distinct probability breakpoints.

    Returns threshold with highest F1 on validation.
    Ties broken by highest threshold (most conservative).
    Includes candidate thresholds 0.0 and np.inf.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    # Get unique probabilities (distinct breakpoints)
    unique_probs = np.unique(y_proba)
    unique_probs = np.sort(unique_probs)[::-1]  # Descending

    # Add 0.0 and np.inf as candidates
    candidates = [np.inf] + list(unique_probs) + [0.0]

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    best_f1 = -1.0
    best_threshold = 0.5
    curve_records = []

    for thresh in candidates:
        if thresh == np.inf:
            predicted = np.zeros(len(y_true), dtype=np.int32)
        else:
            predicted = (y_proba >= thresh).astype(np.int32)

        tp = np.sum((y_true == 1) & (predicted == 1))
        fp = np.sum((y_true == 0) & (predicted == 1))
        fn = np.sum((y_true == 1) & (predicted == 0))
        tn = np.sum((y_true == 0) & (predicted == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        curve_records.append({
            'threshold': float(thresh) if thresh != np.inf else float('inf'),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'F1': float(f1),
            'Precision': float(precision),
            'Recall': float(recall)
        })

        # Update best (tie: choose highest threshold)
        if f1 > best_f1 + tolerance:
            best_f1 = f1
            best_threshold = thresh
        elif abs(f1 - best_f1) <= tolerance and thresh > best_threshold:
            best_threshold = thresh

    curve_df = pd.DataFrame(curve_records)
    return float(best_threshold) if best_threshold != np.inf else float('inf'), float(best_f1), curve_df


def safe_auc_metrics(y_true, probabilities, threshold):
    """Compute metrics safely."""
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    if np.unique(y_true).size == 2:
        auroc = float(roc_auc_score(y_true, probabilities))
        prauc = float(average_precision_score(y_true, probabilities))
        mcc = float(matthews_corrcoef(y_true, predicted))
    else:
        auroc = float('nan')
        prauc = float('nan')
        mcc = float('nan')

    return {
        'AUROC': auroc,
        'PRAUC': prauc,
        'F1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        'Precision': float(precision_score(y_true, predicted, zero_division=0)),
        'Recall': float(recall_score(y_true, predicted, zero_division=0)),
        'MCC': mcc,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
    }


def train_e4_cnn(dataset, stage, feature_set=None):
    """
    Main E4 CNN training function.

    Args:
        dataset: dataset name (K562, HEK293t, II5, II6)
        stage: 'p12' or 'biofusion'
        feature_set: biological feature set (e3a, e3b, e3c)
    """
    # Validate inputs
    if dataset not in E4_BASELINES:
        raise ValueError(f"Dataset {dataset} not in E4_BASELINES. Available: {list(E4_BASELINES.keys())}\n"
                         f"BE3 is not supported (E3v2-B baseline not available)")

    if stage not in ('p12', 'biofusion'):
        raise ValueError(f"stage must be 'p12' or 'biofusion', got {stage}")

    expected_feature_set = DATASET_FEATURE_MAP.get(dataset)
    if expected_feature_set is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    if feature_set is not None and feature_set != expected_feature_set:
        raise ValueError(f"Feature set for {dataset} is fixed to {expected_feature_set}")
    feature_set = expected_feature_set

    # Get baseline configuration
    baseline_config = E4_BASELINES[dataset]
    loss_mode = baseline_config['loss_mode']
    focal_gamma = baseline_config['focal_gamma']
    focal_alpha = baseline_config['focal_alpha']
    sampling_mode = baseline_config['sampling_mode']
    checkpoint_strategy = baseline_config['checkpoint_strategy']
    reference_config_path = os.path.join(BASE_DIR, baseline_config['reference_config'])

    sampling_config = SAMPLING_CONFIGS[sampling_mode]
    n_pos = sampling_config['n_pos']
    n_hard = sampling_config['n_hard']
    n_normal = sampling_config['n_normal']

    print(f"\n{'='*80}")
    print(f"E4 CNN Encoder Comparison")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Stage: {stage}")
    print(f"Feature set: {feature_set}")
    print(f"Baseline: {baseline_config['baseline_source']}")
    print(f"Reproduction mode: {baseline_config['baseline_reproduction_mode']}")
    print(f"Loss mode: {loss_mode}")
    if loss_mode == 'focal_bce':
        print(f"  Focal gamma: {focal_gamma}, alpha: {focal_alpha}")
    print(f"Sampling: {sampling_mode} (pos={n_pos}, hard={n_hard}, normal={n_normal})")
    print(f"Checkpoint strategy: {checkpoint_strategy}")
    print(f"CNN variants: {CNN_VARIANTS}")
    print(f"ConvNeXt kernel size: {CONVNEXT_KERNEL_SIZE}")
    print(f"Reference config: {reference_config_path}")
    print(f"{'='*80}\n")

    # Load reference split hashes
    reference_split_hashes = None
    if os.path.exists(reference_config_path):
        print(f"Loading reference split hashes from {reference_config_path}...")
        with open(reference_config_path, 'r') as f:
            reference_config_data = json.load(f)
            reference_split_hashes = reference_config_data.get('split_hashes', {})
        print(f"  Reference test hash: {reference_split_hashes.get('test', 'N/A')}")
    else:
        raise FileNotFoundError(f"Reference config not found: {reference_config_path}")

    # Load data
    data_path = os.path.join(BASE_DIR, "Datasets", f"{dataset}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading data from {data_path}...")
    Negative, Positive = load_data(data_path)
    print(f"  Positive: {len(Positive)}, Negative: {len(Negative)}")

    # E3v2 split protocol
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    from sklearn.model_selection import train_test_split

    # Split positive and negative separately
    train_val_pos, test_pos = train_test_split(Positive, test_size=0.15, random_state=RANDOM_SEED)
    train_val_neg, test_neg = train_test_split(Negative, test_size=0.15, random_state=RANDOM_SEED)

    # Combine and shuffle
    train_val_data = train_val_pos + train_val_neg
    test_data = test_pos + test_neg

    random.seed(RANDOM_SEED)
    random.shuffle(train_val_data)
    random.shuffle(test_data)

    print(f"Train+Val: {len(train_val_data)}, Test: {len(test_data)}")

    # Verify test split matches E3v2 reference
    test_hash = get_split_hash(test_data)
    reference_test_hash = reference_split_hashes.get('test')

    print(f"\nSplit verification:")
    print(f"  E4 test hash: {test_hash}")
    print(f"  E3v2 reference test hash: {reference_test_hash}")

    if test_hash != reference_test_hash:
        raise ValueError(f"Test split mismatch!\n"
                         f"E4 test hash: {test_hash}\n"
                         f"E3v2 reference: {reference_test_hash}\n"
                         f"Splits must match E3v2 reference for fair comparison.")

    print(f"  ✓ Test split matches E3v2 reference")

    # Prepare test set
    test_df = pd.DataFrame(test_data, columns=['sgRNA', 'off', 'label', 'source_id'])
    test_enc = encoding(test_df)
    test_labels = test_df['label'].astype('float32').to_numpy()
    test_tuples = list(test_df[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))
    test_bio = extract_features_batch(test_tuples, feature_set=feature_set)

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_DIR, 'weights', f'{dataset}_E4_CNN_{stage}_{feature_set}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

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

    # Count CNN parameters
    cnn_param_counts = {}
    for variant in CNN_VARIANTS:
        params = count_cnn_parameters(variant, convnext_kernel_size=CONVNEXT_KERNEL_SIZE)
        cnn_param_counts[variant] = params
        print(f"\n{variant}:")
        print(f"  Total params: {params['total_params']}")
        print(f"  Trainable params: {params['trainable_params']}")
        print(f"  Output shape: {params['output_shape']}")

    # 5-fold cross-validation
    train_val_df = pd.DataFrame(train_val_data, columns=['sgRNA', 'off', 'label', 'source_id'])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    labels = train_val_df['label'].astype(float)

    # Storage for results
    fold_results = []
    fold_sampling_fallbacks = {}
    e4_split_hashes = {'test': test_hash}

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

        # Verify against E3v2 reference
        reference_train_hash = reference_split_hashes.get(f'fold_{fold}_train')
        reference_val_hash = reference_split_hashes.get(f'fold_{fold}_val')

        print(f"\nFold {fold} split verification:")
        print(f"  E4 train hash: {train_hash}")
        print(f"  E3v2 reference: {reference_train_hash}")
        print(f"  E4 val hash: {val_hash}")
        print(f"  E3v2 reference: {reference_val_hash}")

        if train_hash != reference_train_hash:
            raise ValueError(f"Fold {fold} train split mismatch! E4={train_hash}, E3v2={reference_train_hash}")
        if val_hash != reference_val_hash:
            raise ValueError(f"Fold {fold} val split mismatch! E4={val_hash}, E3v2={reference_val_hash}")

        print(f"  ✓ Fold {fold} splits match E3v2 reference")

        # Store hashes
        e4_split_hashes[f'fold_{fold}_train'] = train_hash
        e4_split_hashes[f'fold_{fold}_val'] = val_hash

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

        # Prepare training data
        train_positive = list(train_pos[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))
        train_negative = list(train_neg[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))

        # Train each CNN variant with shared initial weights
        fold_variant_results = []
        shared_weights_reference = None
        shared_init_hash_reference = None

        for variant in CNN_VARIANTS:
            print(f"\n{'-'*80}")
            print(f"Fold {fold} - Variant: {variant}")
            print(f"{'-'*80}")

            # Set seeds for reproducible initialization
            fold_seed = RANDOM_SEED + fold
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            tf.random.set_seed(fold_seed)

            # Build model
            model = build_e4_cnn_model(
                cnn_variant=variant,
                stage=stage,
                bio_feature_dim=bio_dim,
                loss_mode=loss_mode,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                convnext_kernel_size=CONVNEXT_KERNEL_SIZE,
                learning_rate=0.0001
            )

            # Enforce shared initial weights
            if variant == 'p12_control':
                # First variant: capture immutable shared weights BEFORE training
                print(f"Capturing shared initial weights from p12_control...")
                shared_weights_reference = capture_shared_weights(model, stage)
                shared_init_hash_reference = hash_shared_weights(shared_weights_reference)
                print(f"  ✓ Shared weights captured")
                print(f"  Reference shared init hash: {shared_init_hash_reference}")
            else:
                # Subsequent variants: apply captured shared weights BEFORE training
                print(f"Applying shared initial weights to {variant}...")
                apply_shared_weights(model, shared_weights_reference, stage)
                print(f"  ✓ Shared weights applied")

                # Verify weights match reference
                print(f"Verifying shared weights match reference...")
                verify_shared_weights_match(model, shared_weights_reference, stage)
                print(f"  ✓ Shared weights verified identical to reference")

            # Get init hashes
            shared_hash = get_shared_weights_hash(model, stage)
            encoder_hash = get_encoder_weights_hash(model, variant)
            print(f"Shared init hash: {shared_hash}")
            print(f"Encoder init hash: {encoder_hash}")

            # Verify hash matches reference
            if shared_hash != shared_init_hash_reference:
                raise RuntimeError(f"Fold {fold}, variant {variant}: Shared init hash mismatch!\n"
                                   f"Expected (reference): {shared_init_hash_reference}\n"
                                   f"Got: {shared_hash}\n"
                                   f"Shared weights must be identical across all variants before training.")

            # Setup checkpoint callback
            variant_checkpoint_dir = os.path.join(checkpoints_dir, variant)

            if stage == 'p12':
                val_data_for_callback = val_enc
            else:
                val_data_for_callback = [val_enc, val_bio]

            checkpoint_callback = DualCheckpointCallback(
                fold,
                variant_checkpoint_dir,
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
            if variant == CNN_VARIANTS[0]:  # Record once per fold
                pool_info = generator_obj.get_pool_info()
                fold_sampling_fallbacks[fold] = pool_info

            # Save history
            history_path = os.path.join(histories_dir, f'fold{fold}_{variant}_history.json')
            with open(history_path, 'w') as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

            # Load best checkpoint based on checkpoint_strategy
            best_checkpoint_path = os.path.join(variant_checkpoint_dir, checkpoint_strategy, f'fold{fold}_best.weights.h5')
            if not os.path.exists(best_checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint_path}")

            model.load_weights(best_checkpoint_path)

            # Get selected epoch
            if checkpoint_strategy == 'val_loss':
                selected_epoch = checkpoint_callback.best_val_loss_epoch
            else:
                selected_epoch = checkpoint_callback.best_val_prauc_epoch

            print(f"Loaded checkpoint: {checkpoint_strategy}, epoch {selected_epoch}")

            # Validation prediction
            if stage == 'p12':
                val_proba = model.predict(val_enc, verbose=0).ravel()
            else:
                val_proba = model.predict([val_enc, val_bio], verbose=0).ravel()

            # Threshold selection on validation
            threshold, best_f1, curve_df = threshold_curve(val_labels, val_proba)
            print(f"Validation threshold (F1-optimal): {threshold:.6f}, F1={best_f1:.6f}")

            # Save threshold curve
            curve_df.to_csv(os.path.join(predictions_dir, f'fold{fold}_{variant}_threshold_curve.csv'), index=False)

            # Validation metrics
            val_metrics = safe_auc_metrics(val_labels, val_proba, threshold)
            print(f"Validation metrics: AUROC={val_metrics['AUROC']:.6f}, PRAUC={val_metrics['PRAUC']:.6f}, F1={val_metrics['F1']:.6f}")

            # Test prediction (blind)
            if stage == 'p12':
                test_proba = model.predict(test_enc, verbose=0).ravel()
            else:
                test_proba = model.predict([test_enc, test_bio_scaled], verbose=0).ravel()

            # Test metrics with validation threshold
            test_metrics = safe_auc_metrics(test_labels, test_proba, threshold)
            print(f"Test metrics: AUROC={test_metrics['AUROC']:.6f}, PRAUC={test_metrics['PRAUC']:.6f}, F1={test_metrics['F1']:.6f}")

            # Save predictions
            val_pred_df = pd.DataFrame({
                'source_id': val_df['source_id'],
                'true_label': val_labels,
                'proba': val_proba,
                'pred_binary': (val_proba >= threshold).astype(int)
            })
            val_pred_df.to_csv(os.path.join(predictions_dir, f'fold{fold}_{variant}_val_predictions.csv'), index=False)

            test_pred_df = pd.DataFrame({
                'source_id': test_df['source_id'],
                'true_label': test_labels,
                'proba': test_proba,
                'pred_binary': (test_proba >= threshold).astype(int)
            })
            test_pred_df.to_csv(os.path.join(predictions_dir, f'fold{fold}_{variant}_test_predictions.csv'), index=False)

            # Store results
            fold_variant_results.append({
                'fold': fold,
                'variant': variant,
                'shared_init_hash': shared_hash,
                'shared_init_hash_reference': shared_init_hash_reference,
                'encoder_init_hash': encoder_hash,
                'train_hash': train_hash,
                'val_hash': val_hash,
                'checkpoint_strategy': checkpoint_strategy,
                'selected_epoch': selected_epoch,
                'threshold': threshold,
                'val_AUROC': val_metrics['AUROC'],
                'val_PRAUC': val_metrics['PRAUC'],
                'val_F1': val_metrics['F1'],
                'val_Precision': val_metrics['Precision'],
                'val_Recall': val_metrics['Recall'],
                'val_MCC': val_metrics['MCC'],
                'val_TP': val_metrics['TP'],
                'val_FP': val_metrics['FP'],
                'val_FN': val_metrics['FN'],
                'val_TN': val_metrics['TN'],
                'test_AUROC': test_metrics['AUROC'],
                'test_PRAUC': test_metrics['PRAUC'],
                'test_F1': test_metrics['F1'],
                'test_Precision': test_metrics['Precision'],
                'test_Recall': test_metrics['Recall'],
                'test_MCC': test_metrics['MCC'],
                'test_TP': test_metrics['TP'],
                'test_FP': test_metrics['FP'],
                'test_FN': test_metrics['FN'],
                'test_TN': test_metrics['TN'],
            })

            # Clear session
            tf.keras.backend.clear_session()

        fold_results.extend(fold_variant_results)

    # Save per-fold results
    per_fold_df = pd.DataFrame(fold_results)
    per_fold_df['dataset'] = dataset
    per_fold_df['feature_set'] = feature_set
    per_fold_df['stage'] = stage

    # Reorder columns
    cols_order = ['dataset', 'feature_set', 'stage', 'fold', 'variant',
                  'shared_init_hash', 'shared_init_hash_reference', 'encoder_init_hash',
                  'train_hash', 'val_hash',
                  'checkpoint_strategy', 'selected_epoch', 'threshold'] + \
                 [c for c in per_fold_df.columns if c.startswith('val_') or c.startswith('test_')]
    per_fold_df = per_fold_df[[c for c in cols_order if c in per_fold_df.columns]]

    per_fold_df.to_csv(os.path.join(output_dir, 'E4_CNN_per_fold.csv'), index=False)

    # Compute paired differences (each variant vs p12_control)
    paired_rows = []
    for fold in range(1, N_FOLDS + 1):
        control = per_fold_df[(per_fold_df.fold == fold) & (per_fold_df.variant == 'p12_control')].iloc[0]
        for variant in ['residual_cnn', 'convnext_1d']:
            variant_row = per_fold_df[(per_fold_df.fold == fold) & (per_fold_df.variant == variant)].iloc[0]
            paired = {
                'fold': fold,
                'variant': variant,
                'comparison': f'{variant} - p12_control'
            }
            for metric in ['threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC', 'TP', 'FP', 'FN', 'TN']:
                for split in ['val', 'test']:
                    col = f'{split}_{metric}'
                    if col in variant_row and col in control:
                        paired[f'diff_{split}_{metric}'] = float(variant_row[col]) - float(control[col])
            paired_rows.append(paired)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(output_dir, 'E4_CNN_paired_differences.csv'), index=False)

    # Compute summary statistics (mean ± std with ddof=0)
    summary_rows = []
    paper = PAPER_REFERENCE.get(dataset, {})

    for variant in CNN_VARIANTS:
        variant_rows = per_fold_df[per_fold_df.variant == variant]
        row = {
            'variant': variant,
            'n_folds': len(variant_rows),
            'paper_reference': f'Zhang et al. IJMS 2024 S1 - {dataset}',
            'paper_auroc': paper.get('auroc', np.nan),
            'paper_prauc': paper.get('prauc', np.nan),
        }

        for split in ['val', 'test']:
            for metric in ['threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC', 'TP', 'FP', 'FN', 'TN']:
                col = f'{split}_{metric}'
                if col in variant_rows.columns:
                    values = variant_rows[col].dropna().astype(float)
                    row[f'{col}_mean'] = float(np.mean(values)) if len(values) > 0 else np.nan
                    row[f'{col}_std'] = float(np.std(values, ddof=0)) if len(values) > 0 else np.nan

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'E4_CNN_summary.csv'), index=False)

    # Build shared_init_hashes map for config
    shared_init_hashes_map = {}
    for fold in range(1, N_FOLDS + 1):
        fold_data = per_fold_df[per_fold_df.fold == fold]
        shared_init_hashes_map[f'fold_{fold}'] = {
            'shared_init_hash_reference': fold_data.iloc[0]['shared_init_hash_reference'],
            'variants': {}
        }
        for _, row in fold_data.iterrows():
            shared_init_hashes_map[f'fold_{fold}']['variants'][row['variant']] = {
                'shared_init_hash': row['shared_init_hash'],
                'encoder_init_hash': row['encoder_init_hash']
            }

    # Save configuration
    config = {
        'experiment': 'E4 CNN Encoder Comparison',
        'dataset': dataset,
        'feature_set': feature_set,
        'stage': stage,
        'dataset_file': data_path,
        'dataset_file_hash': compute_file_hash(data_path),
        'baseline_config': baseline_config,
        'baseline_source': baseline_config['baseline_source'],
        'baseline_source_config': reference_config_path,
        'baseline_reproduction_mode': baseline_config['baseline_reproduction_mode'],
        'loss_mode': loss_mode,
        'focal_gamma': focal_gamma if loss_mode == 'focal_bce' else None,
        'focal_alpha': focal_alpha if loss_mode == 'focal_bce' else None,
        'sampling_mode': sampling_mode,
        'sampling_config': sampling_config,
        'sampling_fallbacks': fold_sampling_fallbacks,
        'checkpoint_strategy': checkpoint_strategy,
        'checkpoint_strategy_source': 'predeclared from baseline config',
        'cnn_variants': CNN_VARIANTS,
        'convnext_kernel_size': CONVNEXT_KERNEL_SIZE,
        'cnn_parameter_counts': cnn_param_counts,
        'epochs': EPOCHS,
        'steps_per_epoch': STEPS_PER_EPOCH,
        'batch_size': BATCH_SIZE,
        'learning_rate': 0.0001,
        'optimizer': 'Adam',
        'loss_function': 'BCEDiceLoss (0.5 BCE + 0.5 Soft Dice)' if loss_mode == 'control' else f'FocalBCELoss (gamma={focal_gamma}, alpha={focal_alpha})',
        'n_folds': N_FOLDS,
        'test_size': 0.15,
        'random_seed': RANDOM_SEED,
        'threshold_selection': {
            'method': 'cumulative_probability_breakpoints',
            'criterion': 'highest_F1_on_validation',
            'tie_tolerance': 1e-12,
            'tie_break': 'highest_threshold'
        },
        'metric_definitions': {
            'AUROC': 'sklearn.metrics.roc_auc_score',
            'PRAUC': 'sklearn.metrics.average_precision_score',
            'F1': '2*TP / (2*TP + FP + FN)',
            'MCC': 'sklearn.metrics.matthews_corrcoef'
        },
        'test_selection_allowed': False,
        'split_protocol': 'E3v2 protocol: split pos/neg separately, shuffle with seed 42',
        'split_reference_config': reference_config_path,
        'split_hashes': e4_split_hashes,
        'split_hash_match': True,
        'split_verification': 'All E4 splits verified against E3v2 reference before training',
        'shared_init_verification': True,
        'shared_init_method': 'p12_control initialized first, shared weights captured BEFORE training; residual_cnn and convnext_1d receive identical shared weights (GRU+head) from captured reference; encoder weights remain independent per variant',
        'shared_init_hashes': shared_init_hashes_map,
        'shared_init_layers': {
            'p12': ['bigru_30', 'bigru_20', 'bigru_10', 'head_dense_128', 'head_dense_64', 'head_dropout', 'prob_output'],
            'biofusion': ['bigru_30', 'bigru_20', 'bigru_10', 'seq_dense_128', 'seq_dense_64', 'seq_dropout',
                          'bio_dense', 'bio_norm', 'bio_dropout', 'fusion', 'fusion_dense', 'fusion_dropout', 'prob_output']
        },
        'paper_reference': paper,
        'timestamp': timestamp
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"E4 CNN Comparison Complete")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Per-fold results: E4_CNN_per_fold.csv")
    print(f"Paired differences: E4_CNN_paired_differences.csv")
    print(f"Summary: E4_CNN_summary.csv")
    print(f"Split verification: All splits matched E3v2 reference")
    print(f"Shared init verification: All variants use identical GRU+head initial weights per fold")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E4 CNN Encoder Comparison')
    parser.add_argument('--dataset', required=True, choices=['K562', 'HEK293t', 'II5', 'II6'],
                        help='Dataset name (BE3 not supported - E3v2-B baseline not available)')
    parser.add_argument('--stage', required=True, choices=['p12', 'biofusion'],
                        help='Training stage: p12 (sequence-only) or biofusion (sequence + bio features)')
    parser.add_argument('--feature-set', default=None,
                        help='Biological feature set (e3a, e3b, e3c). Auto-determined if omitted.')

    args = parser.parse_args()

    train_e4_cnn(args.dataset, args.stage, args.feature_set)
