#!/usr/bin/env python3
"""
E3v2-C: Checkpoint Strategy Comparison

Compares two checkpoint selection strategies on the same training trajectory:
1. val_loss: Select checkpoint with lowest validation loss
2. val_prauc: Select checkpoint with highest validation PRAUC

Uses E3v2-A/B winner configurations (selected by validation PRAUC):
- K562: focal_bce + lower_positive_hard
- HEK293t: control + lower_positive_hard
- II5: control + lower_positive
- II6: control + lower_positive

Each dataset uses its validated best loss and sampling combination.
5-fold stratified CV with 15% blind test holdout.
Single training trajectory per fold, two checkpoints per trajectory.
Threshold selection on validation set only, applied to blind test once.

Usage:
    python train_e3v2_checkpoint.py --dataset II6 [--feature-set e3a]
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
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve, auc
)

# Path setup matching E3v2-B
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
from MODEL_E3V2_Loss import Crispr_SGRU_BioFusion_Loss

# Constants
RANDOM_SEED = 42
EPOCHS = 35
STEPS_PER_EPOCH = 120
BATCH_SIZE = 256

# E3v2-A/B winners by validation PRAUC (NOT test)
E3V2_WINNERS = {
    'K562': {
        'loss_mode': 'focal_bce',
        'sampling_mode': 'lower_positive_hard',
        'selection_metric': 'validation_prauc_auc',
        'selection_source_loss': 'E3v2-A validation PRAUC results',
        'selection_source_sampling': 'E3v2-B validation PRAUC results'
    },
    'HEK293t': {
        'loss_mode': 'control',
        'sampling_mode': 'lower_positive_hard',
        'selection_metric': 'validation_prauc_auc',
        'selection_source_loss': 'E3v2-A validation PRAUC results',
        'selection_source_sampling': 'E3v2-B validation PRAUC results'
    },
    'II5': {
        'loss_mode': 'control',
        'sampling_mode': 'lower_positive',
        'selection_metric': 'validation_prauc_auc',
        'selection_source_loss': 'E3v2-A validation PRAUC results',
        'selection_source_sampling': 'E3v2-B validation PRAUC results'
    },
    'II6': {
        'loss_mode': 'control',
        'sampling_mode': 'lower_positive',
        'selection_metric': 'validation_prauc_auc',
        'selection_source_loss': 'E3v2-A validation PRAUC results',
        'selection_source_sampling': 'E3v2-B validation PRAUC results'
    },
    'I1': {
        'loss_mode': 'control',
        # Provisional until E3v2-B/I1 validation results are available.
        'sampling_mode': 'lower_positive',
        'selection_metric': 'validation_prauc_auc',
        'selection_source_loss': 'E3v2-A/I1 validation PRAUC results',
        'selection_source_sampling': 'provisional; run E3v2-B/I1 first'
    },
    'I2': {
        'loss_mode': 'control',
        'sampling_mode': 'lower_positive_hard',
        'selection_metric': 'validation_prauc_auc',
        'selection_source_loss': 'E3v2-A/I2 validation PRAUC results',
        'selection_source_sampling': 'E3v2-B/I2 validation PRAUC results'
    }
}
# BE3 remains absent until its E3v2-B validation result is available;
# train_e3v2_checkpoint() fails fast instead of selecting a mode from test data.

# Sampling configurations
SAMPLING_CONFIGS = {
    'p12_control': {'n_pos': 64, 'n_hard': 64, 'n_normal': 128},
    'lower_positive': {'n_pos': 32, 'n_hard': 64, 'n_normal': 160},
    'lower_positive_hard': {'n_pos': 32, 'n_hard': 32, 'n_normal': 192}
}

# Feature set defaults (matching E3v2-A/B)
DATASET_FEATURE_MAP = {
    'II6': 'e3a',
    'HEK293t': 'e3c',
    'II5': 'e3b',
    'BE3': 'e3c',
    'K562': 'e3a',
    'I1': 'e3a',
    'I2': 'e3a'
}

# Paper reference results (S1 table)
PAPER_REFERENCE = {
    'K562': {'auroc': 1.000, 'prauc': 0.941},
    'HEK293t': {'auroc': 0.998, 'prauc': 0.801},
    'BE3': {'auroc': 0.995, 'prauc': 0.523},
    'II5': {'auroc': 0.998, 'prauc': 0.521},
    'II6': {'auroc': 0.996, 'prauc': 0.442},
}

# RNG seed offsets (fixed, matching E3v2-B)
SEED_OFFSETS = {
    'positive': 0,
    'hard': 1000,
    'normal': 2000,
    'shuffle': 3000
}


def load_data(file_path):
    """Load data with source row IDs (matching E3v2-A/B)."""
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
    """Encode sequences (matching E3v2-B)."""
    encode = []
    for _, row in data.iterrows():
        en = Encoder_sgRNA_off.Encoder(on_seq=row[0], off_seq=row[1], with_category=True, label=row[2])
        encode.append(en.on_off_code)
    return np.array(encode)


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


def get_model_weights_hash(model):
    """Get hash of model weights for init verification."""
    weights = model.get_weights()
    combined = np.concatenate([w.flatten() for w in weights])
    return hashlib.md5(combined.tobytes()).hexdigest()


def find_optimal_f1_threshold(y_true, y_probs):
    """
    F1-optimal threshold using cumulative confusion matrix (E0).
    Returns smallest threshold on tie (matching E3v2-A/B).
    """
    sorted_indices = np.argsort(y_probs)
    sorted_probs = y_probs[sorted_indices]
    sorted_labels = y_true[sorted_indices]

    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    cum_pos = np.cumsum(sorted_labels == 1)
    cum_neg = np.cumsum(sorted_labels == 0)

    unique_probs, first_indices = np.unique(sorted_probs, return_index=True)

    candidates = []

    def compute_f1(tp, fp, fn):
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # All-positive
    if len(unique_probs) == 0 or unique_probs[0] > 0.0:
        tp, fp, fn = total_pos, total_neg, 0
        f1 = compute_f1(tp, fp, fn)
        candidates.append((f1, 0.0))

    # Breakpoints
    for i, th in enumerate(unique_probs):
        idx = first_indices[i]
        n_pos_below = int(cum_pos[idx - 1]) if idx > 0 else 0
        n_neg_below_actual = idx - n_pos_below
        fn = n_pos_below
        tp = total_pos - fn
        fp = total_neg - n_neg_below_actual
        f1 = compute_f1(tp, fp, fn)
        candidates.append((f1, float(th)))

    # All-negative
    tp, fp, fn = 0, 0, total_pos
    f1 = compute_f1(tp, fp, fn)
    candidates.append((f1, np.inf))

    best_f1 = max(c[0] for c in candidates)
    best_candidates = [c for c in candidates if c[0] == best_f1]
    best_candidates_sorted = sorted(best_candidates, key=lambda x: x[1])
    best_thresh = best_candidates_sorted[0][1]

    return best_thresh, best_f1


class TrackedSamplingGenerator:
    """
    Generator with coverage tracking for E3v2-C checkpoint comparison.

    Uses the dataset-specific winning sampling configuration.
    Tracks draws for exactly epochs × steps_per_epoch batches.
    """
    def __init__(self, pos_enc, pos_bio, pos_source_ids,
                 bio_hard_enc, bio_hard_bio, bio_hard_source_ids,
                 normal_enc, normal_bio, normal_source_ids,
                 n_pos, n_hard, n_norm,
                 epochs=35, steps_per_epoch=120, seed=42):
        self.pos_enc = pos_enc
        self.pos_bio = pos_bio
        self.pos_source_ids = pos_source_ids

        self.bio_hard_enc = bio_hard_enc if len(bio_hard_enc) > 0 else normal_enc
        self.bio_hard_bio = bio_hard_bio if len(bio_hard_bio) > 0 else normal_bio
        self.bio_hard_source_ids = bio_hard_source_ids if len(bio_hard_source_ids) > 0 else normal_source_ids

        self.normal_enc = normal_enc
        self.normal_bio = normal_bio
        self.normal_source_ids = normal_source_ids

        self.n_pos = n_pos
        self.n_hard = n_hard
        self.n_norm = n_norm
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_batches = epochs * steps_per_epoch

        # Verify batch size
        batch_total = n_pos + n_hard + n_norm
        if batch_total != 256:
            raise ValueError(f"Batch size must be 256, got {batch_total} ({n_pos}+{n_hard}+{n_norm})")

        # Four independent RNG streams (matching E3v2-B)
        self.rng_pos = np.random.RandomState(seed + SEED_OFFSETS['positive'])
        self.rng_hard = np.random.RandomState(seed + SEED_OFFSETS['hard'])
        self.rng_normal = np.random.RandomState(seed + SEED_OFFSETS['normal'])
        self.rng_shuffle = np.random.RandomState(seed + SEED_OFFSETS['shuffle'])

        # Coverage tracking
        self.pos_draws = []
        self.hard_draws = []
        self.normal_draws = []
        self.batch_counter = 0

    def __call__(self):
        """Generator function for tf.data.Dataset.from_generator."""
        while self.batch_counter < self.total_batches:
            # Sample
            pos_idx = self.rng_pos.choice(len(self.pos_enc), self.n_pos, replace=True)
            hard_idx = self.rng_hard.choice(len(self.bio_hard_enc), self.n_hard, replace=True)
            norm_idx = self.rng_normal.choice(len(self.normal_enc), self.n_norm, replace=True)

            # Track draws
            self.pos_draws.extend(self.pos_source_ids[pos_idx].tolist())
            self.hard_draws.extend(self.bio_hard_source_ids[hard_idx].tolist())
            self.normal_draws.extend(self.normal_source_ids[norm_idx].tolist())
            self.batch_counter += 1

            # Gather batch
            b_pos_enc = self.pos_enc[pos_idx]
            b_pos_bio = self.pos_bio[pos_idx]

            b_hard_enc = self.bio_hard_enc[hard_idx]
            b_hard_bio = self.bio_hard_bio[hard_idx]

            b_norm_enc = self.normal_enc[norm_idx]
            b_norm_bio = self.normal_bio[norm_idx]

            X_seq = np.concatenate([b_pos_enc, b_hard_enc, b_norm_enc], axis=0)
            X_bio = np.concatenate([b_pos_bio, b_hard_bio, b_norm_bio], axis=0)
            y_batch = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_norm), dtype=np.float32)

            # Shuffle
            p = self.rng_shuffle.permutation(len(y_batch))

            yield (X_seq[p], X_bio[p]), y_batch[p].reshape(-1, 1)

    def verify_batch_count(self):
        """Verify exact batch count after training."""
        if self.batch_counter != self.total_batches:
            raise AssertionError(
                f"Batch count mismatch: expected {self.total_batches}, got {self.batch_counter}"
            )


class DualCheckpointCallback(Callback):
    """
    Callback to save two checkpoints per training trajectory:
    1. Best val_loss checkpoint (strict minimum, tie keeps earlier epoch)
    2. Best val_prauc checkpoint (>1e-12 improvement required, tie keeps earlier epoch)

    Computes PRAUC on full validation set after each epoch.
    """
    def __init__(self, val_data, val_labels, checkpoint_dir_loss, checkpoint_dir_prauc, fold_num):
        super(DualCheckpointCallback, self).__init__()
        self.val_enc, self.val_bio = val_data
        self.val_labels = val_labels
        self.checkpoint_dir_loss = checkpoint_dir_loss
        self.checkpoint_dir_prauc = checkpoint_dir_prauc
        self.fold_num = fold_num

        # Tracking
        self.best_val_loss = np.inf
        self.best_val_prauc = -np.inf
        self.best_epoch_loss = 0
        self.best_epoch_prauc = 0

        # History
        self.epoch_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', np.inf)

        # Predict on validation set
        y_pred = self.model.predict([self.val_enc, self.val_bio], verbose=0).ravel()
        val_prauc = average_precision_score(self.val_labels, y_pred)

        # Record metrics
        self.epoch_metrics.append({
            'epoch': epoch + 1,
            'val_loss': float(val_loss),
            'val_prauc': float(val_prauc)
        })

        # Save val_loss checkpoint (strict minimum, tie keeps earlier)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch_loss = epoch + 1
            weight_path = os.path.join(self.checkpoint_dir_loss, f'fold{self.fold_num}_best.weights.h5')
            self.model.save_weights(weight_path)

        # Save val_prauc checkpoint (requires >1e-12 improvement, tie keeps earlier)
        if val_prauc > self.best_val_prauc + 1e-12:
            self.best_val_prauc = val_prauc
            self.best_epoch_prauc = epoch + 1
            weight_path = os.path.join(self.checkpoint_dir_prauc, f'fold{self.fold_num}_best.weights.h5')
            self.model.save_weights(weight_path)


def identify_hard_negatives(train_data):
    """
    Identify bio-hard negatives: negatives with ≤3 mismatches in first 20 positions.
    Matching E3v2-A/B hard-negative definition.
    """
    hard_indices = []
    for idx, row in train_data.iterrows():
        if float(row[2]) == 1.0:
            continue
        mm = count_mismatches(row[0], row[1])
        if mm <= 3:
            hard_indices.append(idx)
    return hard_indices


def train_e3v2_checkpoint(dataset_name, feature_set=None):
    """
    Train E3v2-C: Checkpoint strategy comparison.

    Uses E3v2-A/B winners (by validation PRAUC) for each dataset.
    Compares val_loss vs val_prauc checkpoint selection on same trajectory.
    """
    # Feature set is fixed by the E3 configuration for each dataset. An
    # explicit argument is accepted for CLI compatibility but cannot override it.
    expected_feature_set = DATASET_FEATURE_MAP.get(dataset_name)
    if expected_feature_set is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: "
            f"{sorted(DATASET_FEATURE_MAP)}"
        )
    if feature_set is not None and feature_set != expected_feature_set:
        raise ValueError(
            f"Feature set for {dataset_name} is fixed to {expected_feature_set}; "
            f"received {feature_set}."
        )
    feature_set = expected_feature_set

    # Get E3v2-A/B winner configuration
    if dataset_name not in E3V2_WINNERS:
        raise ValueError(
            f"Dataset '{dataset_name}' does not have confirmed E3v2-A/B winner configuration. "
            f"Available datasets: {list(E3V2_WINNERS.keys())}. "
            f"E3v2-C requires validation-confirmed loss and sampling modes."
        )

    winner_config = E3V2_WINNERS[dataset_name]
    loss_mode = winner_config['loss_mode']
    sampling_mode = winner_config['sampling_mode']
    sampling_config = SAMPLING_CONFIGS[sampling_mode]

    print(f"\n{'='*80}")
    print(f"E3v2-C: Checkpoint Strategy Comparison")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Feature set: {feature_set}")
    print(f"Loss: {loss_mode} (from E3v2-A winner by validation PRAUC)")
    print(f"Sampling: {sampling_mode} (from E3v2-B winner by validation PRAUC)")
    print(f"  n_pos={sampling_config['n_pos']}, n_hard={sampling_config['n_hard']}, n_normal={sampling_config['n_normal']}")
    print(f"Checkpoint strategies: val_loss, val_prauc")
    print(f"{'='*80}\n")

    # Load data with source IDs (matching E3v2-A/B)
    data_path = os.path.join(BASE_DIR, 'Datasets', f'{dataset_name}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    Negative, Positive = load_data(data_path)
    print(f"Loaded {len(Positive)} positive, {len(Negative)} negative samples")

    # Compute file hash
    file_hash = compute_file_hash(data_path)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_DIR, 'weights', f'{dataset_name}_E3V2_checkpoint_{feature_set}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    checkpoint_loss_dir = os.path.join(checkpoints_dir, 'val_loss')
    checkpoint_prauc_dir = os.path.join(checkpoints_dir, 'val_prauc')
    os.makedirs(checkpoint_loss_dir, exist_ok=True)
    os.makedirs(checkpoint_prauc_dir, exist_ok=True)

    histories_dir = os.path.join(output_dir, 'histories')
    os.makedirs(histories_dir, exist_ok=True)

    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Configuration
    loss_parameters = {}
    if loss_mode == 'control':
        loss_parameters = {
            'bce_weight': 0.5,
            'dice_weight': 0.5,
            'eps': 1e-7,
            'from_logits': False
        }
    elif loss_mode == 'focal_bce':
        loss_parameters = {
            'gamma': 2.0,
            'alpha': 0.25,
            'from_logits': False
        }
    elif loss_mode == 'weighted_bce':
        loss_parameters = {'pos_weight': 'computed_per_fold'}

    config = {
        'experiment': 'E3v2-C Checkpoint Strategy Comparison',
        'dataset': dataset_name,
        'dataset_file': data_path,
        'dataset_file_hash': file_hash,
        'feature_set': feature_set,
        'selected_loss_mode': loss_mode,
        'selected_sampling_mode': sampling_mode,
        'selection_metric': winner_config['selection_metric'],
        'selection_source_loss': winner_config['selection_source_loss'],
        'selection_source_sampling': winner_config['selection_source_sampling'],
        'winner_selection_prauc_method': 'auc(recall, precision) from sklearn.precision_recall_curve',
        'checkpoint_val_prauc_method': 'sklearn.metrics.average_precision_score',
        'reported_test_prauc_method': 'auc(recall, precision) from sklearn.precision_recall_curve',
        'loss_mode': loss_mode,
        'loss_mode_full_name': {
            'control': 'BCE-Dice (0.5 BCE + 0.5 Soft Dice)',
            'weighted_bce': 'Weighted BCE',
            'focal_bce': 'Focal BCE'
        }.get(loss_mode, loss_mode),
        'loss_parameters': loss_parameters,
        'sampling_mode': sampling_mode,
        'sampling_config': {
            'n_pos': sampling_config['n_pos'],
            'n_hard': sampling_config['n_hard'],
            'n_normal': sampling_config['n_normal'],
            'total_batch_size': BATCH_SIZE
        },
        'checkpoint_strategies': ['val_loss', 'val_prauc'],
        'checkpoint_note': 'Two checkpoints per trajectory: best val_loss and best val_prauc',
        'optimizer': 'Adam',
        'learning_rate': 0.0001,
        'random_seed': RANDOM_SEED,
        'epochs': EPOCHS,
        'steps_per_epoch': STEPS_PER_EPOCH,
        'total_gradient_updates': EPOCHS * STEPS_PER_EPOCH,
        'batch_size': BATCH_SIZE,
        'test_split': 0.15,
        'n_folds': 5,
        'hard_negative_definition': {
            'threshold_mismatches': 3,
            'check_length': 20
        },
        'paper_reference': PAPER_REFERENCE.get(dataset_name, {}),
        'split_hashes': {},
        'fold_seeds': {},
        'fold_init_hashes': {},
        'fallback_used': {},
        'timestamp': timestamp
    }

    # Split matching E3v2-A/B: separate positive/negative split, then shuffle
    pos_train_val, pos_test = train_test_split(
        Positive, test_size=0.15, random_state=RANDOM_SEED
    )
    neg_train_val, neg_test = train_test_split(
        Negative, test_size=0.15, random_state=RANDOM_SEED
    )

    # Combine and shuffle
    train_val_data = pos_train_val + neg_train_val
    test_data = pos_test + neg_test

    random.seed(RANDOM_SEED)
    random.shuffle(train_val_data)
    random.shuffle(test_data)

    # Compute split hashes
    config['split_hashes']['test'] = get_split_hash(test_data)

    print(f"Train+Val: {len(train_val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    print(f"Test hash: {config['split_hashes']['test']}")

    # Convert to DataFrame
    train_val_df = pd.DataFrame(train_val_data, columns=['sgRNA', 'off', 'label', 'source_id'])
    test_df = pd.DataFrame(test_data, columns=['sgRNA', 'off', 'label', 'source_id'])

    # Create splits directory
    splits_dir = os.path.join(output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # Save test split source IDs
    test_split_path = os.path.join(splits_dir, 'test_source_ids.txt')
    with open(test_split_path, 'w') as f:
        for sid in test_df['source_id'].values:
            f.write(f"{sid}\n")

    # Encode test set once
    test_enc = encoding(test_df[['sgRNA', 'off', 'label']]).reshape((-1, 1, 24, 7)).astype("float32")
    test_labels = test_df['label'].astype('float32').values

    # Extract test biological features
    test_data_tuples = [(row[0], row[1], row[2], row[3]) for row in test_data]
    test_bio = extract_features_batch(test_data_tuples, feature_set=feature_set)

    # Prepare cross-validation (matching E3v2-A/B)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Results storage
    results_per_fold = []

    # Training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['label'])):
        print(f"\n{'='*80}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*80}")

        # Split fold
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        # Save fold split source IDs
        train_split_path = os.path.join(splits_dir, f'fold_{fold+1}_train_source_ids.txt')
        val_split_path = os.path.join(splits_dir, f'fold_{fold+1}_val_source_ids.txt')
        with open(train_split_path, 'w') as f:
            for sid in train_df['source_id'].values:
                f.write(f"{sid}\n")
        with open(val_split_path, 'w') as f:
            for sid in val_df['source_id'].values:
                f.write(f"{sid}\n")

        # Compute fold split hashes
        fold_train_hash = get_split_hash([(row['sgRNA'], row['off'], row['label'], row['source_id']) for _, row in train_df.iterrows()])
        fold_val_hash = get_split_hash([(row['sgRNA'], row['off'], row['label'], row['source_id']) for _, row in val_df.iterrows()])

        # Save to config
        config['split_hashes'][f'fold_{fold+1}_train'] = fold_train_hash
        config['split_hashes'][f'fold_{fold+1}_val'] = fold_val_hash

        print(f"Train: {len(train_df)} samples ({int(train_df['label'].astype(float).sum())} pos)")
        print(f"Val: {len(val_df)} samples ({int(val_df['label'].astype(float).sum())} pos)")
        print(f"Train hash: {fold_train_hash}")
        print(f"Val hash: {fold_val_hash}")

        # Identify hard negatives in training set
        # Convert label to float for comparison (label may be string from CSV)
        train_df['label'] = train_df['label'].astype(float)
        val_df['label'] = val_df['label'].astype(float)

        train_neg = train_df[train_df['label'] == 0.0].reset_index(drop=True)
        train_pos = train_df[train_df['label'] == 1.0].reset_index(drop=True)

        # Split negatives into bio-hard and normal
        train_bio_hard_list = []
        train_normal_neg_list = []
        for idx, row in train_neg.iterrows():
            mm = count_mismatches(row['sgRNA'], row['off'])
            if mm <= 3:
                train_bio_hard_list.append(row)
            else:
                train_normal_neg_list.append(row)

        # Fallback if bio_hard is empty (matching E3v2-B)
        fallback_used = False
        if len(train_bio_hard_list) == 0:
            train_bio_hard_list = train_neg.to_dict('records')
            train_normal_neg_list = train_neg.to_dict('records')
            fallback_used = True

        train_bio_hard = pd.DataFrame(train_bio_hard_list) if train_bio_hard_list else pd.DataFrame(columns=train_neg.columns)
        train_normal_neg = pd.DataFrame(train_normal_neg_list) if train_normal_neg_list else pd.DataFrame(columns=train_neg.columns)

        print(f"  Positive: {len(train_pos)}")
        print(f"  Bio-hard: {len(train_bio_hard)}")
        print(f"  Normal neg: {len(train_normal_neg)}")
        if fallback_used:
            print(f"  WARNING: Bio-hard fallback used (no samples with ≤3 mismatches)")

        # Record fallback status
        config['fallback_used'][f'fold_{fold+1}'] = {
            'hard_pool_fallback': fallback_used,
            'hard_pool_size': len(train_bio_hard),
            'normal_pool_size': len(train_normal_neg),
            'note': 'Both hard and normal pools use all negatives' if fallback_used else 'Standard split by mismatch count'
        }

        # Encode sequences
        pos_enc = encoding(train_pos[['sgRNA', 'off', 'label']]).reshape((-1, 1, 24, 7)).astype("float32")
        bio_hard_enc = encoding(train_bio_hard[['sgRNA', 'off', 'label']]).reshape((-1, 1, 24, 7)).astype("float32") if len(train_bio_hard) > 0 else np.empty((0, 1, 24, 7), dtype="float32")
        normal_enc = encoding(train_normal_neg[['sgRNA', 'off', 'label']]).reshape((-1, 1, 24, 7)).astype("float32")
        val_enc = encoding(val_df[['sgRNA', 'off', 'label']]).reshape((-1, 1, 24, 7)).astype("float32")

        # Compute biological features using correct signature
        pos_data_tuples = list(train_pos[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))
        bio_hard_data_tuples = list(train_bio_hard[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None)) if len(train_bio_hard) > 0 else []
        normal_data_tuples = list(train_normal_neg[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))
        val_data_tuples = list(val_df[['sgRNA', 'off', 'label', 'source_id']].itertuples(index=False, name=None))

        if len(pos_data_tuples) == 0:
            raise ValueError(f"Fold {fold+1}: No positive samples in training set")

        pos_bio = extract_features_batch(pos_data_tuples, feature_set=feature_set)
        bio_hard_bio = extract_features_batch(bio_hard_data_tuples, feature_set=feature_set) if len(bio_hard_data_tuples) > 0 else np.empty((0, pos_bio.shape[1]), dtype="float32")
        normal_bio = extract_features_batch(normal_data_tuples, feature_set=feature_set)
        val_bio = extract_features_batch(val_data_tuples, feature_set=feature_set)

        bio_feature_dim = pos_bio.shape[1]

        # Fit StandardScaler on training fold only
        train_all_bio = np.vstack([pos_bio, bio_hard_bio, normal_bio]) if len(bio_hard_bio) > 0 else np.vstack([pos_bio, normal_bio])
        scaler = StandardScaler()
        scaler.fit(train_all_bio)

        pos_bio_scaled = scaler.transform(pos_bio).astype('float32')
        bio_hard_bio_scaled = scaler.transform(bio_hard_bio).astype('float32') if len(bio_hard_bio) > 0 else np.empty((0, bio_feature_dim), dtype='float32')
        normal_bio_scaled = scaler.transform(normal_bio).astype('float32')
        val_bio_scaled = scaler.transform(val_bio).astype('float32')
        test_bio_scaled = scaler.transform(test_bio).astype('float32')

        # Source IDs for coverage tracking
        pos_source_ids = train_pos['source_id'].values
        bio_hard_source_ids = train_bio_hard['source_id'].values if len(train_bio_hard) > 0 else np.array([])
        normal_source_ids = train_normal_neg['source_id'].values

        val_labels = val_df['label'].values.astype('float32')

        # Set seeds for this fold (matching E3v2-B)
        mode_seed = RANDOM_SEED + fold
        random.seed(mode_seed)
        np.random.seed(mode_seed)
        tf.random.set_seed(mode_seed)

        # Record fold seed and split hashes
        config['fold_seeds'][f'fold_{fold+1}'] = {
            'base_seed': mode_seed,
            'pos_rng_seed': mode_seed + SEED_OFFSETS['positive'],
            'hard_rng_seed': mode_seed + SEED_OFFSETS['hard'],
            'normal_rng_seed': mode_seed + SEED_OFFSETS['normal'],
            'shuffle_rng_seed': mode_seed + SEED_OFFSETS['shuffle'],
            'train_hash': fold_train_hash,
            'val_hash': fold_val_hash
        }

        # Build model (matching E3v2-A/B)
        model = Crispr_SGRU_BioFusion_Loss(
            bio_feature_dim=bio_feature_dim,
            loss_mode=loss_mode,
            learning_rate=0.0001
        )

        # Record init hash
        init_hash = get_model_weights_hash(model)
        config['fold_init_hashes'][f'fold_{fold+1}'] = init_hash
        print(f"Model init hash: {init_hash[:16]}")

        # Create tracked generator
        gen_seed = RANDOM_SEED + fold
        train_gen = TrackedSamplingGenerator(
            pos_enc, pos_bio_scaled, pos_source_ids,
            bio_hard_enc, bio_hard_bio_scaled, bio_hard_source_ids,
            normal_enc, normal_bio_scaled, normal_source_ids,
            sampling_config['n_pos'], sampling_config['n_hard'], sampling_config['n_normal'],
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            seed=gen_seed
        )

        # Create tf.data.Dataset from generator
        output_signature = (
            (
                tf.TensorSpec(shape=(None, 1, 24, 7), dtype=tf.float32),
                tf.TensorSpec(shape=(None, bio_feature_dim), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
        train_dataset = tf.data.Dataset.from_generator(
            train_gen,
            output_signature=output_signature
        )

        # Callbacks
        dual_checkpoint = DualCheckpointCallback(
            val_data=(val_enc, val_bio_scaled),
            val_labels=val_labels,
            checkpoint_dir_loss=checkpoint_loss_dir,
            checkpoint_dir_prauc=checkpoint_prauc_dir,
            fold_num=fold+1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0
        )

        # Train (single trajectory)
        print("Training...")
        history = model.fit(
            train_dataset,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            validation_data=([val_enc, val_bio_scaled], val_labels.reshape(-1, 1)),
            callbacks=[dual_checkpoint, reduce_lr],
            verbose=0
        )

        # Verify batch count
        train_gen.verify_batch_count()

        # Save epoch metrics and full history
        epoch_metrics_path = os.path.join(histories_dir, f'fold{fold+1}_epoch_metrics.json')
        with open(epoch_metrics_path, 'w') as f:
            json.dump(dual_checkpoint.epoch_metrics, f, indent=2)

        # Save full history.history for audit
        full_history_path = os.path.join(histories_dir, f'fold{fold+1}_full_history.json')
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(full_history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"\nBest checkpoints:")
        print(f"  val_loss: epoch {dual_checkpoint.best_epoch_loss}, loss={dual_checkpoint.best_val_loss:.6f}")
        print(f"  val_prauc: epoch {dual_checkpoint.best_epoch_prauc}, prauc={dual_checkpoint.best_val_prauc:.6f}")

        # Evaluate both checkpoints
        for strategy in ['val_loss', 'val_prauc']:
            print(f"\n--- Strategy: {strategy} ---")

            # Load checkpoint
            if strategy == 'val_loss':
                weight_path = os.path.join(checkpoint_loss_dir, f'fold{fold+1}_best.weights.h5')
                selected_epoch = dual_checkpoint.best_epoch_loss
                selected_val_loss = dual_checkpoint.best_val_loss
                selected_val_prauc = dual_checkpoint.epoch_metrics[selected_epoch-1]['val_prauc']
            else:  # val_prauc
                weight_path = os.path.join(checkpoint_prauc_dir, f'fold{fold+1}_best.weights.h5')
                selected_epoch = dual_checkpoint.best_epoch_prauc
                selected_val_prauc = dual_checkpoint.best_val_prauc
                selected_val_loss = dual_checkpoint.epoch_metrics[selected_epoch-1]['val_loss']

            model.load_weights(weight_path)

            # Predict on validation set
            val_pred = model.predict([val_enc, val_bio_scaled], verbose=0).ravel()

            # Find F1-optimal threshold on validation (using E0 cumulative method)
            threshold, _ = find_optimal_f1_threshold(val_labels, val_pred)
            print(f"F1-optimal threshold on validation: {threshold:.6f}")

            # Save validation predictions
            val_pred_df = pd.DataFrame({
                'source_id': val_df['source_id'].values,
                'true_label': val_labels,
                'pred_proba': val_pred
            })
            val_pred_path = os.path.join(predictions_dir, f'fold{fold+1}_{strategy}_val_predictions.csv')
            val_pred_df.to_csv(val_pred_path, index=False)

            # Predict on test set (once, with validation-selected threshold)
            test_pred = model.predict([test_enc, test_bio_scaled], verbose=0).ravel()
            test_pred_binary = (test_pred >= threshold).astype(int)

            # Save test predictions
            test_pred_df = pd.DataFrame({
                'source_id': test_df['source_id'].values,
                'true_label': test_labels,
                'pred_proba': test_pred,
                'pred_binary': test_pred_binary
            })
            test_pred_path = os.path.join(predictions_dir, f'fold{fold+1}_{strategy}_test_predictions.csv')
            test_pred_df.to_csv(test_pred_path, index=False)

            # Compute metrics on test set
            test_auroc = roc_auc_score(test_labels, test_pred)

            # PRAUC using precision_recall_curve + auc
            precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_pred)
            test_prauc = auc(recall_vals, precision_vals)

            test_f1 = f1_score(test_labels, test_pred_binary, zero_division=0)
            test_precision = precision_score(test_labels, test_pred_binary, zero_division=0)
            test_recall = recall_score(test_labels, test_pred_binary, zero_division=0)
            test_mcc = matthews_corrcoef(test_labels, test_pred_binary)
            tn, fp, fn, tp = confusion_matrix(test_labels, test_pred_binary).ravel()

            print(f"Test metrics:")
            print(f"  AUROC: {test_auroc:.6f}")
            print(f"  PRAUC: {test_prauc:.6f}")
            print(f"  F1: {test_f1:.6f}")
            print(f"  Precision: {test_precision:.6f}")
            print(f"  Recall: {test_recall:.6f}")
            print(f"  MCC: {test_mcc:.6f}")
            print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

            # Record results
            results_per_fold.append({
                'fold': fold + 1,
                'strategy': strategy,
                'selected_epoch': selected_epoch,
                'best_val_loss': dual_checkpoint.best_val_loss,
                'best_val_prauc': dual_checkpoint.best_val_prauc,
                'selected_val_loss': selected_val_loss,
                'selected_val_prauc': selected_val_prauc,
                'threshold': threshold,
                'AUROC': test_auroc,
                'PRAUC': test_prauc,
                'F1': test_f1,
                'Precision': test_precision,
                'Recall': test_recall,
                'MCC': test_mcc,
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn),
                'TN': int(tn)
            })

        # Clear Keras session
        tf.keras.backend.clear_session()

    # Save per-fold results
    results_df = pd.DataFrame(results_per_fold)
    results_path = os.path.join(output_dir, 'E3V2_checkpoint_per_fold.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nPer-fold results saved to: {results_path}")

    # Compute paired differences (val_prauc - val_loss for each fold)
    paired_diffs = []
    for fold_num in range(1, 6):
        fold_loss = results_df[(results_df['fold'] == fold_num) & (results_df['strategy'] == 'val_loss')].iloc[0]
        fold_prauc = results_df[(results_df['fold'] == fold_num) & (results_df['strategy'] == 'val_prauc')].iloc[0]

        paired_diffs.append({
            'fold': fold_num,
            'diff_selected_epoch': fold_prauc['selected_epoch'] - fold_loss['selected_epoch'],
            'diff_threshold': fold_prauc['threshold'] - fold_loss['threshold'],
            'diff_AUROC': fold_prauc['AUROC'] - fold_loss['AUROC'],
            'diff_PRAUC': fold_prauc['PRAUC'] - fold_loss['PRAUC'],
            'diff_F1': fold_prauc['F1'] - fold_loss['F1'],
            'diff_Precision': fold_prauc['Precision'] - fold_loss['Precision'],
            'diff_Recall': fold_prauc['Recall'] - fold_loss['Recall'],
            'diff_MCC': fold_prauc['MCC'] - fold_loss['MCC'],
            'diff_TP': fold_prauc['TP'] - fold_loss['TP'],
            'diff_FP': fold_prauc['FP'] - fold_loss['FP'],
            'diff_FN': fold_prauc['FN'] - fold_loss['FN'],
            'diff_TN': fold_prauc['TN'] - fold_loss['TN']
        })

    paired_df = pd.DataFrame(paired_diffs)
    paired_path = os.path.join(output_dir, 'E3V2_checkpoint_paired_differences.csv')
    paired_df.to_csv(paired_path, index=False)
    print(f"Paired differences saved to: {paired_path}")

    # Compute summary statistics
    summary_rows = []
    paper_ref = PAPER_REFERENCE.get(dataset_name, {})

    for strategy in ['val_loss', 'val_prauc']:
        strategy_results = results_df[results_df['strategy'] == strategy]

        summary_row = {
            'strategy': strategy,
            'selected_epoch_mean': strategy_results['selected_epoch'].mean(),
            'selected_epoch_std': strategy_results['selected_epoch'].std(ddof=0),
            'best_val_loss_mean': strategy_results['best_val_loss'].mean(),
            'best_val_loss_std': strategy_results['best_val_loss'].std(ddof=0),
            'best_val_prauc_mean': strategy_results['best_val_prauc'].mean(),
            'best_val_prauc_std': strategy_results['best_val_prauc'].std(ddof=0),
            'selected_val_loss_mean': strategy_results['selected_val_loss'].mean(),
            'selected_val_loss_std': strategy_results['selected_val_loss'].std(ddof=0),
            'selected_val_prauc_mean': strategy_results['selected_val_prauc'].mean(),
            'selected_val_prauc_std': strategy_results['selected_val_prauc'].std(ddof=0),
            'threshold_mean': strategy_results['threshold'].mean(),
            'threshold_std': strategy_results['threshold'].std(ddof=0),
            'AUROC_mean': strategy_results['AUROC'].mean(),
            'AUROC_std': strategy_results['AUROC'].std(ddof=0),
            'PRAUC_mean': strategy_results['PRAUC'].mean(),
            'PRAUC_std': strategy_results['PRAUC'].std(ddof=0),
            'F1_mean': strategy_results['F1'].mean(),
            'F1_std': strategy_results['F1'].std(ddof=0),
            'Precision_mean': strategy_results['Precision'].mean(),
            'Precision_std': strategy_results['Precision'].std(ddof=0),
            'Recall_mean': strategy_results['Recall'].mean(),
            'Recall_std': strategy_results['Recall'].std(ddof=0),
            'MCC_mean': strategy_results['MCC'].mean(),
            'MCC_std': strategy_results['MCC'].std(ddof=0),
            'TP_mean': strategy_results['TP'].mean(),
            'TP_std': strategy_results['TP'].std(ddof=0),
            'FP_mean': strategy_results['FP'].mean(),
            'FP_std': strategy_results['FP'].std(ddof=0),
            'FN_mean': strategy_results['FN'].mean(),
            'FN_std': strategy_results['FN'].std(ddof=0),
            'TN_mean': strategy_results['TN'].mean(),
            'TN_std': strategy_results['TN'].std(ddof=0),
            'paper_reference': f'Zhang et al. IJMS 2024 S1 - {dataset_name}' if paper_ref else '',
            'paper_auroc': paper_ref.get('auroc', np.nan),
            'paper_prauc': paper_ref.get('prauc', np.nan)
        }
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'E3V2_checkpoint_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")

    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    print(f"\n{'='*80}")
    print(f"E3v2-C complete!")
    print(f"Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E3v2-C: Checkpoint Strategy Comparison')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., II6, HEK293t)')
    parser.add_argument('--feature-set', default=None, help='Feature set (e3a/e3b/e3c); auto-select if None')

    args = parser.parse_args()
    train_e3v2_checkpoint(args.dataset, args.feature_set)
