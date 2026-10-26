#!/usr/bin/env python3
"""
E5-D Loss Function Experiment — Isolate effect of loss function

Single factor change: loss function ONLY
- bce: Pure BinaryCrossentropy
- bce_dice: Control (0.5 BCE + 0.5 Soft Dice)
- bce_ranking: BCE + lambda_rank * pairwise ranking loss

All other factors FIXED to E3v2 winner configuration per dataset.
Control 7-channel encoding. E3v2 architecture.
"""

import os
import sys
import argparse
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    auc as sklearn_auc, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, os.path.join(BASE_DIR, "Train"))
sys.path.insert(0, CURRENT_DIR)

from Encoder_sgRNA_off import Encoder
from biological_features import extract_features_batch
from MODEL_E5D_Loss import (
    build_e5d_model, encode_control, copy_all_weights_fair,
    get_shared_layer_hashes, BCELoss, BCEDiceLoss, BCERankingLoss
)

E3V2_CONFIGS = {
    'K562': {
        'stage': 'p12',
        'feature_set': 'e3a',
        'sampling': 'lower_positive_hard',
        'checkpoint_monitor': 'val_prauc'
    },
    'HEK293t': {
        'stage': 'p12',
        'feature_set': 'e3c',
        'sampling': 'lower_positive_hard',
        'checkpoint_monitor': 'val_loss'
    },
    'II5': {
        'stage': 'p12',
        'feature_set': 'e3b',
        'sampling': 'lower_positive',
        'checkpoint_monitor': 'val_prauc'
    },
    'II6': {
        'stage': 'p12',
        'feature_set': 'e3a',
        'sampling': 'lower_positive',
        'checkpoint_monitor': 'val_loss'
    }
}

E5D_REFERENCE_CONFIGS = {
    'K562': {
        'checkpoint_dir': r'C:\Users\LENOVO\Crispr-SGRU\weights\K562_E3V2_checkpoint_e3a_20260925_131725',
        'config_json': r'C:\Users\LENOVO\Crispr-SGRU\weights\K562_E3V2_checkpoint_e3a_20260925_131725\config.json'
    },
    'HEK293t': {
        'checkpoint_dir': r'C:\Users\LENOVO\Crispr-SGRU\weights\HEK293t_E3V2_checkpoint_e3c_20260925_160337',
        'config_json': r'C:\Users\LENOVO\Crispr-SGRU\weights\HEK293t_E3V2_checkpoint_e3c_20260925_160337\config.json'
    },
    'II5': {
        'checkpoint_dir': r'C:\Users\LENOVO\Crispr-SGRU\weights\II5_E3V2_checkpoint_e3b_20260925_133909',
        'config_json': r'C:\Users\LENOVO\Crispr-SGRU\weights\II5_E3V2_checkpoint_e3b_20260925_133909\config.json'
    },
    'II6': {
        'checkpoint_dir': r'C:\Users\LENOVO\Crispr-SGRU\weights\II6_E3V2_checkpoint_e3a_20260925_141116',
        'config_json': r'C:\Users\LENOVO\Crispr-SGRU\weights\II6_E3V2_checkpoint_e3a_20260925_141116\config.json'
    }
}


def compute_hash(indices):
    """
    Compute MD5 hash of sorted indices (first 16 chars).

    Converts string indices to int for consistent hashing with E3v2.
    """
    # Normalize: convert string IDs to int
    try:
        normalized = [int(str(idx).strip()) for idx in indices]
    except ValueError as e:
        raise ValueError(f"Cannot convert source_id to int for hashing: {e}")

    return hashlib.md5(str(sorted(normalized)).encode()).hexdigest()[:16]


def verify_splits_against_reference(fold_idx, train_source_ids, val_source_ids, test_source_ids, reference_config):
    """
    Verify split hashes match reference checkpoint.

    Args:
        fold_idx: 0-4
        train/val/test_source_ids: lists of source_id strings
        reference_config: dict with 'checkpoint_dir' and 'config_json'

    Returns:
        dict: {'train_hash': str, 'val_hash': str, 'test_hash': str}
    """
    fold_num = fold_idx + 1

    # Compute current hashes
    train_hash = compute_hash(train_source_ids)
    val_hash = compute_hash(val_source_ids)
    test_hash = compute_hash(test_source_ids)

    # Load reference config
    with open(reference_config['config_json'], 'r') as f:
        ref_config = json.load(f)

    # Reference uses flat keys: fold_1_train, fold_1_val, ..., test
    ref_train_key = f'fold_{fold_num}_train'
    ref_val_key = f'fold_{fold_num}_val'
    ref_test_key = 'test'

    if ref_train_key not in ref_config['split_hashes']:
        raise KeyError(f"Reference config missing key: {ref_train_key}")
    if ref_val_key not in ref_config['split_hashes']:
        raise KeyError(f"Reference config missing key: {ref_val_key}")
    if ref_test_key not in ref_config['split_hashes']:
        raise KeyError(f"Reference config missing key: {ref_test_key}")

    ref_train_hash = ref_config['split_hashes'][ref_train_key]
    ref_val_hash = ref_config['split_hashes'][ref_val_key]
    ref_test_hash = ref_config['split_hashes'][ref_test_key]

    if train_hash != ref_train_hash:
        raise AssertionError(f"Fold {fold_num} train hash mismatch: {train_hash} != {ref_train_hash}")
    if val_hash != ref_val_hash:
        raise AssertionError(f"Fold {fold_num} val hash mismatch: {val_hash} != {ref_val_hash}")
    if test_hash != ref_test_hash:
        raise AssertionError(f"Test hash mismatch: {test_hash} != {ref_test_hash}")

    print(f"✓ Fold {fold_num} splits verified against reference")

    return {'train_hash': train_hash, 'val_hash': val_hash, 'test_hash': test_hash}


def is_hard_negative(sgrna, offtarget):
    """
    Hard negative: mismatch ≤ 3 in first 20 positions (sgRNA guide region).
    """
    if len(sgrna) < 20 or len(offtarget) < 20:
        return False

    mismatch_count = 0
    for i in range(20):
        sg_char = sgrna[i] if i < len(sgrna) else '-'
        off_char = offtarget[i] if i < len(offtarget) else '-'

        # Count mismatches (not gaps)
        if sg_char != '-' and off_char != '-' and sg_char != '_' and off_char != '_':
            if sg_char != off_char:
                mismatch_count += 1

    return mismatch_count <= 3


def load_data(dataset_name, test_csv_path=None):
    """
    Load dataset CSV without header.

    Returns DataFrame with columns: sgRNA, DNA, Label, source_id
    source_id is generated as string indices matching reference split TXTs.
    """
    if test_csv_path:
        csv_path = test_csv_path
    else:
        data_dir = os.path.join(BASE_DIR, 'Datasets')
        if dataset_name == 'CHANGE-seq':
            csv_path = os.path.join(data_dir, 'CHANGE-seq', 'CHANGE-seq.csv')
        else:
            csv_path = os.path.join(data_dir, f'{dataset_name}.csv')

    # Read CSV without header
    df = pd.read_csv(csv_path, header=None, names=['sgRNA', 'DNA', 'Label'])

    # Generate source_id as string indices (0, 1, 2, ...)
    df['source_id'] = np.arange(len(df), dtype=int).astype(str)

    # Ensure Label is numeric
    df['Label'] = df['Label'].astype(int)

    return df


def load_splits_from_reference(fold_idx, dataset_df, reference_config):
    """
    Load train/val/test splits from reference checkpoint TXT files.

    Args:
        fold_idx: 0-4
        dataset_df: full dataset DataFrame with source_id column
        reference_config: dict with 'checkpoint_dir'

    Returns:
        train_df, val_df, test_df
    """
    fold_num = fold_idx + 1
    ref_split_dir = os.path.join(reference_config['checkpoint_dir'], 'splits')

    # Ensure dataset source_id is string
    dataset_df = dataset_df.copy()
    dataset_df['source_id'] = dataset_df['source_id'].astype(str)

    # Load train source IDs
    train_txt = os.path.join(ref_split_dir, f'fold_{fold_num}_train_source_ids.txt')
    with open(train_txt, 'r') as f:
        train_source_ids = [str(line.strip()) for line in f if line.strip()]

    # Load val source IDs
    val_txt = os.path.join(ref_split_dir, f'fold_{fold_num}_val_source_ids.txt')
    with open(val_txt, 'r') as f:
        val_source_ids = [str(line.strip()) for line in f if line.strip()]

    # Load test source IDs
    test_txt = os.path.join(ref_split_dir, 'test_source_ids.txt')
    with open(test_txt, 'r') as f:
        test_source_ids = [str(line.strip()) for line in f if line.strip()]

    # Create DataFrames
    train_df = dataset_df[dataset_df['source_id'].isin(train_source_ids)].copy()
    val_df = dataset_df[dataset_df['source_id'].isin(val_source_ids)].copy()
    test_df = dataset_df[dataset_df['source_id'].isin(test_source_ids)].copy()

    # Verify splits are not empty and match expected counts
    if len(train_df) == 0:
        raise ValueError(f"Fold {fold_num}: train split is empty")
    if len(val_df) == 0:
        raise ValueError(f"Fold {fold_num}: val split is empty")
    if len(test_df) == 0:
        raise ValueError(f"Test split is empty")

    if len(train_df) != len(train_source_ids):
        raise ValueError(f"Fold {fold_num}: train split has {len(train_df)} rows but TXT has {len(train_source_ids)} IDs")
    if len(val_df) != len(val_source_ids):
        raise ValueError(f"Fold {fold_num}: val split has {len(val_df)} rows but TXT has {len(val_source_ids)} IDs")
    if len(test_df) != len(test_source_ids):
        raise ValueError(f"Test split has {len(test_df)} rows but TXT has {len(test_source_ids)} IDs")

    # Sort by source_id to ensure reproducibility
    train_df = train_df.sort_values('source_id').reset_index(drop=True)
    val_df = val_df.sort_values('source_id').reset_index(drop=True)
    test_df = test_df.sort_values('source_id').reset_index(drop=True)

    # Verify hashes after creating DataFrames
    train_source_ids_final = train_df['source_id'].tolist()
    val_source_ids_final = val_df['source_id'].tolist()
    test_source_ids_final = test_df['source_id'].tolist()

    # Verify against reference config
    verify_splits_against_reference(fold_idx, train_source_ids_final, val_source_ids_final, test_source_ids_final, reference_config)

    return train_df, val_df, test_df


def select_threshold_f1(y_true, y_pred_probs):
    """
    Select F1 threshold using E0 cumulative protocol.

    Returns: (threshold, f1_score)
    """
    # Candidate thresholds: 0.0, distinct probabilities, np.inf
    unique_probs = np.unique(y_pred_probs)
    candidates = np.concatenate([[0.0], unique_probs, [np.inf]])

    best_f1 = -1.0
    best_threshold = 0.5

    for thr in candidates:
        if thr == np.inf:
            y_pred = np.zeros_like(y_true, dtype=int)
        else:
            y_pred = (y_pred_probs >= thr).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0.0)

        # Select highest threshold if F1 equal within tolerance
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_threshold = thr
        elif abs(f1 - best_f1) <= 1e-12 and thr > best_threshold:
            best_threshold = thr

    return best_threshold, best_f1


def train_e5d_one_fold(
    fold_idx,
    train_df,
    val_df,
    test_df,
    stage,
    feature_set,
    sampling_mode,
    checkpoint_monitor,
    lambda_rank,
    run_dir,
    reference_config
):
    """
    Train three loss modes for one fold with fair initialization.

    Returns:
        fold_results: dict with keys 'bce', 'bce_dice', 'bce_ranking'
        split_hashes: dict with train/val/test hashes
    """
    fold_num = fold_idx + 1
    fold_key = f'fold_{fold_num}'

    # Create splits directory (flat structure, not per-fold subdirectories)
    split_dir = os.path.join(run_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    # Copy split TXT files from reference (flat structure)
    ref_split_dir = os.path.join(reference_config['checkpoint_dir'], 'splits')
    for fname in [f'fold_{fold_num}_train_source_ids.txt', f'fold_{fold_num}_val_source_ids.txt', 'test_source_ids.txt']:
        src = os.path.join(ref_split_dir, fname)
        dst = os.path.join(split_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Note: splits already verified in load_splits_from_reference()
    # Extract source_id lists for hash storage
    train_source_ids = train_df['source_id'].tolist()
    val_source_ids = val_df['source_id'].tolist()
    test_source_ids = test_df['source_id'].tolist()

    split_hashes = {
        'train_hash': compute_hash(train_source_ids),
        'val_hash': compute_hash(val_source_ids),
        'test_hash': compute_hash(test_source_ids)
    }

    # Prepare data
    train_sgRNAs = train_df['sgRNA'].values
    train_offs = train_df['DNA'].values
    train_labels = train_df['Label'].astype(np.float32).values

    val_sgRNAs = val_df['sgRNA'].values
    val_offs = val_df['DNA'].values
    val_labels = val_df['Label'].astype(np.float32).values

    test_sgRNAs = test_df['sgRNA'].values
    test_offs = test_df['DNA'].values
    test_labels = test_df['Label'].astype(np.float32).values

    # Bio features if biofusion
    if stage == 'biofusion':
        train_tuples = list(zip(train_sgRNAs, train_offs, train_labels))
        val_tuples = list(zip(val_sgRNAs, val_offs, val_labels))
        test_tuples = list(zip(test_sgRNAs, test_offs, test_labels))
        train_bio = extract_features_batch(train_tuples, feature_set)
        val_bio = extract_features_batch(val_tuples, feature_set)
        test_bio = extract_features_batch(test_tuples, feature_set)

    # Pre-encode all samples (control 7-channel only)
    print(f"[Pre-encoding] Train: {len(train_sgRNAs)} samples...")
    train_X_list = []
    for sg, off in zip(train_sgRNAs, train_offs):
        train_X_list.append(encode_control(sg, off))
    train_X = np.stack(train_X_list, axis=0)  # (N_train, 1, 24, 7)
    assert train_X.shape[1:] == (1, 24, 7), f"Expected train shape (N, 1, 24, 7), got {train_X.shape}"

    print(f"[Pre-encoding] Validation: {len(val_sgRNAs)} samples...")
    val_X_list = []
    for sg, off in zip(val_sgRNAs, val_offs):
        val_X_list.append(encode_control(sg, off))
    val_X = np.stack(val_X_list, axis=0)  # (N_val, 1, 24, 7)
    assert val_X.shape[1:] == (1, 24, 7), f"Expected val shape (N, 1, 24, 7), got {val_X.shape}"

    print(f"[Pre-encoding] Test: {len(test_sgRNAs)} samples...")
    test_X_list = []
    for sg, off in zip(test_sgRNAs, test_offs):
        test_X_list.append(encode_control(sg, off))
    test_X = np.stack(test_X_list, axis=0)  # (N_test, 1, 24, 7)
    assert test_X.shape[1:] == (1, 24, 7), f"Expected test shape (N, 1, 24, 7), got {test_X.shape}"

    print(f"[Pre-encoding] Complete: train {train_X.shape}, val {val_X.shape}, test {test_X.shape}")

    # Prepare validation inputs for fit()
    if stage == 'p12':
        val_inputs = val_X
    else:
        val_inputs = (val_X, val_bio)

    # FAIR INITIALIZATION: Create base model, then copy to three loss modes
    if stage == 'p12':
        bio_dim = 0
    else:
        bio_dim = {'e3a': 25, 'e3b': 40, 'e3c': 44}[feature_set]

    # Build base model (bce_dice as reference)
    model_base = build_e5d_model(stage, 'bce_dice', bio_dim, lambda_rank, 0.0001)
    base_init_hashes = get_shared_layer_hashes(model_base)

    # Build three models and copy all weights from base
    model_bce = build_e5d_model(stage, 'bce', bio_dim, lambda_rank, 0.0001)
    model_bce_dice = build_e5d_model(stage, 'bce_dice', bio_dim, lambda_rank, 0.0001)
    model_bce_ranking = build_e5d_model(stage, 'bce_ranking', bio_dim, lambda_rank, 0.0001)

    # Copy all weights from base
    copy_all_weights_fair(model_base, model_bce)
    copy_all_weights_fair(model_base, model_bce_dice)
    copy_all_weights_fair(model_base, model_bce_ranking)

    # Verify all layer hashes match
    bce_init_hashes = get_shared_layer_hashes(model_bce)
    bce_dice_init_hashes = get_shared_layer_hashes(model_bce_dice)
    bce_ranking_init_hashes = get_shared_layer_hashes(model_bce_ranking)

    # Check all layers match
    all_layer_names = set(base_init_hashes.keys())
    for layer_name in all_layer_names:
        if base_init_hashes[layer_name] != bce_init_hashes[layer_name]:
            raise AssertionError(f"Fair init failed: {layer_name} hash mismatch base vs bce")
        if base_init_hashes[layer_name] != bce_dice_init_hashes[layer_name]:
            raise AssertionError(f"Fair init failed: {layer_name} hash mismatch base vs bce_dice")
        if base_init_hashes[layer_name] != bce_ranking_init_hashes[layer_name]:
            raise AssertionError(f"Fair init failed: {layer_name} hash mismatch base vs bce_ranking")

    print(f"✓ Fair initialization verified: all {len(all_layer_names)} layers have identical weights across three loss modes")

    # Save init metadata
    init_metadata_dir = os.path.join(run_dir, 'loss_metadata')
    os.makedirs(init_metadata_dir, exist_ok=True)

    init_metadata = {
        'base_init_hashes': base_init_hashes,
        'bce_init_hashes': bce_init_hashes,
        'bce_dice_init_hashes': bce_dice_init_hashes,
        'bce_ranking_init_hashes': bce_ranking_init_hashes,
        'all_layers_verified': sorted(all_layer_names),
        'fair_init_verified': True
    }

    init_json_path = os.path.join(init_metadata_dir, f'fold_{fold_num}_initialization.json')
    with open(init_json_path, 'w') as f:
        json.dump(init_metadata, f, indent=2)

    # Sampling config with explicit batch composition (single source of truth)
    if sampling_mode == 'lower_positive':
        sampling_config = {
            'n_pos': 32,
            'n_hard': 64,
            'n_normal': 160,
            'total_batch_size': 256
        }
    elif sampling_mode == 'lower_positive_hard':
        sampling_config = {
            'n_pos': 32,
            'n_hard': 32,
            'n_normal': 192,
            'total_batch_size': 256
        }
    else:  # p12_control
        sampling_config = {
            'n_pos': 64,
            'n_hard': 64,
            'n_normal': 128,
            'total_batch_size': 256
        }

    # Verify total
    assert sampling_config['n_pos'] + sampling_config['n_hard'] + sampling_config['n_normal'] == sampling_config['total_batch_size'], \
        f"Sampling config total mismatch: {sampling_config['n_pos']} + {sampling_config['n_hard']} + {sampling_config['n_normal']} != {sampling_config['total_batch_size']}"

    # Prepare three pools with E3v2 protocol
    n_pos_per_batch = sampling_config['n_pos']
    n_hard_neg_per_batch = sampling_config['n_hard']
    n_normal_neg_per_batch = sampling_config['n_normal']

    # Verify batch composition matches expected mode
    if sampling_mode == 'p12_control':
        assert n_pos_per_batch == 64 and n_hard_neg_per_batch == 64 and n_normal_neg_per_batch == 128, \
            f"p12_control mode requires 64/64/128, got {n_pos_per_batch}/{n_hard_neg_per_batch}/{n_normal_neg_per_batch}"
    elif sampling_mode == 'lower_positive':
        assert n_pos_per_batch == 32 and n_hard_neg_per_batch == 64 and n_normal_neg_per_batch == 160, \
            f"lower_positive mode requires 32/64/160, got {n_pos_per_batch}/{n_hard_neg_per_batch}/{n_normal_neg_per_batch}"
    elif sampling_mode == 'lower_positive_hard':
        assert n_pos_per_batch == 32 and n_hard_neg_per_batch == 32 and n_normal_neg_per_batch == 192, \
            f"lower_positive_hard mode requires 32/32/192, got {n_pos_per_batch}/{n_hard_neg_per_batch}/{n_normal_neg_per_batch}"

    # Prepare three pools
    pos_indices = np.where(train_labels == 1)[0]
    all_neg_indices = np.where(train_labels == 0)[0]

    # Split negatives into hard and normal
    hard_neg_indices = np.array([i for i in all_neg_indices if is_hard_negative(train_sgRNAs[i], train_offs[i])])
    normal_neg_indices = np.array([i for i in all_neg_indices if not is_hard_negative(train_sgRNAs[i], train_offs[i])])

    # Check pools are not empty
    if len(pos_indices) == 0:
        raise ValueError(f"Fold {fold_num}: positive pool is empty")
    if len(hard_neg_indices) == 0:
        raise ValueError(f"Fold {fold_num}: hard-negative pool is empty")
    if len(normal_neg_indices) == 0:
        raise ValueError(f"Fold {fold_num}: normal-negative pool is empty")

    print(f"Pool sizes: pos={len(pos_indices)}, hard_neg={len(hard_neg_indices)}, normal_neg={len(normal_neg_indices)}")

    # Fold-specific RNG seeds (same for all three loss modes within fold)
    seed_pos = 42 + fold_idx
    seed_hard = 1042 + fold_idx
    seed_normal = 2042 + fold_idx
    seed_shuffle = 3042 + fold_idx

    # Calculate total batches and expected draws
    steps_per_epoch = 120
    epochs = 35
    total_batches = steps_per_epoch * epochs  # 4200
    expected_draws = {
        'positive': n_pos_per_batch * total_batches,
        'hard_negative': n_hard_neg_per_batch * total_batches,
        'normal_negative': n_normal_neg_per_batch * total_batches
    }

    # Train three loss modes sequentially with same sampling
    fold_results = {}
    loss_modes = ['bce', 'bce_dice', 'bce_ranking']
    models = {'bce': model_bce, 'bce_dice': model_bce_dice, 'bce_ranking': model_bce_ranking}

    for loss_mode in loss_modes:
        print(f"\n{'='*70}")
        print(f"FOLD {fold_num} — LOSS MODE: {loss_mode.upper()}")
        print(f"{'='*70}\n")

        model = models[loss_mode]

        # Create fresh RNG for this loss mode (same seeds as other modes in same fold)
        rng_pos = np.random.RandomState(seed_pos)
        rng_hard = np.random.RandomState(seed_hard)
        rng_normal = np.random.RandomState(seed_normal)
        rng_shuffle = np.random.RandomState(seed_shuffle)

        # Stateful batch counter for generator
        batch_counter = [0]  # Use list to allow modification in nested function

        def data_generator():
            """Optimized generator using pre-encoded arrays"""
            while batch_counter[0] < total_batches:
                # Sample from three pools with replacement
                batch_pos = rng_pos.choice(pos_indices, n_pos_per_batch, replace=True)
                batch_hard_neg = rng_hard.choice(hard_neg_indices, n_hard_neg_per_batch, replace=True)
                batch_normal_neg = rng_normal.choice(normal_neg_indices, n_normal_neg_per_batch, replace=True)
                batch_indices = np.concatenate([batch_pos, batch_hard_neg, batch_normal_neg])

                # Shuffle batch
                rng_shuffle.shuffle(batch_indices)

                # Index into pre-encoded array (no encoding needed)
                batch_X = train_X[batch_indices]  # (256, 1, 24, 7)
                batch_y = train_labels[batch_indices]

                # Assert shape before yield
                assert batch_X.shape == (256, 1, 24, 7), \
                    f"Expected batch shape (256, 1, 24, 7), got {batch_X.shape}"

                batch_counter[0] += 1

                if stage == 'p12':
                    yield batch_X, batch_y
                else:
                    batch_bio = train_bio[batch_indices]
                    yield (batch_X, batch_bio), batch_y

        if stage == 'p12':
            output_signature = (
                tf.TensorSpec(shape=(None, 1, 24, 7), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        else:
            bio_dim_val = {'e3a': 25, 'e3b': 40, 'e3c': 44}[feature_set]
            output_signature = (
                (
                    tf.TensorSpec(shape=(None, 1, 24, 7), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, bio_dim_val), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )

        train_ds = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        # Create checkpoint callback AFTER validation data is ready
        ckpt_dir = os.path.join(run_dir, 'checkpoints', loss_mode)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'fold_{fold_num}.weights.h5')

        if checkpoint_monitor == 'val_prauc':
            class PRaucCallback(tf.keras.callbacks.Callback):
                def __init__(self, val_data, val_labels, ckpt_path):
                    super().__init__()
                    self.val_data = val_data
                    self.val_labels = val_labels
                    self.ckpt_path = ckpt_path
                    self.best_prauc = -np.inf

                def on_epoch_end(self, epoch, logs=None):
                    val_preds = self.model.predict(self.val_data, batch_size=256, verbose=0)
                    if stage == 'biofusion':
                        val_probs = val_preds.ravel()
                    else:
                        val_probs = val_preds.ravel()

                    prauc = average_precision_score(self.val_labels, val_probs)
                    logs['val_prauc'] = prauc

                    # Save only if improved beyond tolerance (favor earlier epoch)
                    if prauc > self.best_prauc + 1e-12:
                        self.best_prauc = prauc
                        self.model.save_weights(self.ckpt_path)
                        print(f" → val_prauc={prauc:.6f} (saved)")
                    else:
                        print(f" → val_prauc={prauc:.6f}")

            checkpoint_cb = PRaucCallback(val_inputs, val_labels, ckpt_path)
        else:
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )

        # Train
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(val_inputs, val_labels),
            callbacks=[checkpoint_cb],
            verbose=1
        )

        # Assert generator yielded exactly total_batches
        assert batch_counter[0] == total_batches, \
            f"Generator yielded {batch_counter[0]} batches, expected {total_batches}"

        # Save history
        hist_dir = os.path.join(run_dir, 'histories')
        os.makedirs(hist_dir, exist_ok=True)
        hist_path = os.path.join(hist_dir, f'fold_{fold_num}_{loss_mode}_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history.history, f, indent=2)

        # Load best checkpoint
        model.load_weights(ckpt_path)
        print(f"✓ Loaded best checkpoint from: {ckpt_path}")

        # Validation predictions
        val_preds = model.predict(val_inputs, batch_size=256, verbose=0)
        val_probs = val_preds.ravel()

        # Compute validation metrics
        val_auroc = roc_auc_score(val_labels, val_probs)
        val_prauc = average_precision_score(val_labels, val_probs)

        # Select F1 threshold using E0 cumulative protocol
        val_thr_f1, _ = select_threshold_f1(val_labels, val_probs)

        # Recompute validation metrics at selected threshold
        val_pred_f1 = (val_probs >= val_thr_f1 - 1e-12).astype(int)
        val_cm_f1 = confusion_matrix(val_labels, val_pred_f1)
        val_tn_f1, val_fp_f1, val_fn_f1, val_tp_f1 = val_cm_f1.ravel()
        val_precision_f1 = val_tp_f1 / (val_tp_f1 + val_fp_f1) if (val_tp_f1 + val_fp_f1) > 0 else 0.0
        val_recall_f1 = val_tp_f1 / (val_tp_f1 + val_fn_f1) if (val_tp_f1 + val_fn_f1) > 0 else 0.0
        val_f1 = f1_score(val_labels, val_pred_f1, zero_division=0.0)
        val_mcc_f1 = matthews_corrcoef(val_labels, val_pred_f1)

        val_metrics = {
            'auroc': val_auroc,
            'prauc': val_prauc,
            'f1': val_f1,
            'precision': val_precision_f1,
            'recall': val_recall_f1,
            'mcc': val_mcc_f1,
            'threshold_f1': val_thr_f1,
            'tp_f1': int(val_tp_f1),
            'fp_f1': int(val_fp_f1),
            'fn_f1': int(val_fn_f1),
            'tn_f1': int(val_tn_f1)
        }

        print(f"\nValidation metrics:")
        print(f"  AUROC: {val_auroc:.6f}")
        print(f"  PRAUC: {val_prauc:.6f}")
        print(f"  F1: {val_f1:.6f} (threshold={val_thr_f1:.6f})")
        print(f"  Precision: {val_precision_f1:.6f}, Recall: {val_recall_f1:.6f}, MCC: {val_mcc_f1:.6f}")

        # Test predictions (BLIND TEST - only after checkpoint and threshold locked)
        if stage == 'p12':
            test_inputs = test_X
        else:
            test_inputs = (test_X, test_bio)

        test_probs = model.predict(test_inputs, batch_size=256, verbose=0)
        test_probs = test_probs.ravel()

        # Evaluate test with LOCKED thresholds from validation
        test_auroc = roc_auc_score(test_labels, test_probs)
        test_prauc = average_precision_score(test_labels, test_probs)

        # Test F1 metrics with locked threshold
        test_pred_f1 = (test_probs >= val_thr_f1 - 1e-12).astype(int)
        test_cm_f1 = confusion_matrix(test_labels, test_pred_f1)
        test_tn_f1, test_fp_f1, test_fn_f1, test_tp_f1 = test_cm_f1.ravel()
        test_precision_f1 = test_tp_f1 / (test_tp_f1 + test_fp_f1) if (test_tp_f1 + test_fp_f1) > 0 else 0.0
        test_recall_f1 = test_tp_f1 / (test_tp_f1 + test_fn_f1) if (test_tp_f1 + test_fn_f1) > 0 else 0.0
        test_f1 = f1_score(test_labels, test_pred_f1, zero_division=0.0)
        test_mcc_f1 = matthews_corrcoef(test_labels, test_pred_f1)

        test_metrics = {
            'auroc': test_auroc,
            'prauc': test_prauc,
            'f1': test_f1,
            'precision': test_precision_f1,
            'recall': test_recall_f1,
            'mcc': test_mcc_f1,
            'threshold_f1': val_thr_f1,
            'tp_f1': int(test_tp_f1),
            'fp_f1': int(test_fp_f1),
            'fn_f1': int(test_fn_f1),
            'tn_f1': int(test_tn_f1)
        }

        print(f"\nTest metrics (blind test with locked threshold={val_thr_f1:.6f}):")
        print(f"  AUROC: {test_auroc:.6f}")
        print(f"  PRAUC: {test_prauc:.6f}")
        print(f"  F1: {test_f1:.6f}")
        print(f"  Precision: {test_precision_f1:.6f}, Recall: {test_recall_f1:.6f}, MCC: {test_mcc_f1:.6f}")

        # Save predictions
        pred_dir = os.path.join(run_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        val_pred_df = pd.DataFrame({
            'source_id': val_df['source_id'].values,
            'true_label': val_labels,
            'pred_prob': val_probs
        })
        val_pred_df.to_csv(os.path.join(pred_dir, f'fold_{fold_num}_{loss_mode}_val_predictions.csv'), index=False)

        test_pred_df = pd.DataFrame({
            'source_id': test_df['source_id'].values,
            'true_label': test_labels,
            'pred_prob': test_probs
        })
        test_pred_df.to_csv(os.path.join(pred_dir, f'fold_{fold_num}_{loss_mode}_test_predictions.csv'), index=False)

        # Store both validation and test metrics
        fold_results[loss_mode] = {
            'validation': val_metrics,
            'test': test_metrics,
            'sampling_config': sampling_config,
            'actual_draws': {
                'positive': n_pos_per_batch * batch_counter[0],
                'hard_negative': n_hard_neg_per_batch * batch_counter[0],
                'normal_negative': n_normal_neg_per_batch * batch_counter[0],
                'total_batches': batch_counter[0]
            },
            'rng_seeds': {
                'seed_pos': seed_pos,
                'seed_hard': seed_hard,
                'seed_normal': seed_normal,
                'seed_shuffle': seed_shuffle
            },
            'expected_draws': expected_draws
        }

        # Assert expected vs actual draws match
        assert fold_results[loss_mode]['actual_draws']['positive'] == expected_draws['positive'], \
            f"Positive draws mismatch: {fold_results[loss_mode]['actual_draws']['positive']} != {expected_draws['positive']}"
        assert fold_results[loss_mode]['actual_draws']['hard_negative'] == expected_draws['hard_negative'], \
            f"Hard negative draws mismatch: {fold_results[loss_mode]['actual_draws']['hard_negative']} != {expected_draws['hard_negative']}"
        assert fold_results[loss_mode]['actual_draws']['normal_negative'] == expected_draws['normal_negative'], \
            f"Normal negative draws mismatch: {fold_results[loss_mode]['actual_draws']['normal_negative']} != {expected_draws['normal_negative']}"

    # Clear session after all three loss modes complete
    tf.keras.backend.clear_session()

    # Prepare metadata for this fold
    fold_metadata = {
        'init_hashes': {
            'base_init_hashes': base_init_hashes,
            'all_layers_verified': True
        },
        'batch_metadata': {}
    }

    for loss_mode in ['bce', 'bce_dice', 'bce_ranking']:
        fold_metadata['batch_metadata'][loss_mode] = {
            'total_batches': fold_results[loss_mode]['actual_draws']['total_batches'],
            'expected_draws': fold_results[loss_mode]['expected_draws'],
            'actual_draws': {
                'positive': fold_results[loss_mode]['actual_draws']['positive'],
                'hard_negative': fold_results[loss_mode]['actual_draws']['hard_negative'],
                'normal_negative': fold_results[loss_mode]['actual_draws']['normal_negative']
            },
            'rng_seeds': fold_results[loss_mode]['rng_seeds']
        }

    return fold_results, split_hashes, fold_metadata


def generate_five_fold_summary(per_fold_csv_path, output_path):
    """
    Generate summary CSV with mean ± std across 5 folds.

    Args:
        per_fold_csv_path: path to E5D_loss_per_fold.csv
        output_path: path to save E5D_loss_summary.csv
    """
    df = pd.read_csv(per_fold_csv_path)

    # Verify data completeness
    fold_indices = df['fold_idx'].unique()
    loss_modes = df['loss_mode'].unique()
    splits = df['split'].unique()

    if len(fold_indices) != 5:
        raise ValueError(f"Expected 5 folds, got {len(fold_indices)}: {sorted(fold_indices)}")
    if set(loss_modes) != {'bce', 'bce_dice', 'bce_ranking'}:
        raise ValueError(f"Expected loss modes [bce, bce_dice, bce_ranking], got {sorted(loss_modes)}")
    if set(splits) != {'validation', 'test'}:
        raise ValueError(f"Expected splits [validation, test], got {sorted(splits)}")

    # Verify each mode/split has exactly 5 rows
    for loss_mode in loss_modes:
        for split in splits:
            subset = df[(df['loss_mode'] == loss_mode) & (df['split'] == split)]
            if len(subset) != 5:
                raise ValueError(f"Loss mode '{loss_mode}' split '{split}' has {len(subset)} rows, expected 5")

    # Compute mean ± std (ddof=0 for population std)
    summary_rows = []
    for loss_mode in sorted(loss_modes):
        for split in sorted(splits):
            subset = df[(df['loss_mode'] == loss_mode) & (df['split'] == split)]

            row = {
                'loss_mode': loss_mode,
                'split': split,
                'auroc_mean': subset['auroc'].mean(),
                'auroc_std': subset['auroc'].std(ddof=0),
                'prauc_mean': subset['prauc'].mean(),
                'prauc_std': subset['prauc'].std(ddof=0),
                'f1_mean': subset['f1'].mean(),
                'f1_std': subset['f1'].std(ddof=0),
                'precision_mean': subset['precision'].mean(),
                'precision_std': subset['precision'].std(ddof=0),
                'recall_mean': subset['recall'].mean(),
                'recall_std': subset['recall'].std(ddof=0),
                'mcc_mean': subset['mcc'].mean(),
                'mcc_std': subset['mcc'].std(ddof=0)
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Add paper reference and selection note
    summary_df['paper_reference'] = 'Zhang et al., IJMS 2024'
    summary_df['test_metrics_used_for_selection'] = False

    summary_df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"✓ Saved summary: {output_path}")


def generate_paired_differences(per_fold_csv_path, output_path):
    """
    Generate paired differences CSV: bce vs bce_dice, bce_ranking vs bce_dice.

    Control is bce_dice. Compute deltas per fold for paired statistics.
    """
    df = pd.read_csv(per_fold_csv_path)

    # Verify folds
    fold_indices = df['fold_idx'].unique()
    if len(fold_indices) != 5:
        raise ValueError(f"Expected 5 folds, got {len(fold_indices)}")

    # Compute paired differences
    diff_rows = []
    for fold_idx in sorted(fold_indices):
        for split in ['validation', 'test']:
            fold_split_df = df[(df['fold_idx'] == fold_idx) & (df['split'] == split)]

            # Extract metrics for each loss mode
            bce_row = fold_split_df[fold_split_df['loss_mode'] == 'bce'].iloc[0]
            bce_dice_row = fold_split_df[fold_split_df['loss_mode'] == 'bce_dice'].iloc[0]
            bce_ranking_row = fold_split_df[fold_split_df['loss_mode'] == 'bce_ranking'].iloc[0]

            # bce vs bce_dice (control)
            diff_rows.append({
                'fold_idx': fold_idx,
                'split': split,
                'comparison': 'bce_vs_bce_dice',
                'delta_auroc': bce_row['auroc'] - bce_dice_row['auroc'],
                'delta_prauc': bce_row['prauc'] - bce_dice_row['prauc'],
                'delta_f1': bce_row['f1'] - bce_dice_row['f1'],
                'delta_precision': bce_row['precision'] - bce_dice_row['precision'],
                'delta_recall': bce_row['recall'] - bce_dice_row['recall'],
                'delta_mcc': bce_row['mcc'] - bce_dice_row['mcc']
            })

            # bce_ranking vs bce_dice (control)
            diff_rows.append({
                'fold_idx': fold_idx,
                'split': split,
                'comparison': 'bce_ranking_vs_bce_dice',
                'delta_auroc': bce_ranking_row['auroc'] - bce_dice_row['auroc'],
                'delta_prauc': bce_ranking_row['prauc'] - bce_dice_row['prauc'],
                'delta_f1': bce_ranking_row['f1'] - bce_dice_row['f1'],
                'delta_precision': bce_ranking_row['precision'] - bce_dice_row['precision'],
                'delta_recall': bce_ranking_row['recall'] - bce_dice_row['recall'],
                'delta_mcc': bce_ranking_row['mcc'] - bce_dice_row['mcc']
            })

    diff_df = pd.DataFrame(diff_rows)

    # Compute mean and std of deltas (ddof=0 for paired differences)
    summary_rows = []
    for comparison in ['bce_vs_bce_dice', 'bce_ranking_vs_bce_dice']:
        for split in ['validation', 'test']:
            subset = diff_df[(diff_df['comparison'] == comparison) & (diff_df['split'] == split)]

            row = {
                'comparison': comparison,
                'split': split,
                'delta_auroc_mean': subset['delta_auroc'].mean(),
                'delta_auroc_std': subset['delta_auroc'].std(ddof=0),
                'delta_prauc_mean': subset['delta_prauc'].mean(),
                'delta_prauc_std': subset['delta_prauc'].std(ddof=0),
                'delta_f1_mean': subset['delta_f1'].mean(),
                'delta_f1_std': subset['delta_f1'].std(ddof=0),
                'delta_precision_mean': subset['delta_precision'].mean(),
                'delta_precision_std': subset['delta_precision'].std(ddof=0),
                'delta_recall_mean': subset['delta_recall'].mean(),
                'delta_recall_std': subset['delta_recall'].std(ddof=0),
                'delta_mcc_mean': subset['delta_mcc'].mean(),
                'delta_mcc_std': subset['delta_mcc'].std(ddof=0),
                'fold_idx': np.nan  # Summary rows have no fold_idx
            }
            summary_rows.append(row)

    # Append summary to paired differences
    diff_df = pd.concat([diff_df, pd.DataFrame(summary_rows)], ignore_index=True)
    diff_df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"✓ Saved paired differences: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='E5-D Loss Function Experiment')
    parser.add_argument('--dataset', type=str, required=True, choices=['K562', 'HEK293t', 'II5', 'II6'],
                        help='Dataset name')
    parser.add_argument('--stage', type=str, required=True, choices=['p12', 'biofusion'],
                        help='Stage: p12 or biofusion')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index 0-4 (None = run all 5 folds)')
    parser.add_argument('--lambda_rank', type=float, default=0.01,
                        help='Ranking loss weight (default 0.01)')

    args = parser.parse_args()

    # Load E3v2 config for this dataset
    e3v2_config = E3V2_CONFIGS[args.dataset]
    reference_config = E5D_REFERENCE_CONFIGS[args.dataset]

    feature_set = e3v2_config['feature_set']
    sampling_mode = e3v2_config['sampling']
    checkpoint_monitor = e3v2_config['checkpoint_monitor']

    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(BASE_DIR, 'weights',
                           f'{args.dataset}_E5D_loss_{args.stage}_{feature_set}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    df = load_data(args.dataset)

    if args.fold is not None:
        # Run single fold
        fold_idx = args.fold

        # Load splits from reference
        train_df, val_df, test_df = load_splits_from_reference(fold_idx, df, reference_config)

        fold_results, split_hashes, fold_metadata = train_e5d_one_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            stage=args.stage,
            feature_set=feature_set,
            sampling_mode=sampling_mode,
            checkpoint_monitor=checkpoint_monitor,
            lambda_rank=args.lambda_rank,
            run_dir=run_dir,
            reference_config=reference_config
        )

        # Save single-fold config
        fold_num = fold_idx + 1
        config = {
            'experiment': 'E5D_loss',
            'dataset': args.dataset,
            'stage': args.stage,
            'feature_set': feature_set,
            'sampling_mode': sampling_mode,
            'checkpoint_monitor': checkpoint_monitor,
            'lambda_rank': args.lambda_rank,
            'fold_idx': fold_num,  # Use 1-based for consistency
            'reference_checkpoint': reference_config['checkpoint_dir'],
            'split_hashes': {
                f'fold_{fold_num}': split_hashes
            },
            'fold_batch_metadata': {
                f'fold_{fold_num}': fold_metadata
            },
            'timestamp': timestamp,
            'n_folds_completed': 1,
            'paper_reference': 'Zhang et al., IJMS 2024',
            'test_metrics_used_for_selection': False
        }

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Single fold complete. Results saved to: {run_dir}")
        print(f"{'='*70}\n")

    else:
        # Run all 5 folds
        all_fold_results = []
        all_split_hashes = {}
        all_fold_metadata = {}

        for fold_idx in range(5):
            # Load splits from reference
            train_df, val_df, test_df = load_splits_from_reference(fold_idx, df, reference_config)

            fold_results, split_hashes, fold_metadata = train_e5d_one_fold(
                fold_idx=fold_idx,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                stage=args.stage,
                feature_set=feature_set,
                sampling_mode=sampling_mode,
                checkpoint_monitor=checkpoint_monitor,
                lambda_rank=args.lambda_rank,
                run_dir=run_dir,
                reference_config=reference_config
            )

            all_fold_results.append(fold_results)
            all_split_hashes[f'fold_{fold_idx+1}'] = split_hashes
            all_fold_metadata[f'fold_{fold_idx+1}'] = fold_metadata

        # Generate per_fold CSV
        per_fold_rows = []
        for fold_idx, fold_results in enumerate(all_fold_results):
            for loss_mode in ['bce', 'bce_dice', 'bce_ranking']:
                for split in ['validation', 'test']:
                    metrics = fold_results[loss_mode][split]
                    row = {
                        'fold_idx': fold_idx + 1,  # Use 1-5 consistently
                        'loss_mode': loss_mode,
                        'split': split,
                        **metrics
                    }
                    per_fold_rows.append(row)

        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_csv = os.path.join(run_dir, 'E5D_loss_per_fold.csv')
        per_fold_df.to_csv(per_fold_csv, index=False, float_format='%.6f')
        print(f"✓ Saved per-fold results: {per_fold_csv}")

        # Generate paired differences CSV
        paired_diff_csv = os.path.join(run_dir, 'E5D_loss_paired_differences.csv')
        generate_paired_differences(per_fold_csv, paired_diff_csv)

        # Generate summary CSV
        summary_csv = os.path.join(run_dir, 'E5D_loss_summary.csv')
        generate_five_fold_summary(per_fold_csv, summary_csv)

        # Save five-fold config
        config = {
            'experiment': 'E5D_loss',
            'dataset': args.dataset,
            'stage': args.stage,
            'feature_set': feature_set,
            'sampling_mode': sampling_mode,
            'checkpoint_monitor': checkpoint_monitor,
            'lambda_rank': args.lambda_rank,
            'n_folds': 5,
            'reference_checkpoint': reference_config['checkpoint_dir'],
            'split_hashes': all_split_hashes,
            'fold_batch_metadata': all_fold_metadata,
            'timestamp': timestamp,
            'n_folds_completed': 5,
            'paper_reference': 'Zhang et al., IJMS 2024',
            'test_metrics_used_for_selection': False
        }

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*70}")
        print(f"All 5 folds complete. Results saved to: {run_dir}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
