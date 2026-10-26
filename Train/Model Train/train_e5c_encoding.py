#!/usr/bin/env python3
"""
E5-C Encoding Experiment — Isolate effect of augmented encoding

Single factor change: encoding representation ONLY
- Control: 7-channel (5 nucleotide + 2 direction)
- Mask_augmented: 10-channel (7 base + 3 masks: mismatch/indel/PAM)

All other factors FIXED to E3v2 winner configuration per dataset.
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
from MODEL_E5C_Encoding import (
    build_e5c_model, encode_with_masks, copy_cnn_weights_fair,
    get_shared_layer_hashes, BCEDiceLoss, FocalBCELoss
)

E3V2_CONFIGS = {
    'K562': {
        'stage': 'p12',
        'feature_set': 'e3a',
        'loss': 'focal_bce',
        'sampling': 'lower_positive_hard',
        'checkpoint_monitor': 'val_prauc'
    },
    'HEK293t': {
        'stage': 'p12',
        'feature_set': 'e3c',
        'loss': 'control',
        'sampling': 'lower_positive_hard',
        'checkpoint_monitor': 'val_loss'
    },
    'II5': {
        'stage': 'p12',
        'feature_set': 'e3b',
        'loss': 'control',
        'sampling': 'lower_positive',
        'checkpoint_monitor': 'val_prauc'
    },
    'II6': {
        'stage': 'p12',
        'feature_set': 'e3a',
        'loss': 'control',
        'sampling': 'lower_positive',
        'checkpoint_monitor': 'val_loss'
    }
}

E5C_REFERENCE_CONFIGS = {
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
    """Compute MD5 hash of sorted indices (first 16 chars)."""
    return hashlib.md5(str(sorted(indices)).encode()).hexdigest()[:16]


def verify_splits_against_reference(fold_idx, train_source_ids, val_source_ids, test_source_ids, reference_config):
    """
    Verify split hashes match reference checkpoint.

    Args:
        fold_idx: 0-based fold index
        train_source_ids: list of integer source_ids for training
        val_source_ids: list of integer source_ids for validation
        test_source_ids: list of integer source_ids for test

    Returns:
        dict with 'train', 'val', 'test' hashes

    Raises:
        ValueError if any hash does not match reference
    """
    ref_json_path = reference_config['config_json']

    with open(ref_json_path, 'r') as f:
        ref_cfg = json.load(f)

    fold_num = fold_idx + 1

    # Reference uses keys: fold_1_train, fold_1_val, test
    train_key = f'fold_{fold_num}_train'
    val_key = f'fold_{fold_num}_val'
    test_key = 'test'

    expected_train = ref_cfg['split_hashes'][train_key]
    expected_val = ref_cfg['split_hashes'][val_key]
    expected_test = ref_cfg['split_hashes'][test_key]

    train_hash = compute_hash(train_source_ids)
    val_hash = compute_hash(val_source_ids)
    test_hash = compute_hash(test_source_ids)

    if train_hash != expected_train:
        raise ValueError(f"Split hash mismatch for train fold {fold_num}: expected {expected_train}, got {train_hash}")

    if val_hash != expected_val:
        raise ValueError(f"Split hash mismatch for val fold {fold_num}: expected {expected_val}, got {val_hash}")

    if test_hash != expected_test:
        raise ValueError(f"Split hash mismatch for test fold {fold_num}: expected {expected_test}, got {test_hash}")

    return {
        'train': train_hash,
        'val': val_hash,
        'test': test_hash
    }


def is_hard_negative(sgRNA, offtarget):
    """Check if sample is hard negative (mismatch <= 3 in first 20 positions)."""
    count = 0
    for i in range(min(20, len(sgRNA), len(offtarget))):
        if sgRNA[i] != offtarget[i]:
            count += 1
    return count <= 3


def optimal_threshold_for_f1(y_true, y_pred_probs):
    """
    Find optimal threshold for F1 using cumulative precision-recall.

    Thresholds: all unique probabilities + [0.0, np.inf]
    Tolerance: 1e-12
    Tie-breaking: choose highest threshold
    """
    unique_probs = np.unique(y_pred_probs)
    thresholds = np.concatenate([[0.0], unique_probs, [np.inf]])

    best_f1 = 0.0
    best_threshold = 0.5

    for thr in thresholds:
        y_pred_binary = (y_pred_probs >= thr).astype(int)

        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Tie-breaking: prefer higher threshold
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-12 and thr > best_threshold):
            best_f1 = f1
            best_threshold = thr

    return best_threshold, best_f1


def optimal_threshold_for_mcc(y_true, y_pred_probs):
    """Find optimal threshold for MCC using same protocol as F1."""
    unique_probs = np.unique(y_pred_probs)
    thresholds = np.concatenate([[0.0], unique_probs, [np.inf]])

    best_mcc = -1.0
    best_threshold = 0.5

    for thr in thresholds:
        y_pred_binary = (y_pred_probs >= thr).astype(int)

        try:
            mcc = matthews_corrcoef(y_true, y_pred_binary)
        except:
            mcc = 0.0

        if mcc > best_mcc or (abs(mcc - best_mcc) < 1e-12 and thr > best_threshold):
            best_mcc = mcc
            best_threshold = thr

    return best_threshold, best_mcc


def evaluate_fold_predictions(y_true, y_pred_probs, fold_idx, mode):
    """Compute all metrics for one fold."""
    auroc = roc_auc_score(y_true, y_pred_probs)
    prauc = average_precision_score(y_true, y_pred_probs)

    # PR-curve AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_curve_auc = sklearn_auc(recall_vals, precision_vals)

    # Optimal thresholds
    thr_f1, best_f1 = optimal_threshold_for_f1(y_true, y_pred_probs)
    thr_mcc, best_mcc = optimal_threshold_for_mcc(y_true, y_pred_probs)

    # Binary predictions
    y_pred_f1 = (y_pred_probs >= thr_f1).astype(int)
    y_pred_mcc = (y_pred_probs >= thr_mcc).astype(int)

    # Confusion matrices (use labels=[0, 1])
    cm_f1 = confusion_matrix(y_true, y_pred_f1, labels=[0, 1])
    cm_mcc = confusion_matrix(y_true, y_pred_mcc, labels=[0, 1])

    tn_f1, fp_f1, fn_f1, tp_f1 = cm_f1.ravel()
    tn_mcc, fp_mcc, fn_mcc, tp_mcc = cm_mcc.ravel()

    prec_f1 = precision_score(y_true, y_pred_f1, zero_division=0.0)
    rec_f1 = recall_score(y_true, y_pred_f1, zero_division=0.0)

    prec_mcc = precision_score(y_true, y_pred_mcc, zero_division=0.0)
    rec_mcc = recall_score(y_true, y_pred_mcc, zero_division=0.0)

    return {
        'fold_idx': fold_idx,
        'mode': mode,
        'auroc': auroc,
        'prauc': prauc,
        'pr_curve_auc': pr_curve_auc,
        'threshold_f1': thr_f1,
        'f1': best_f1,
        'precision_f1': prec_f1,
        'recall_f1': rec_f1,
        'tp_f1': int(tp_f1),
        'fp_f1': int(fp_f1),
        'tn_f1': int(tn_f1),
        'fn_f1': int(fn_f1),
        'threshold_mcc': thr_mcc,
        'mcc': best_mcc,
        'precision_mcc': prec_mcc,
        'recall_mcc': rec_mcc,
        'tp_mcc': int(tp_mcc),
        'fp_mcc': int(fp_mcc),
        'tn_mcc': int(tn_mcc),
        'fn_mcc': int(fn_mcc)
    }


def train_e5c_one_fold(
    dataset_name, fold_idx, train_df, val_df, test_df,
    stage, feature_set, loss_mode, sampling_mode, sampling_config,
    checkpoint_monitor, reference_config, run_dir
):
    """
    Train one fold for BOTH control AND mask_augmented modes.

    REQUIREMENT 2: run_dir is passed from main().
    REQUIREMENT 5: Clear session AFTER both modes are done.
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

    # Verify splits (must use source_id, not DataFrame index)
    train_source_ids = train_df['source_id'].tolist()
    val_source_ids = val_df['source_id'].tolist()
    test_source_ids = test_df['source_id'].tolist()

    split_hashes = verify_splits_against_reference(fold_idx, train_source_ids, val_source_ids, test_source_ids, reference_config)

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

    # PRE-ENCODE ALL DATA ONCE (CPU optimization)
    print(f"\n[Pre-encoding] Encoding {len(train_sgRNAs)} train samples (control + mask_augmented)...")

    # Train control encoding
    train_X_ctrl_list = []
    for sg, off in zip(train_sgRNAs, train_offs):
        enc = encode_with_masks(sg, off, 'control')  # (1, 24, 7)
        train_X_ctrl_list.append(enc[0])  # (24, 7)
    train_X_control = np.stack(train_X_ctrl_list, axis=0)  # (N_train, 24, 7)
    train_X_control = train_X_control[:, np.newaxis, :, :]  # (N_train, 1, 24, 7)
    assert train_X_control.shape == (len(train_sgRNAs), 1, 24, 7), \
        f"Expected train control shape ({len(train_sgRNAs)}, 1, 24, 7), got {train_X_control.shape}"

    # Train mask_augmented encoding
    train_X_aug_list = []
    for sg, off in zip(train_sgRNAs, train_offs):
        enc = encode_with_masks(sg, off, 'mask_augmented')  # (1, 24, 10)
        train_X_aug_list.append(enc[0])  # (24, 10)
    train_X_mask_augmented = np.stack(train_X_aug_list, axis=0)  # (N_train, 24, 10)
    train_X_mask_augmented = train_X_mask_augmented[:, np.newaxis, :, :]  # (N_train, 1, 24, 10)
    assert train_X_mask_augmented.shape == (len(train_sgRNAs), 1, 24, 10), \
        f"Expected train mask_augmented shape ({len(train_sgRNAs)}, 1, 24, 10), got {train_X_mask_augmented.shape}"

    print(f"[Pre-encoding] Train: control {train_X_control.shape}, mask_augmented {train_X_mask_augmented.shape}")

    # Validation control encoding
    print(f"[Pre-encoding] Encoding {len(val_sgRNAs)} validation samples...")
    val_X_list = []
    for sg, off in zip(val_sgRNAs, val_offs):
        enc = encode_with_masks(sg, off, 'control')  # (1, 24, 7)
        val_X_list.append(enc[0])  # Extract (24, 7)
    val_X_ctrl = np.stack(val_X_list, axis=0)  # (N_val, 24, 7)
    val_X_ctrl = val_X_ctrl[:, np.newaxis, :, :]  # (N_val, 1, 24, 7)
    assert val_X_ctrl.shape[1:] == (1, 24, 7), f"Expected control val shape (N, 1, 24, 7), got {val_X_ctrl.shape}"

    # Validation mask_augmented encoding
    val_X_aug_list = []
    for sg, off in zip(val_sgRNAs, val_offs):
        enc = encode_with_masks(sg, off, 'mask_augmented')  # (1, 24, 10)
        val_X_aug_list.append(enc[0])  # (24, 10)
    val_X_mask_aug = np.stack(val_X_aug_list, axis=0)  # (N_val, 24, 10)
    val_X_mask_aug = val_X_mask_aug[:, np.newaxis, :, :]  # (N_val, 1, 24, 10)
    assert val_X_mask_aug.shape[1:] == (1, 24, 10), f"Expected mask_augmented val shape (N, 1, 24, 10), got {val_X_mask_aug.shape}"

    print(f"[Pre-encoding] Validation: control {val_X_ctrl.shape}, mask_augmented {val_X_mask_aug.shape}")

    # Test control encoding
    print(f"[Pre-encoding] Encoding {len(test_sgRNAs)} test samples...")
    test_X_ctrl_list = []
    for sg, off in zip(test_sgRNAs, test_offs):
        enc = encode_with_masks(sg, off, 'control')  # (1, 24, 7)
        test_X_ctrl_list.append(enc[0])  # (24, 7)
    test_X_control = np.stack(test_X_ctrl_list, axis=0)  # (N_test, 24, 7)
    test_X_control = test_X_control[:, np.newaxis, :, :]  # (N_test, 1, 24, 7)
    assert test_X_control.shape[1:] == (1, 24, 7), f"Expected test control shape (N, 1, 24, 7), got {test_X_control.shape}"

    # Test mask_augmented encoding
    test_X_aug_list = []
    for sg, off in zip(test_sgRNAs, test_offs):
        enc = encode_with_masks(sg, off, 'mask_augmented')  # (1, 24, 10)
        test_X_aug_list.append(enc[0])  # (24, 10)
    test_X_mask_augmented = np.stack(test_X_aug_list, axis=0)  # (N_test, 24, 10)
    test_X_mask_augmented = test_X_mask_augmented[:, np.newaxis, :, :]  # (N_test, 1, 24, 10)
    assert test_X_mask_augmented.shape[1:] == (1, 24, 10), f"Expected test mask_augmented shape (N, 1, 24, 10), got {test_X_mask_augmented.shape}"

    print(f"[Pre-encoding] Test: control {test_X_control.shape}, mask_augmented {test_X_mask_augmented.shape}")

    # Prepare validation inputs for fit()
    if stage == 'p12':
        val_inputs_ctrl = val_X_ctrl
        val_inputs_mask_aug = val_X_mask_aug
    else:
        val_inputs_ctrl = (val_X_ctrl, val_bio)
        val_inputs_mask_aug = (val_X_mask_aug, val_bio)

    # FAIR INITIALIZATION: Create both models at start before training loop
    if stage == 'p12':
        bio_dim = 0
    else:
        bio_dim = {'e3a': 25, 'e3b': 40, 'e3c': 44}[feature_set]

    # Build both models
    model_control = build_e5c_model('control', stage, loss_mode, bio_dim, 2.0, 0.25, 0.0001)
    model_mask_augmented = build_e5c_model('mask_augmented', stage, loss_mode, bio_dim, 2.0, 0.25, 0.0001)

    # Copy shared CNN initialization from control to mask_augmented
    copy_cnn_weights_fair(model_control, model_mask_augmented)

    # Verify shared-layer hashes match after fair initialization
    control_init_hashes = get_shared_layer_hashes(model_control)
    mask_augmented_init_hashes = get_shared_layer_hashes(model_mask_augmented)

    # Identify actual shared layers (present in both models with same hash)
    shared_layers = []
    for layer_name in control_init_hashes.keys():
        if layer_name in mask_augmented_init_hashes:
            if control_init_hashes[layer_name] == mask_augmented_init_hashes[layer_name]:
                shared_layers.append(layer_name)
            else:
                raise AssertionError(f"Fair init failed: shared layer '{layer_name}' has different weights. "
                                     f"Control: {control_init_hashes[layer_name]}, "
                                     f"Mask_augmented: {mask_augmented_init_hashes[layer_name]}")

    print(f"✓ Fair initialization verified: {len(shared_layers)} shared layers have identical weights")

    # Identify independent layers (only in one model or different shape)
    independent_layers = {
        'control_only': [],
        'mask_augmented_only': []
    }

    # Get all layer names from both models
    all_control_layers = [layer.name for layer in model_control.layers]
    all_mask_augmented_layers = [layer.name for layer in model_mask_augmented.layers]

    for layer_name in all_control_layers:
        if layer_name not in all_mask_augmented_layers:
            independent_layers['control_only'].append(layer_name)
        elif layer_name not in shared_layers and layer_name in control_init_hashes:
            # Present in both but not in shared_layers (different shape or not in hash list)
            independent_layers['control_only'].append(layer_name)

    for layer_name in all_mask_augmented_layers:
        if layer_name not in all_control_layers:
            independent_layers['mask_augmented_only'].append(layer_name)
        elif layer_name not in shared_layers and layer_name in mask_augmented_init_hashes:
            # Present in both but not in shared_layers (different shape or not in hash list)
            independent_layers['mask_augmented_only'].append(layer_name)

    # Save init metadata
    init_metadata_dir = os.path.join(run_dir, 'encoding_metadata')
    os.makedirs(init_metadata_dir, exist_ok=True)

    init_metadata = {
        'control_init_hashes': control_init_hashes,
        'mask_augmented_init_hashes': mask_augmented_init_hashes,
        'shared_layers': sorted(shared_layers),
        'independent_layers': independent_layers,
        'fair_init_verified': True
    }

    init_json_path = os.path.join(init_metadata_dir, f'fold_{fold_num}_initialization.json')
    with open(init_json_path, 'w') as f:
        json.dump(init_metadata, f, indent=2)

    # Train BOTH modes
    fold_results = {}
    models = {'control': model_control, 'mask_augmented': model_mask_augmented}

    for mode in ['control', 'mask_augmented']:
        print(f"\n{'='*70}")
        print(f"FOLD {fold_num} — MODE: {mode.upper()}")
        print(f"{'='*70}")

        # Use pre-built model
        model = models[mode]

        # Checkpoint directory
        ckpt_dir = os.path.join(run_dir, 'checkpoints', mode, fold_key)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'best_model.h5')

        # Generator with per-batch balanced sampling (E3v2 protocol)
        # Use sampling_config as single source of truth
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

        # Verify all three pools are non-empty
        if len(pos_indices) == 0:
            raise ValueError(f"Fold {fold_num}: positive pool is empty")
        if len(hard_neg_indices) == 0:
            raise ValueError(f"Fold {fold_num}: hard-negative pool is empty")
        if len(normal_neg_indices) == 0:
            raise ValueError(f"Fold {fold_num}: normal-negative pool is empty")

        # Fold-specific RNG seeds (same for both encoding modes within fold)
        seed_pos = 42 + fold_idx
        seed_hard = 1042 + fold_idx
        seed_normal = 2042 + fold_idx
        seed_shuffle = 3042 + fold_idx

        rng_pos = np.random.RandomState(seed_pos)
        rng_hard = np.random.RandomState(seed_hard)
        rng_normal = np.random.RandomState(seed_normal)
        rng_shuffle = np.random.RandomState(seed_shuffle)

        # Calculate total batches and expected draws
        steps_per_epoch = 120
        epochs = 35
        total_batches = steps_per_epoch * epochs  # 4200
        expected_draws = {
            'positive': n_pos_per_batch * total_batches,
            'hard_negative': n_hard_neg_per_batch * total_batches,
            'normal_negative': n_normal_neg_per_batch * total_batches
        }

        # Stateful batch counter for generator
        batch_counter = [0]  # Use list to allow modification in nested function

        def data_generator():
            """Optimized generator using pre-encoded arrays"""
            # Select pre-encoded array for current mode
            if mode == 'control':
                train_X_encoded = train_X_control
            else:
                train_X_encoded = train_X_mask_augmented

            while batch_counter[0] < total_batches:
                # Sample from three pools with replacement
                batch_pos = rng_pos.choice(pos_indices, n_pos_per_batch, replace=True)
                batch_hard_neg = rng_hard.choice(hard_neg_indices, n_hard_neg_per_batch, replace=True)
                batch_normal_neg = rng_normal.choice(normal_neg_indices, n_normal_neg_per_batch, replace=True)
                batch_indices = np.concatenate([batch_pos, batch_hard_neg, batch_normal_neg])

                # Shuffle batch
                rng_shuffle.shuffle(batch_indices)

                # Index into pre-encoded array (no encoding needed)
                batch_X = train_X_encoded[batch_indices]  # (256, 1, 24, channels)
                batch_y = train_labels[batch_indices]

                # Assert shape before yield
                expected_channels = 7 if mode == 'control' else 10
                assert batch_X.shape == (256, 1, 24, expected_channels), \
                    f"Expected batch shape (256, 1, 24, {expected_channels}), got {batch_X.shape}"

                batch_counter[0] += 1

                if stage == 'p12':
                    yield batch_X, batch_y
                else:
                    batch_bio = train_bio[batch_indices]
                    yield (batch_X, batch_bio), batch_y

        if stage == 'p12':
            output_signature = (
                tf.TensorSpec(shape=(None, 1, 24, 7 if mode == 'control' else 10), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        else:
            bio_dim_val = {'e3a': 25, 'e3b': 40, 'e3c': 44}[feature_set]
            output_signature = (
                (
                    tf.TensorSpec(shape=(None, 1, 24, 7 if mode == 'control' else 10), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, bio_dim_val), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )

        train_ds = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        # Select pre-encoded validation data for current mode
        if mode == 'mask_augmented':
            val_inputs_current = val_inputs_mask_aug
        else:
            val_inputs_current = val_inputs_ctrl

        # Create checkpoint callback AFTER validation data is ready
        if checkpoint_monitor == 'val_prauc':
            class PRaucCallback(tf.keras.callbacks.Callback):
                def __init__(self, val_data, val_labels, ckpt_path):
                    super().__init__()
                    self.val_data = val_data
                    self.val_labels = val_labels
                    self.ckpt_path = ckpt_path
                    self.best_prauc = -np.inf
                    self.best_epoch = -1

                def on_epoch_end(self, epoch, logs=None):
                    val_preds = self.model.predict(self.val_data, verbose=0)
                    if val_preds.ndim == 2 and val_preds.shape[1] == 2:
                        val_probs = val_preds[:, 1]
                    else:
                        val_probs = val_preds.ravel()

                    prauc = average_precision_score(self.val_labels, val_probs)
                    logs['val_prauc'] = prauc

                    # Save if improved, with 1e-12 tolerance favoring earlier epoch
                    if prauc > self.best_prauc + 1e-12:
                        self.best_prauc = prauc
                        self.best_epoch = epoch
                        self.model.save_weights(self.ckpt_path)
                        print(f"\nEpoch {epoch+1}: val_prauc improved to {prauc:.6f}, saving model to {self.ckpt_path}")

            checkpoint_cb = PRaucCallback(val_inputs_current, val_labels, ckpt_path)
        else:
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                ckpt_path,
                monitor=checkpoint_monitor,
                save_best_only=True,
                mode='min',
                verbose=1
            )

        # Train
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(val_inputs_current, val_labels),
            callbacks=[checkpoint_cb],
            verbose=1
        )

        # Assert generator yielded exactly total_batches
        assert batch_counter[0] == total_batches, \
            f"Generator should yield exactly {total_batches} batches, got {batch_counter[0]}"

        # Calculate actual draws (with replacement, so equal to expected)
        actual_draws = {
            'positive': n_pos_per_batch * batch_counter[0],
            'hard_negative': n_hard_neg_per_batch * batch_counter[0],
            'normal_negative': n_normal_neg_per_batch * batch_counter[0]
        }

        print(f"\n[Batch verification] Total batches: {batch_counter[0]}/{total_batches}")
        print(f"[Draw counts] Positive: {actual_draws['positive']}, "
              f"Hard-neg: {actual_draws['hard_negative']}, "
              f"Normal-neg: {actual_draws['normal_negative']}")

        # Save history
        history_dir = os.path.join(run_dir, 'histories')
        os.makedirs(history_dir, exist_ok=True)
        history_json = os.path.join(history_dir, f'{mode}_fold_{fold_num}_history.json')
        with open(history_json, 'w') as f:
            json.dump(history.history, f, indent=2)

        # Load best checkpoint
        model.load_weights(ckpt_path)

        # VALIDATION PREDICTION AND THRESHOLD SELECTION
        # Use pre-encoded validation data
        val_probs = model.predict(val_inputs_current, batch_size=256, verbose=0)
        val_probs = val_probs.ravel()

        # Select thresholds from VALIDATION set
        val_thr_f1, val_best_f1 = optimal_threshold_for_f1(val_labels, val_probs)
        val_thr_mcc, val_best_mcc = optimal_threshold_for_mcc(val_labels, val_probs)

        # Compute validation metrics with selected thresholds
        val_auroc = roc_auc_score(val_labels, val_probs)
        val_prauc = average_precision_score(val_labels, val_probs)

        # Validation F1 metrics
        val_pred_f1 = (val_probs >= val_thr_f1 - 1e-12).astype(int)
        val_cm_f1 = confusion_matrix(val_labels, val_pred_f1, labels=[0, 1])
        val_tn_f1, val_fp_f1, val_fn_f1, val_tp_f1 = val_cm_f1.ravel()
        val_prec_f1 = precision_score(val_labels, val_pred_f1, zero_division=0.0)
        val_rec_f1 = recall_score(val_labels, val_pred_f1, zero_division=0.0)

        # Validation MCC metrics
        val_pred_mcc = (val_probs >= val_thr_mcc - 1e-12).astype(int)
        val_cm_mcc = confusion_matrix(val_labels, val_pred_mcc, labels=[0, 1])
        val_tn_mcc, val_fp_mcc, val_fn_mcc, val_tp_mcc = val_cm_mcc.ravel()
        val_prec_mcc = precision_score(val_labels, val_pred_mcc, zero_division=0.0)
        val_rec_mcc = recall_score(val_labels, val_pred_mcc, zero_division=0.0)

        val_metrics = {
            'fold_idx': fold_num,
            'mode': mode,
            'split': 'validation',
            'auroc': val_auroc,
            'prauc': val_prauc,
            'threshold_f1': val_thr_f1,
            'f1': val_best_f1,
            'precision_f1': val_prec_f1,
            'recall_f1': val_rec_f1,
            'tp_f1': int(val_tp_f1),
            'fp_f1': int(val_fp_f1),
            'tn_f1': int(val_tn_f1),
            'fn_f1': int(val_fn_f1),
            'threshold_mcc': val_thr_mcc,
            'mcc': val_best_mcc,
            'precision_mcc': val_prec_mcc,
            'recall_mcc': val_rec_mcc,
            'tp_mcc': int(val_tp_mcc),
            'fp_mcc': int(val_fp_mcc),
            'tn_mcc': int(val_tn_mcc),
            'fn_mcc': int(val_fn_mcc)
        }

        # Save validation predictions
        pred_dir = os.path.join(run_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        val_pred_json = os.path.join(pred_dir, f'{mode}_fold_{fold_num}_val_predictions.json')
        with open(val_pred_json, 'w') as f:
            json.dump({
                'val_probs': val_probs.tolist(),
                'val_labels': val_labels.tolist(),
                'metrics': val_metrics
            }, f, indent=2)

        # TEST PREDICTION WITH LOCKED THRESHOLDS
        # Use pre-encoded test data
        if mode == 'control':
            test_X = test_X_control
        else:
            test_X = test_X_mask_augmented

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
        test_cm_f1 = confusion_matrix(test_labels, test_pred_f1, labels=[0, 1])
        test_tn_f1, test_fp_f1, test_fn_f1, test_tp_f1 = test_cm_f1.ravel()
        test_prec_f1 = precision_score(test_labels, test_pred_f1, zero_division=0.0)
        test_rec_f1 = recall_score(test_labels, test_pred_f1, zero_division=0.0)
        test_f1 = f1_score(test_labels, test_pred_f1, zero_division=0.0)

        # Test MCC metrics with locked threshold
        test_pred_mcc = (test_probs >= val_thr_mcc - 1e-12).astype(int)
        test_cm_mcc = confusion_matrix(test_labels, test_pred_mcc, labels=[0, 1])
        test_tn_mcc, test_fp_mcc, test_fn_mcc, test_tp_mcc = test_cm_mcc.ravel()
        test_prec_mcc = precision_score(test_labels, test_pred_mcc, zero_division=0.0)
        test_rec_mcc = recall_score(test_labels, test_pred_mcc, zero_division=0.0)
        test_mcc = matthews_corrcoef(test_labels, test_pred_mcc)

        test_metrics = {
            'fold_idx': fold_num,
            'mode': mode,
            'split': 'test',
            'auroc': test_auroc,
            'prauc': test_prauc,
            'threshold_f1': val_thr_f1,  # locked from validation
            'f1': test_f1,
            'precision_f1': test_prec_f1,
            'recall_f1': test_rec_f1,
            'tp_f1': int(test_tp_f1),
            'fp_f1': int(test_fp_f1),
            'tn_f1': int(test_tn_f1),
            'fn_f1': int(test_fn_f1),
            'threshold_mcc': val_thr_mcc,  # locked from validation
            'mcc': test_mcc,
            'precision_mcc': test_prec_mcc,
            'recall_mcc': test_rec_mcc,
            'tp_mcc': int(test_tp_mcc),
            'fp_mcc': int(test_fp_mcc),
            'tn_mcc': int(test_tn_mcc),
            'fn_mcc': int(test_fn_mcc)
        }

        # Save test predictions
        test_pred_json = os.path.join(pred_dir, f'{mode}_fold_{fold_num}_test_predictions.json')
        with open(test_pred_json, 'w') as f:
            json.dump({
                'test_probs': test_probs.tolist(),
                'test_labels': test_labels.tolist(),
                'metrics': test_metrics
            }, f, indent=2)

        # Store both validation and test metrics
        fold_results[mode] = {
            'validation': val_metrics,
            'test': test_metrics,
            'sampling_config': sampling_config,
            'total_batches': total_batches,
            'expected_draws': expected_draws,
            'actual_draws': actual_draws,
            'rng_seeds': {
                'seed_pos': seed_pos,
                'seed_hard': seed_hard,
                'seed_normal': seed_normal,
                'seed_shuffle': seed_shuffle
            }
        }

        print(f"\n{mode.upper()} Results:")
        print(f"  VALIDATION:")
        print(f"    AUROC: {val_metrics['auroc']:.6f}")
        print(f"    PRAUC: {val_metrics['prauc']:.6f}")
        print(f"    F1: {val_metrics['f1']:.6f} @ threshold {val_metrics['threshold_f1']:.6f}")
        print(f"    MCC: {val_metrics['mcc']:.6f} @ threshold {val_metrics['threshold_mcc']:.6f}")
        print(f"  TEST (with locked thresholds):")
        print(f"    AUROC: {test_metrics['auroc']:.6f}")
        print(f"    PRAUC: {test_metrics['prauc']:.6f}")
        print(f"    F1: {test_metrics['f1']:.6f} @ threshold {test_metrics['threshold_f1']:.6f}")
        print(f"    MCC: {test_metrics['mcc']:.6f} @ threshold {test_metrics['threshold_mcc']:.6f}")

    # REQUIREMENT 5: Clear session AFTER both modes
    tf.keras.backend.clear_session()

    return fold_results, split_hashes


def generate_five_fold_summary(run_dir, dataset_name, stage, feature_set):
    """
    Generate 5-fold summary CSV when all 5 folds complete.

    Args:
        run_dir: Single run directory containing all 5 folds (fold_idx=None case)
        dataset_name: Dataset name
        stage: 'p12' or 'biofusion'
        feature_set: 'e3a', 'e3b', or 'e3c'

    Loads per_fold CSV from run_dir, computes mean and std (ddof=0) for each metric.
    """
    per_fold_csv = os.path.join(run_dir, 'E5C_encoding_per_fold.csv')

    if not os.path.exists(per_fold_csv):
        raise FileNotFoundError(f"Per-fold CSV not found at {per_fold_csv}. Cannot generate summary.")

    # Load fold results
    combined_df = pd.read_csv(per_fold_csv)

    # Check we have 5 folds worth of data
    unique_folds = combined_df['fold_idx'].unique()
    n_folds = len(unique_folds)
    if n_folds != 5:
        raise ValueError(f"Expected 5 folds, found {n_folds} folds: {sorted(unique_folds)}. Cannot generate summary.")

    # Verify we have both modes and both splits
    modes = set(combined_df['mode'].unique())
    splits = set(combined_df['split'].unique())
    if not ({'control', 'mask_augmented'}.issubset(modes)):
        raise ValueError(f"Missing encoding modes. Found: {modes}, expected: {{'control', 'mask_augmented'}}.")
    if not ({'validation', 'test'}.issubset(splits)):
        raise ValueError(f"Missing splits. Found: {splits}, expected: {{'validation', 'test'}}.")

    # Separate by mode and split
    control_val = combined_df[(combined_df['mode'] == 'control') & (combined_df['split'] == 'validation')]
    control_test = combined_df[(combined_df['mode'] == 'control') & (combined_df['split'] == 'test')]
    mask_val = combined_df[(combined_df['mode'] == 'mask_augmented') & (combined_df['split'] == 'validation')]
    mask_test = combined_df[(combined_df['mode'] == 'mask_augmented') & (combined_df['split'] == 'test')]

    # Verify each subset has exactly 5 rows (one per fold)
    if len(control_val) != 5:
        raise ValueError(f"control/validation should have 5 rows, got {len(control_val)}. Rows: {control_val['fold_idx'].tolist()}")
    if len(control_test) != 5:
        raise ValueError(f"control/test should have 5 rows, got {len(control_test)}. Rows: {control_test['fold_idx'].tolist()}")
    if len(mask_val) != 5:
        raise ValueError(f"mask_augmented/validation should have 5 rows, got {len(mask_val)}. Rows: {mask_val['fold_idx'].tolist()}")
    if len(mask_test) != 5:
        raise ValueError(f"mask_augmented/test should have 5 rows, got {len(mask_test)}. Rows: {mask_test['fold_idx'].tolist()}")

    # Verify fold_idx alignment (should be 1..5, not 0..4)
    control_val_folds = set(control_val['fold_idx'].values)
    control_test_folds = set(control_test['fold_idx'].values)
    mask_val_folds = set(mask_val['fold_idx'].values)
    mask_test_folds = set(mask_test['fold_idx'].values)

    expected_folds = {1, 2, 3, 4, 5}
    if not (control_val_folds == expected_folds):
        raise ValueError(f"control/validation fold_idx should be {{1,2,3,4,5}}, got {control_val_folds}")
    if not (control_test_folds == expected_folds):
        raise ValueError(f"control/test fold_idx should be {{1,2,3,4,5}}, got {control_test_folds}")
    if not (mask_val_folds == expected_folds):
        raise ValueError(f"mask_augmented/validation fold_idx should be {{1,2,3,4,5}}, got {mask_val_folds}")
    if not (mask_test_folds == expected_folds):
        raise ValueError(f"mask_augmented/test fold_idx should be {{1,2,3,4,5}}, got {mask_test_folds}")

    print(f"\n[Summary validation] All checks passed:")
    print(f"  - 5 folds present: {sorted(unique_folds)}")
    print(f"  - 2 modes present: {sorted(modes)}")
    print(f"  - 2 splits present: {sorted(splits)}")
    print(f"  - Each mode/split combo has exactly 5 rows")
    print(f"  - fold_idx aligned: {{1,2,3,4,5}}")

    # Metrics to summarize
    metrics = ['auroc', 'prauc', 'f1', 'precision_f1', 'recall_f1', 'threshold_f1',
               'tp_f1', 'fp_f1', 'tn_f1', 'fn_f1',
               'mcc', 'precision_mcc', 'recall_mcc', 'threshold_mcc',
               'tp_mcc', 'fp_mcc', 'tn_mcc', 'fn_mcc']

    summary_rows = []

    for metric in metrics:
        # Validation - align by fold_idx
        ctrl_val_sorted = control_val.sort_values('fold_idx').set_index('fold_idx')[metric]
        mask_val_sorted = mask_val.sort_values('fold_idx').set_index('fold_idx')[metric]

        ctrl_val_mean = ctrl_val_sorted.mean()
        ctrl_val_std = ctrl_val_sorted.std(ddof=0)
        mask_val_mean = mask_val_sorted.mean()
        mask_val_std = mask_val_sorted.std(ddof=0)

        # Paired difference (aligned by fold_idx)
        paired_delta_val = mask_val_sorted - ctrl_val_sorted
        delta_val_mean = paired_delta_val.mean()
        delta_val_std = paired_delta_val.std(ddof=0)

        # Verify delta_std is not NaN
        if pd.isna(delta_val_std):
            raise ValueError(f"delta_std is NaN for metric '{metric}' (validation). "
                           f"ctrl_val_sorted: {ctrl_val_sorted.values}, "
                           f"mask_val_sorted: {mask_val_sorted.values}, "
                           f"paired_delta: {paired_delta_val.values}")

        summary_rows.append({
            'metric': f'val_{metric}',
            'control_mean': ctrl_val_mean,
            'control_std': ctrl_val_std,
            'mask_augmented_mean': mask_val_mean,
            'mask_augmented_std': mask_val_std,
            'delta_mean': delta_val_mean,
            'delta_std': delta_val_std
        })

        # Test - align by fold_idx
        ctrl_test_sorted = control_test.sort_values('fold_idx').set_index('fold_idx')[metric]
        mask_test_sorted = mask_test.sort_values('fold_idx').set_index('fold_idx')[metric]

        ctrl_test_mean = ctrl_test_sorted.mean()
        ctrl_test_std = ctrl_test_sorted.std(ddof=0)
        mask_test_mean = mask_test_sorted.mean()
        mask_test_std = mask_test_sorted.std(ddof=0)

        # Paired difference (aligned by fold_idx)
        paired_delta_test = mask_test_sorted - ctrl_test_sorted
        delta_test_mean = paired_delta_test.mean()
        delta_test_std = paired_delta_test.std(ddof=0)

        # Verify delta_std is not NaN
        if pd.isna(delta_test_std):
            raise ValueError(f"delta_std is NaN for metric '{metric}' (test). "
                           f"ctrl_test_sorted: {ctrl_test_sorted.values}, "
                           f"mask_test_sorted: {mask_test_sorted.values}, "
                           f"paired_delta: {paired_delta_test.values}")

        summary_rows.append({
            'metric': f'test_{metric}',
            'control_mean': ctrl_test_mean,
            'control_std': ctrl_test_std,
            'mask_augmented_mean': mask_test_mean,
            'mask_augmented_std': mask_test_std,
            'delta_mean': delta_test_mean,
            'delta_std': delta_test_std
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save to run directory
    summary_path = os.path.join(run_dir, 'E5C_encoding_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n5-fold summary saved: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description='E5-C Encoding Experiment')
    parser.add_argument('--dataset', required=True, choices=['K562', 'HEK293t', 'II5', 'II6'])
    parser.add_argument('--stage', required=True, choices=['p12', 'biofusion'])
    parser.add_argument('--fold', type=int, default=None, help='Fold index 0-4, or None for all 5 folds')
    args = parser.parse_args()

    dataset_name = args.dataset
    stage_arg = args.stage
    fold_idx = args.fold

    if fold_idx is not None and (fold_idx < 0 or fold_idx > 4):
        raise ValueError("fold must be 0-4 or None")

    # Load reference config
    reference_config = E5C_REFERENCE_CONFIGS[dataset_name]
    e3v2_cfg = E3V2_CONFIGS[dataset_name]

    # Use stage from CLI argument
    stage = stage_arg
    feature_set = e3v2_cfg['feature_set']
    loss_mode = e3v2_cfg['loss']
    sampling_mode = e3v2_cfg['sampling']

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

    checkpoint_monitor = e3v2_cfg['checkpoint_monitor']

    # Create unique run directory at start of main()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, 'weights', f'{dataset_name}_E5C_{stage}_{feature_set}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"E5-C Encoding Experiment")
    print(f"Dataset: {dataset_name}")
    if fold_idx is not None:
        print(f"Fold: {fold_idx} (CLI) -> fold_{fold_idx + 1} (file)")
    else:
        print(f"Fold: All 5 folds")
    print(f"Run directory: {run_dir}")
    print(f"Stage: {stage}")
    print(f"Feature set: {feature_set}")
    print(f"Loss mode: {loss_mode}")
    print(f"Sampling: {sampling_mode}")
    print(f"Checkpoint monitor: {checkpoint_monitor}")

    # Load dataset (CSV has no header, 3 columns: sgRNA, off-target, label)
    dataset_path = os.path.join(BASE_DIR, 'Datasets', f'{dataset_name}.csv')
    df = pd.read_csv(dataset_path, header=None, names=['sgRNA', 'DNA', 'Label'])

    # Create source_id as integer index
    df['source_id'] = np.arange(len(df), dtype=int)

    # Load split from TXT source_id files
    def load_split_from_txt(fold_num, split_name):
        """Load source_ids from TXT and filter dataset."""
        if split_name == 'test':
            txt_path = os.path.join(reference_config['checkpoint_dir'], 'splits', 'test_source_ids.txt')
        else:
            txt_path = os.path.join(reference_config['checkpoint_dir'], 'splits', f'fold_{fold_num}_{split_name}_source_ids.txt')

        with open(txt_path, 'r') as f:
            source_ids = [int(line.strip()) for line in f if line.strip()]

        # Filter dataset by source_id
        split_df = df[df['source_id'].isin(source_ids)].copy()

        # Verify all IDs are present
        found_ids = set(split_df['source_id'].values)
        missing_ids = set(source_ids) - found_ids
        if missing_ids:
            raise ValueError(f"{split_name} split: {len(missing_ids)} source_ids missing from dataset")

        return split_df

    # Handle fold_idx = None: run all 5 folds sequentially
    if fold_idx is None:
        all_fold_results = []
        all_paired_rows = []
        all_split_hashes = {}

        for current_fold in range(5):
            print(f"\n{'='*80}")
            print(f"Processing Fold {current_fold + 1} of 5")
            print(f"{'='*80}\n")

            train_df = load_split_from_txt(current_fold + 1, 'train')
            val_df = load_split_from_txt(current_fold + 1, 'val')
            test_df = load_split_from_txt(current_fold + 1, 'test')

            # Train fold (BOTH modes)
            fold_results, split_hashes = train_e5c_one_fold(
                dataset_name, current_fold, train_df, val_df, test_df,
                stage, feature_set, loss_mode, sampling_mode, sampling_config,
                checkpoint_monitor, reference_config, run_dir
            )

            # Store results
            all_fold_results.append(fold_results)
            all_split_hashes[f'fold_{current_fold + 1}'] = split_hashes

            # Build paired row for this fold
            ctrl_val = fold_results['control']['validation']
            aug_val = fold_results['mask_augmented']['validation']
            ctrl_test = fold_results['control']['test']
            aug_test = fold_results['mask_augmented']['test']

            paired_row = {
                'fold_idx': current_fold + 1,
                # Validation deltas
                'val_delta_auroc': aug_val['auroc'] - ctrl_val['auroc'],
                'val_delta_prauc': aug_val['prauc'] - ctrl_val['prauc'],
                'val_delta_f1': aug_val['f1'] - ctrl_val['f1'],
                'val_delta_precision_f1': aug_val['precision_f1'] - ctrl_val['precision_f1'],
                'val_delta_recall_f1': aug_val['recall_f1'] - ctrl_val['recall_f1'],
                'val_delta_threshold_f1': aug_val['threshold_f1'] - ctrl_val['threshold_f1'],
                'val_delta_tp_f1': aug_val['tp_f1'] - ctrl_val['tp_f1'],
                'val_delta_fp_f1': aug_val['fp_f1'] - ctrl_val['fp_f1'],
                'val_delta_tn_f1': aug_val['tn_f1'] - ctrl_val['tn_f1'],
                'val_delta_fn_f1': aug_val['fn_f1'] - ctrl_val['fn_f1'],
                'val_delta_mcc': aug_val['mcc'] - ctrl_val['mcc'],
                'val_delta_precision_mcc': aug_val['precision_mcc'] - ctrl_val['precision_mcc'],
                'val_delta_recall_mcc': aug_val['recall_mcc'] - ctrl_val['recall_mcc'],
                'val_delta_threshold_mcc': aug_val['threshold_mcc'] - ctrl_val['threshold_mcc'],
                'val_delta_tp_mcc': aug_val['tp_mcc'] - ctrl_val['tp_mcc'],
                'val_delta_fp_mcc': aug_val['fp_mcc'] - ctrl_val['fp_mcc'],
                'val_delta_tn_mcc': aug_val['tn_mcc'] - ctrl_val['tn_mcc'],
                'val_delta_fn_mcc': aug_val['fn_mcc'] - ctrl_val['fn_mcc'],
                # Test deltas
                'test_delta_auroc': aug_test['auroc'] - ctrl_test['auroc'],
                'test_delta_prauc': aug_test['prauc'] - ctrl_test['prauc'],
                'test_delta_f1': aug_test['f1'] - ctrl_test['f1'],
                'test_delta_precision_f1': aug_test['precision_f1'] - ctrl_test['precision_f1'],
                'test_delta_recall_f1': aug_test['recall_f1'] - ctrl_test['recall_f1'],
                'test_delta_threshold_f1': aug_test['threshold_f1'] - ctrl_test['threshold_f1'],
                'test_delta_tp_f1': aug_test['tp_f1'] - ctrl_test['tp_f1'],
                'test_delta_fp_f1': aug_test['fp_f1'] - ctrl_test['fp_f1'],
                'test_delta_tn_f1': aug_test['tn_f1'] - ctrl_test['tn_f1'],
                'test_delta_fn_f1': aug_test['fn_f1'] - ctrl_test['fn_f1'],
                'test_delta_mcc': aug_test['mcc'] - ctrl_test['mcc'],
                'test_delta_precision_mcc': aug_test['precision_mcc'] - ctrl_test['precision_mcc'],
                'test_delta_recall_mcc': aug_test['recall_mcc'] - ctrl_test['recall_mcc'],
                'test_delta_threshold_mcc': aug_test['threshold_mcc'] - ctrl_test['threshold_mcc'],
                'test_delta_tp_mcc': aug_test['tp_mcc'] - ctrl_test['tp_mcc'],
                'test_delta_fp_mcc': aug_test['fp_mcc'] - ctrl_test['fp_mcc'],
                'test_delta_tn_mcc': aug_test['tn_mcc'] - ctrl_test['tn_mcc'],
                'test_delta_fn_mcc': aug_test['fn_mcc'] - ctrl_test['fn_mcc'],
                # Control validation
                'control_val_auroc': ctrl_val['auroc'],
                'control_val_prauc': ctrl_val['prauc'],
                'control_val_f1': ctrl_val['f1'],
                'control_val_mcc': ctrl_val['mcc'],
                # Control test
                'control_test_auroc': ctrl_test['auroc'],
                'control_test_prauc': ctrl_test['prauc'],
                'control_test_f1': ctrl_test['f1'],
                'control_test_mcc': ctrl_test['mcc'],
                # Augmented validation
                'augmented_val_auroc': aug_val['auroc'],
                'augmented_val_prauc': aug_val['prauc'],
                'augmented_val_f1': aug_val['f1'],
                'augmented_val_mcc': aug_val['mcc'],
                # Augmented test
                'augmented_test_auroc': aug_test['auroc'],
                'augmented_test_prauc': aug_test['prauc'],
                'augmented_test_f1': aug_test['f1'],
                'augmented_test_mcc': aug_test['mcc']
            }
            all_paired_rows.append(paired_row)

        # Save combined per-fold CSV (all 5 folds x 2 modes x 2 splits = 20 rows)
        csv_path = os.path.join(run_dir, 'E5C_encoding_per_fold.csv')
        rows = []
        for fold_results in all_fold_results:
            for mode in ['control', 'mask_augmented']:
                rows.append(fold_results[mode]['validation'])
                rows.append(fold_results[mode]['test'])
        per_fold_df = pd.DataFrame(rows)
        per_fold_df.to_csv(csv_path, index=False)
        print(f"\nPer-fold CSV saved: {csv_path}")

        # Save paired differences CSV (5 rows, one per fold)
        paired_csv = os.path.join(run_dir, 'E5C_encoding_paired_differences.csv')
        paired_df = pd.DataFrame(all_paired_rows)
        paired_df.to_csv(paired_csv, index=False)
        print(f"Paired differences CSV saved: {paired_csv}")

        # Save config.json with all fold hashes and draw metadata
        config_data = {
            'experiment': 'E5C_Encoding',
            'dataset': dataset_name,
            'stage': stage,
            'feature_set': feature_set,
            'loss_mode': loss_mode,
            'sampling_mode': sampling_mode,
            'sampling_config': sampling_config,
            'checkpoint_monitor': checkpoint_monitor,
            'epochs': 35,
            'steps_per_epoch': 120,
            'batch_size': 256,
            'learning_rate': 0.0001,
            'reference_checkpoint': reference_config['checkpoint_dir'],
            'split_hashes': all_split_hashes,
            'mask_definitions': {
                'mismatch': 'sgRNA[i] != off[i] and no gap',
                'indel': '_ in sgRNA[i] or off[i]',
                'pam': 'positions 20-22'
            },
            'test_metrics_used_for_selection': False,
            'n_folds_completed': 5,
            'fold_batch_metadata': {},  # Will populate below
            'note': 'Five-fold cross-validation complete.'
        }

        # Add batch/draw metadata for each fold and mode
        for i, fold_results in enumerate(all_fold_results):
            fold_num = i + 1
            for mode in ['control', 'mask_augmented']:
                key = f'fold_{fold_num}_{mode}'
                config_data['fold_batch_metadata'][key] = {
                    'total_batches': fold_results[mode]['total_batches'],
                    'expected_draws': fold_results[mode]['expected_draws'],
                    'actual_draws': fold_results[mode]['actual_draws'],
                    'rng_seeds': fold_results[mode]['rng_seeds']
                }

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"\nConfig saved: {config_path}")

        # Generate 5-fold summary
        print("\nGenerating 5-fold summary...")
        summary_path = generate_five_fold_summary(run_dir, dataset_name, stage, feature_set)
        if summary_path:
            print(f"5-fold summary saved: {summary_path}")

        print(f"\nAll 5 folds completed successfully.")

    else:
        # Single fold execution
        train_df = load_split_from_txt(fold_idx + 1, 'train')
        val_df = load_split_from_txt(fold_idx + 1, 'val')
        test_df = load_split_from_txt(fold_idx + 1, 'test')

        # Train fold (BOTH modes)
        fold_results, split_hashes = train_e5c_one_fold(
            dataset_name, fold_idx, train_df, val_df, test_df,
            stage, feature_set, loss_mode, sampling_mode, sampling_config,
            checkpoint_monitor, reference_config, run_dir
        )

        # Save per-fold CSV (separate rows for validation and test)
        csv_path = os.path.join(run_dir, 'E5C_encoding_per_fold.csv')
        rows = []
        for mode in ['control', 'mask_augmented']:
            rows.append(fold_results[mode]['validation'])
            rows.append(fold_results[mode]['test'])
        per_fold_df = pd.DataFrame(rows)
        per_fold_df.to_csv(csv_path, index=False)

        print(f"\nPer-fold CSV saved: {csv_path}")

        # REQUIREMENT 7: Paired differences (match by fold_idx) - COMPLETE
        ctrl_val = fold_results['control']['validation']
        aug_val = fold_results['mask_augmented']['validation']
        ctrl_test = fold_results['control']['test']
        aug_test = fold_results['mask_augmented']['test']

        paired_row = {
            'fold_idx': fold_idx + 1,
            # Validation deltas
            'val_delta_auroc': aug_val['auroc'] - ctrl_val['auroc'],
            'val_delta_prauc': aug_val['prauc'] - ctrl_val['prauc'],
            'val_delta_f1': aug_val['f1'] - ctrl_val['f1'],
            'val_delta_precision_f1': aug_val['precision_f1'] - ctrl_val['precision_f1'],
            'val_delta_recall_f1': aug_val['recall_f1'] - ctrl_val['recall_f1'],
            'val_delta_threshold_f1': aug_val['threshold_f1'] - ctrl_val['threshold_f1'],
            'val_delta_tp_f1': aug_val['tp_f1'] - ctrl_val['tp_f1'],
            'val_delta_fp_f1': aug_val['fp_f1'] - ctrl_val['fp_f1'],
            'val_delta_tn_f1': aug_val['tn_f1'] - ctrl_val['tn_f1'],
            'val_delta_fn_f1': aug_val['fn_f1'] - ctrl_val['fn_f1'],
            'val_delta_mcc': aug_val['mcc'] - ctrl_val['mcc'],
            'val_delta_precision_mcc': aug_val['precision_mcc'] - ctrl_val['precision_mcc'],
            'val_delta_recall_mcc': aug_val['recall_mcc'] - ctrl_val['recall_mcc'],
            'val_delta_threshold_mcc': aug_val['threshold_mcc'] - ctrl_val['threshold_mcc'],
            'val_delta_tp_mcc': aug_val['tp_mcc'] - ctrl_val['tp_mcc'],
            'val_delta_fp_mcc': aug_val['fp_mcc'] - ctrl_val['fp_mcc'],
            'val_delta_tn_mcc': aug_val['tn_mcc'] - ctrl_val['tn_mcc'],
            'val_delta_fn_mcc': aug_val['fn_mcc'] - ctrl_val['fn_mcc'],
            # Test deltas
            'test_delta_auroc': aug_test['auroc'] - ctrl_test['auroc'],
            'test_delta_prauc': aug_test['prauc'] - ctrl_test['prauc'],
            'test_delta_f1': aug_test['f1'] - ctrl_test['f1'],
            'test_delta_precision_f1': aug_test['precision_f1'] - ctrl_test['precision_f1'],
            'test_delta_recall_f1': aug_test['recall_f1'] - ctrl_test['recall_f1'],
            'test_delta_threshold_f1': aug_test['threshold_f1'] - ctrl_test['threshold_f1'],
            'test_delta_tp_f1': aug_test['tp_f1'] - ctrl_test['tp_f1'],
            'test_delta_fp_f1': aug_test['fp_f1'] - ctrl_test['fp_f1'],
            'test_delta_tn_f1': aug_test['tn_f1'] - ctrl_test['tn_f1'],
            'test_delta_fn_f1': aug_test['fn_f1'] - ctrl_test['fn_f1'],
            'test_delta_mcc': aug_test['mcc'] - ctrl_test['mcc'],
            'test_delta_precision_mcc': aug_test['precision_mcc'] - ctrl_test['precision_mcc'],
            'test_delta_recall_mcc': aug_test['recall_mcc'] - ctrl_test['recall_mcc'],
            'test_delta_threshold_mcc': aug_test['threshold_mcc'] - ctrl_test['threshold_mcc'],
            'test_delta_tp_mcc': aug_test['tp_mcc'] - ctrl_test['tp_mcc'],
            'test_delta_fp_mcc': aug_test['fp_mcc'] - ctrl_test['fp_mcc'],
            'test_delta_tn_mcc': aug_test['tn_mcc'] - ctrl_test['tn_mcc'],
            'test_delta_fn_mcc': aug_test['fn_mcc'] - ctrl_test['fn_mcc'],
            # Control validation
            'control_val_auroc': ctrl_val['auroc'],
            'control_val_prauc': ctrl_val['prauc'],
            'control_val_f1': ctrl_val['f1'],
            'control_val_mcc': ctrl_val['mcc'],
            # Control test
            'control_test_auroc': ctrl_test['auroc'],
            'control_test_prauc': ctrl_test['prauc'],
            'control_test_f1': ctrl_test['f1'],
            'control_test_mcc': ctrl_test['mcc'],
            # Augmented validation
            'augmented_val_auroc': aug_val['auroc'],
            'augmented_val_prauc': aug_val['prauc'],
            'augmented_val_f1': aug_val['f1'],
            'augmented_val_mcc': aug_val['mcc'],
            # Augmented test
            'augmented_test_auroc': aug_test['auroc'],
            'augmented_test_prauc': aug_test['prauc'],
            'augmented_test_f1': aug_test['f1'],
            'augmented_test_mcc': aug_test['mcc']
        }

        paired_csv = os.path.join(run_dir, 'E5C_encoding_paired_differences.csv')
        paired_df = pd.DataFrame([paired_row])
        paired_df.to_csv(paired_csv, index=False)

        print(f"Paired differences CSV saved: {paired_csv}")

        # Save config.json for single fold run
        config_data = {
            'experiment': 'E5C_Encoding',
            'dataset': dataset_name,
            'stage': stage,
            'feature_set': feature_set,
            'loss_mode': loss_mode,
            'sampling_mode': sampling_mode,
            'sampling_config': sampling_config,
            'checkpoint_monitor': checkpoint_monitor,
            'epochs': 35,
            'steps_per_epoch': 120,
            'batch_size': 256,
            'learning_rate': 0.0001,
            'reference_checkpoint': reference_config['checkpoint_dir'],
            'split_hashes': {f'fold_{fold_idx + 1}': split_hashes},
            'mask_definitions': {
                'mismatch': 'sgRNA[i] != off[i] and no gap',
                'indel': '_ in sgRNA[i] or off[i]',
                'pam': 'positions 20-22'
            },
            'test_metrics_used_for_selection': False,
            'n_folds_completed': 1,
            'fold_batch_metadata': {},  # Will populate below
            'note': 'This is a single-fold run. Full config requires all 5 folds.'
        }

        # Add batch/draw metadata for this single fold
        for mode in ['control', 'mask_augmented']:
            key = f'fold_{fold_idx + 1}_{mode}'
            config_data['fold_batch_metadata'][key] = {
                'total_batches': fold_results[mode]['total_batches'],
                'expected_draws': fold_results[mode]['expected_draws'],
                'actual_draws': fold_results[mode]['actual_draws'],
                'rng_seeds': fold_results[mode]['rng_seeds']
            }

        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"\nConfig saved: {config_path}")
        print(f"\nFold {fold_idx + 1} completed successfully.")
        print("\nSingle fold run complete. Run all 5 folds to generate summary.")


if __name__ == '__main__':
    main()

