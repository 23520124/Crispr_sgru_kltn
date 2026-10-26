"""
E3v2-A — Loss Function Comparison for Imbalanced Data

So sánh riêng tác dụng của hàm loss trên cùng architecture/splits/sampling.

Ba chế độ loss (chỉ loss thay đổi, tất cả yếu tố khác giống hệt):
1. control: 0.5 BCE + 0.5 Soft Dice (E3 baseline)
2. weighted_bce: class-weighted BCE với pos_weight tính từ training fold
3. focal_bce: focal loss với gamma=2.0, alpha=0.25

Dataset-specific feature sets (theo paper reference baseline):
- II6: e3a (25 features)
- HEK293t: e3c (44 features)
- II5: e3b (40 features)
- BE3: e3c (44 features)
- K562: e3a (25 features)

Kiểm soát nghiêm ngặt:
- Cùng init weights cho 3 loss modes mỗi fold
- Cùng splits (giống train_e3_biofusion.py)
- Cùng seeds, generator riêng cho mỗi mode với cùng seed
- Cùng backbone, epochs, batch, checkpoint rule
- Cùng threshold tuning protocol (E0 cumulative trên validation)
- Blind test: dự đoán một lần, không dùng để tuning

Output: weights/{dataset}_E3V2_loss_{feature_set}_{timestamp}/
"""

import os
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import hashlib
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
import tensorflow.keras.backend as K

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

from biological_features import BiologicalFeatureExtractor, extract_features_batch
from MODEL_E3V2_Loss import Crispr_SGRU_BioFusion_Loss


# Dataset-specific feature sets (từ paper reference)
DATASET_FEATURE_MAP = {
    'II6': 'e3a',
    'HEK293t': 'e3c',
    'II5': 'e3b',
    'BE3': 'e3c',
    'K562': 'e3a',
    'I1': 'e3a',
    'I2': 'e3a',
    'CHANGE-seq': 'e3a'
}

# Paper reference values (S1 table)
PAPER_REFERENCE = {
    'K562': {'auroc': 1.000, 'prauc': 0.941},
    'HEK293t': {'auroc': 0.998, 'prauc': 0.801},
    'BE3': {'auroc': 0.995, 'prauc': 0.523},
    'II5': {'auroc': 0.998, 'prauc': 0.521},
    'II6': {'auroc': 0.996, 'prauc': 0.442},
    'CHANGE-seq': {'auroc': 0.997, 'prauc': 0.959},
    'I1': {'auroc': 0.986, 'prauc': 0.757},
    'I2': {'auroc': 0.999, 'prauc': 0.621},
}


def load_data(file_path):
    """Load data with source row IDs (giống train_e3_biofusion.py)."""
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
    """Encode sequences for P12."""
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
    Returns smallest threshold on tie.
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


class BioFusionGenerator:
    """
    Generator for E3 BioFusion: yields [sequence, bio_features], labels.
    """
    def __init__(self, pos_enc, pos_bio, bio_hard_enc, bio_hard_bio, normal_enc, normal_bio,
                 batch_size=256, steps_per_epoch=120, seed=42):
        self.pos_enc = pos_enc
        self.pos_bio = pos_bio
        self.bio_hard_enc = bio_hard_enc if len(bio_hard_enc) > 0 else normal_enc
        self.bio_hard_bio = bio_hard_bio if len(bio_hard_bio) > 0 else normal_bio
        self.normal_enc = normal_enc
        self.normal_bio = normal_bio
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.n_pos = 64
        self.n_hard = 64
        self.n_norm = 128

        self.rng = np.random.RandomState(seed)

    def flow(self):
        while True:
            for _ in range(self.steps_per_epoch):
                pos_idx = self.rng.choice(len(self.pos_enc), self.n_pos, replace=True)
                b_pos_enc = self.pos_enc[pos_idx]
                b_pos_bio = self.pos_bio[pos_idx]

                hard_idx = self.rng.choice(len(self.bio_hard_enc), self.n_hard, replace=True)
                b_hard_enc = self.bio_hard_enc[hard_idx]
                b_hard_bio = self.bio_hard_bio[hard_idx]

                norm_idx = self.rng.choice(len(self.normal_enc), self.n_norm, replace=True)
                b_norm_enc = self.normal_enc[norm_idx]
                b_norm_bio = self.normal_bio[norm_idx]

                X_seq = np.concatenate([b_pos_enc, b_hard_enc, b_norm_enc], axis=0)
                X_bio = np.concatenate([b_pos_bio, b_hard_bio, b_norm_bio], axis=0)
                y_batch = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_norm), dtype=np.float32)

                p = self.rng.permutation(len(y_batch))
                yield [X_seq[p], X_bio[p]], y_batch[p].reshape(-1, 1)


def train_e3v2_loss(dataset_name, feature_set):
    """
    E3v2-A experiment: compare 3 loss functions với blind test.

    Args:
        dataset_name: dataset name (II6, HEK293t, II5, BE3, K562)
        feature_set: biological feature set (e3a/e3b/e3c)
    """
    print("="*80)
    print(f"E3v2-A Loss Comparison: {dataset_name} with {feature_set.upper()}")
    print("="*80)

    # ==================== Configuration ====================
    EPOCHS = 35
    BATCH_SIZE = 256
    STEPS_PER_EPOCH = 120
    LR_INITIAL = 0.0001
    N_FOLDS = 5
    RANDOM_SEED = 42
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25

    # Feature dimensions
    feature_dims = {'e3a': 25, 'e3b': 40, 'e3c': 44}
    bio_feature_dim = feature_dims[feature_set]

    # Loss modes
    LOSS_MODES = ['control', 'weighted_bce', 'focal_bce']

    # Dataset path
    dataset_file = os.path.join(BASE_DIR, "Datasets", f"{dataset_name}.csv")
    # Special case: CHANGE-seq
    if dataset_name == "CHANGE-seq":
        dataset_file = os.path.join(BASE_DIR, "Datasets", "CHANGE-seq", "CHANGE-seq.csv")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "weights", f"{dataset_name}_E3V2_loss_{feature_set}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    splits_dir = os.path.join(output_dir, "splits")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    predictions_dir = os.path.join(output_dir, "predictions")
    histories_dir = os.path.join(output_dir, "histories")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(histories_dir, exist_ok=True)

    for loss_mode in LOSS_MODES:
        os.makedirs(os.path.join(checkpoints_dir, loss_mode), exist_ok=True)

    print(f"Dataset: {dataset_file}")
    print(f"Feature set: {feature_set.upper()} ({bio_feature_dim} features)")
    print(f"Output: {output_dir}")
    print(f"Loss modes: {LOSS_MODES}")

    # ==================== Load Data ====================
    print("\n[Load Data]")
    Negative, Positive = load_data(dataset_file)
    print(f"Loaded: {len(Positive)} positives, {len(Negative)} negatives")

    # Reproducibility metadata
    dataset_hash = compute_file_hash(dataset_file)
    config = {
        'experiment': 'E3v2-A Loss Function Comparison',
        'dataset': dataset_name,
        'dataset_file': dataset_file,
        'dataset_hash': dataset_hash,
        'feature_set': feature_set,
        'bio_feature_dim': bio_feature_dim,
        'loss_modes': LOSS_MODES,
        'loss_parameters': {
            'control': 'BCE-Dice (0.5 BCE + 0.5 Soft Dice)',
            'weighted_bce': 'Class-weighted BCE with pos_weight per fold',
            'focal_bce': f'Focal BCE with gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}'
        },
        'focal_gamma': FOCAL_GAMMA,
        'focal_alpha': FOCAL_ALPHA,
        'n_positives': len(Positive),
        'n_negatives': len(Negative),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'steps_per_epoch': STEPS_PER_EPOCH,
        'learning_rate': LR_INITIAL,
        'optimizer': 'Adam',
        'checkpoint_criterion': 'val_loss',
        'threshold_rule': 'F1-optimal on validation (cumulative E0)',
        'prauc_rule': 'auc(recall, precision) from sklearn.precision_recall_curve',
        'n_folds': N_FOLDS,
        'random_seed': RANDOM_SEED,
        'timestamp': timestamp,
        'split_hashes': {},
        'init_hashes': {},
        'pos_weights_per_fold': {},
        'blind_test_note': 'Test set used only for final evaluation, not for tuning',
        'paper_reference': PAPER_REFERENCE.get(dataset_name, {})
    }

    # ==================== Train-Test Split (giống train_e3_biofusion.py) ====================
    print("\n[Train-Test Split]")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Split positive và negative riêng
    pos_train, pos_test = train_test_split(
        Positive, test_size=0.15, random_state=RANDOM_SEED
    )
    neg_train, neg_test = train_test_split(
        Negative, test_size=0.15, random_state=RANDOM_SEED
    )

    # Ghép lại
    train_val_data = pos_train + neg_train
    test_data = pos_test + neg_test

    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(train_val_data)
    random.shuffle(test_data)

    config['split_hashes']['test'] = get_split_hash(test_data)
    print(f"Train+Val: {len(train_val_data)} ({len(pos_train)} pos, {len(neg_train)} neg)")
    print(f"Test (blind): {len(test_data)} ({len(pos_test)} pos, {len(neg_test)} neg)")

    # Save test split IDs
    test_source_ids = [row[3] for row in test_data]
    with open(os.path.join(splits_dir, 'test_source_ids.txt'), 'w') as f:
        f.write('\n'.join(map(str, test_source_ids)))

    # Encode and prepare test set (ONCE, for all modes)
    print("\n[Prepare Test Set]")
    test_df = pd.DataFrame(test_data, columns=['sgRNA', 'off', 'label', 'source_id'])
    test_enc = encoding(test_df).reshape((-1, 1, 24, 7)).astype("float32")
    test_bio = extract_features_batch(test_data, feature_set=feature_set)
    test_labels = test_df['label'].astype(float).values

    # ==================== 5-Fold CV ====================
    print("\n[5-Fold Cross-Validation]")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train_val_labels = np.array([float(row[2]) for row in train_val_data])

    # Storage for per-fold results
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_data, train_val_labels)):
        print("\n" + "="*80)
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print("="*80)

        # Split data
        train_fold = [train_val_data[i] for i in train_idx]
        val_fold = [train_val_data[i] for i in val_idx]

        config['split_hashes'][f'fold_{fold+1}_train'] = get_split_hash(train_fold)
        config['split_hashes'][f'fold_{fold+1}_val'] = get_split_hash(val_fold)

        # Save fold split IDs
        train_source_ids = [row[3] for row in train_fold]
        val_source_ids = [row[3] for row in val_fold]
        with open(os.path.join(splits_dir, f'fold{fold+1}_train_source_ids.txt'), 'w') as f:
            f.write('\n'.join(map(str, train_source_ids)))
        with open(os.path.join(splits_dir, f'fold{fold+1}_val_source_ids.txt'), 'w') as f:
            f.write('\n'.join(map(str, val_source_ids)))

        train_pos = [row for row in train_fold if float(row[2]) == 1.0]
        train_neg = [row for row in train_fold if float(row[2]) == 0.0]
        val_pos = [row for row in val_fold if float(row[2]) == 1.0]
        val_neg = [row for row in val_fold if float(row[2]) == 0.0]

        print(f"Train: {len(train_pos)} pos, {len(train_neg)} neg")
        print(f"Val: {len(val_pos)} pos, {len(val_neg)} neg")

        # Calculate pos_weight for THIS fold
        pos_weight_fold = len(train_neg) / len(train_pos) if len(train_pos) > 0 else 1.0
        config['pos_weights_per_fold'][f'fold_{fold+1}'] = float(pos_weight_fold)
        print(f"pos_weight for this fold: {pos_weight_fold:.4f}")

        # Define bio-hard negatives (<= 3 mismatches)
        train_bio_hard = [row for row in train_neg if count_mismatches(row[0], row[1]) <= 3]
        train_normal_neg = [row for row in train_neg if count_mismatches(row[0], row[1]) > 3]
        if len(train_bio_hard) == 0:
            train_bio_hard = train_neg
            train_normal_neg = train_neg
        print(f"Bio-hard: {len(train_bio_hard)}, Normal: {len(train_normal_neg)}")

        # Encode training data
        train_pos_df = pd.DataFrame(train_pos, columns=['sgRNA', 'off', 'label', 'source_id'])
        train_bio_hard_df = pd.DataFrame(train_bio_hard, columns=['sgRNA', 'off', 'label', 'source_id'])
        train_normal_neg_df = pd.DataFrame(train_normal_neg, columns=['sgRNA', 'off', 'label', 'source_id'])

        pos_enc = encoding(train_pos_df).reshape((-1, 1, 24, 7)).astype("float32")
        bio_hard_enc = encoding(train_bio_hard_df).reshape((-1, 1, 24, 7)).astype("float32") if len(train_bio_hard) > 0 else np.empty((0, 1, 24, 7), dtype="float32")
        normal_enc = encoding(train_normal_neg_df).reshape((-1, 1, 24, 7)).astype("float32")

        # Extract biological features
        pos_bio = extract_features_batch(train_pos, feature_set=feature_set)
        bio_hard_bio = extract_features_batch(train_bio_hard, feature_set=feature_set)
        normal_bio = extract_features_batch(train_normal_neg, feature_set=feature_set)

        # Fit scaler on TRAIN only (all training samples)
        train_all_bio = np.vstack([pos_bio, bio_hard_bio, normal_bio])
        scaler = StandardScaler()
        scaler.fit(train_all_bio)

        # Transform all splits
        pos_bio_scaled = scaler.transform(pos_bio)
        bio_hard_bio_scaled = scaler.transform(bio_hard_bio)
        normal_bio_scaled = scaler.transform(normal_bio)

        # Encode validation data
        val_fold_df = pd.DataFrame(val_fold, columns=['sgRNA', 'off', 'label', 'source_id'])
        val_enc = encoding(val_fold_df).reshape((-1, 1, 24, 7)).astype("float32")
        val_bio = extract_features_batch(val_fold, feature_set=feature_set)
        val_bio_scaled = scaler.transform(val_bio)
        val_labels = val_fold_df['label'].astype(float).values

        # Transform test bio features with this fold's scaler
        test_bio_scaled_fold = scaler.transform(test_bio)

        # ==================== Create Base Model for Init Weights ====================
        print("\n[Initialize Base Model]")
        base_model = Crispr_SGRU_BioFusion_Loss(
            bio_feature_dim=bio_feature_dim,
            loss_mode='control',
            learning_rate=LR_INITIAL
        )
        base_init_hash = get_model_weights_hash(base_model)
        base_init_weights = [w.copy() for w in base_model.get_weights()]
        param_count = base_model.count_params()
        print(f"Base init hash: {base_init_hash}")
        print(f"Parameters: {param_count:,}")

        config['init_hashes'][f'fold_{fold+1}_base'] = base_init_hash

        K.clear_session()
        del base_model
        gc.collect()

        # ==================== Train 3 Loss Modes ====================
        fold_mode_results = {}

        for loss_mode in LOSS_MODES:
            print("\n" + "-"*80)
            print(f"Loss Mode: {loss_mode.upper()}")
            print("-"*80)

            # Build model
            if loss_mode == 'weighted_bce':
                model = Crispr_SGRU_BioFusion_Loss(
                    bio_feature_dim=bio_feature_dim,
                    loss_mode=loss_mode,
                    pos_weight=pos_weight_fold,
                    learning_rate=LR_INITIAL
                )
            elif loss_mode == 'focal_bce':
                model = Crispr_SGRU_BioFusion_Loss(
                    bio_feature_dim=bio_feature_dim,
                    loss_mode=loss_mode,
                    focal_gamma=FOCAL_GAMMA,
                    focal_alpha=FOCAL_ALPHA,
                    learning_rate=LR_INITIAL
                )
            else:  # control
                model = Crispr_SGRU_BioFusion_Loss(
                    bio_feature_dim=bio_feature_dim,
                    loss_mode=loss_mode,
                    learning_rate=LR_INITIAL
                )

            # Set init weights (same as base)
            model.set_weights(base_init_weights)
            init_hash_check = get_model_weights_hash(model)
            print(f"Init hash: {init_hash_check}")

            if init_hash_check != base_init_hash:
                raise ValueError(f"Init hash mismatch for {loss_mode}: {init_hash_check} != {base_init_hash}")

            config['init_hashes'][f'fold_{fold+1}_{loss_mode}'] = init_hash_check

            # Set seeds for this mode to ensure same sampling
            mode_seed = RANDOM_SEED + fold
            random.seed(mode_seed)
            np.random.seed(mode_seed)
            tf.random.set_seed(mode_seed)

            # Create generator (SAME seed for reproducible sampling)
            gen_seed = RANDOM_SEED + fold
            train_gen = BioFusionGenerator(
                pos_enc, pos_bio_scaled,
                bio_hard_enc, bio_hard_bio_scaled,
                normal_enc, normal_bio_scaled,
                batch_size=BATCH_SIZE,
                steps_per_epoch=STEPS_PER_EPOCH,
                seed=gen_seed
            )

            # Callbacks
            weight_file = os.path.join(checkpoints_dir, loss_mode, f'fold{fold+1}_best.weights.h5')
            checkpoint = ModelCheckpoint(
                weight_file,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                verbose=0
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=0
            )

            # Train
            print("Training...")
            history = model.fit(
                train_gen.flow(),
                steps_per_epoch=STEPS_PER_EPOCH,
                epochs=EPOCHS,
                validation_data=([val_enc, val_bio_scaled], val_labels.reshape(-1, 1)),
                callbacks=[checkpoint, reduce_lr],
                verbose=0
            )

            # Save history
            history_df = pd.DataFrame(history.history)
            history_csv = os.path.join(histories_dir, f'fold{fold+1}_{loss_mode}_history.csv')
            history_df.to_csv(history_csv, index=False)

            # Load best weights
            model.load_weights(weight_file)

            # ==================== VALIDATION Evaluation ====================
            print("\n[Validation Evaluation]")
            val_probs = model.predict([val_enc, val_bio_scaled], verbose=0).ravel()

            val_auroc = roc_auc_score(val_labels, val_probs)
            val_precision_curve, val_recall_curve, _ = precision_recall_curve(
                val_labels, val_probs
            )
            val_prauc = auc(val_recall_curve, val_precision_curve)

            # F1 threshold on validation
            f1_thresh, val_f1_at_thresh = find_optimal_f1_threshold(val_labels, val_probs)
            val_preds = (val_probs >= f1_thresh).astype(int)

            val_f1 = f1_score(val_labels, val_preds)
            val_prec = precision_score(val_labels, val_preds, zero_division=0)
            val_rec = recall_score(val_labels, val_preds, zero_division=0)
            val_mcc = matthews_corrcoef(val_labels, val_preds)

            print(f"Val AUROC: {val_auroc:.4f}, PRAUC: {val_prauc:.4f}")
            print(f"Val F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, MCC: {val_mcc:.4f}")
            print(f"F1 threshold: {f1_thresh:.6f}")

            # Save validation predictions
            val_pred_df = pd.DataFrame({
                'source_id': val_fold_df['source_id'].values,
                'true_label': val_labels,
                'predicted_prob': val_probs,
                'predicted_label': val_preds
            })
            val_pred_csv = os.path.join(predictions_dir, f'fold{fold+1}_{loss_mode}_val_predictions.csv')
            val_pred_df.to_csv(val_pred_csv, index=False)

            # ==================== BLIND TEST Evaluation ====================
            print("\n[Blind Test Evaluation]")
            test_probs = model.predict([test_enc, test_bio_scaled_fold], verbose=0).ravel()

            test_auroc = roc_auc_score(test_labels, test_probs)
            test_precision_curve, test_recall_curve, _ = precision_recall_curve(
                test_labels, test_probs
            )
            test_prauc = auc(test_recall_curve, test_precision_curve)

            # Apply validation threshold to test
            test_preds = (test_probs >= f1_thresh).astype(int)

            test_f1 = f1_score(test_labels, test_preds)
            test_prec = precision_score(test_labels, test_preds, zero_division=0)
            test_rec = recall_score(test_labels, test_preds, zero_division=0)
            test_mcc = matthews_corrcoef(test_labels, test_preds)

            print(f"Test AUROC: {test_auroc:.4f}, PRAUC: {test_prauc:.4f}")
            print(f"Test F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, MCC: {test_mcc:.4f}")
            print(f"Applied threshold: {f1_thresh:.6f}")

            # Save test predictions
            test_pred_df = pd.DataFrame({
                'source_id': test_df['source_id'].values,
                'true_label': test_labels,
                'predicted_prob': test_probs,
                'predicted_label': test_preds
            })
            test_pred_csv = os.path.join(predictions_dir, f'fold{fold+1}_{loss_mode}_test_predictions.csv')
            test_pred_df.to_csv(test_pred_csv, index=False)

            # Store results
            fold_mode_results[loss_mode] = {
                'fold': fold + 1,
                'val_auroc': val_auroc,
                'val_prauc': val_prauc,
                'val_f1': val_f1,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_mcc': val_mcc,
                'test_auroc': test_auroc,
                'test_prauc': test_prauc,
                'test_f1': test_f1,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_mcc': test_mcc,
                'threshold': f1_thresh
            }

            # Cleanup
            K.clear_session()
            del model, train_gen
            gc.collect()

        # Save per-fold results
        fold_results.append(fold_mode_results)

        # Cleanup fold data
        del val_enc, val_bio_scaled, test_bio_scaled_fold
        gc.collect()

    # ==================== Aggregate Results ====================
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)

    # Per-fold CSV
    per_fold_rows = []
    for fold_data in fold_results:
        for loss_mode in LOSS_MODES:
            row = fold_data[loss_mode].copy()
            row['loss_mode'] = loss_mode
            per_fold_rows.append(row)

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_csv = os.path.join(output_dir, 'E3V2_loss_per_fold.csv')
    per_fold_df.to_csv(per_fold_csv, index=False)
    print(f"Per-fold results: {per_fold_csv}")

    # Summary statistics (Mean ± Std, ddof=0)
    summary_rows = []
    for loss_mode in LOSS_MODES:
        mode_data = per_fold_df[per_fold_df['loss_mode'] == loss_mode]

        row = {
            'loss_mode': loss_mode,
            'val_auroc_mean': mode_data['val_auroc'].mean(),
            'val_auroc_std': mode_data['val_auroc'].std(ddof=0),
            'val_prauc_mean': mode_data['val_prauc'].mean(),
            'val_prauc_std': mode_data['val_prauc'].std(ddof=0),
            'test_auroc_mean': mode_data['test_auroc'].mean(),
            'test_auroc_std': mode_data['test_auroc'].std(ddof=0),
            'test_prauc_mean': mode_data['test_prauc'].mean(),
            'test_prauc_std': mode_data['test_prauc'].std(ddof=0),
            'test_f1_mean': mode_data['test_f1'].mean(),
            'test_f1_std': mode_data['test_f1'].std(ddof=0),
            'test_precision_mean': mode_data['test_precision'].mean(),
            'test_precision_std': mode_data['test_precision'].std(ddof=0),
            'test_recall_mean': mode_data['test_recall'].mean(),
            'test_recall_std': mode_data['test_recall'].std(ddof=0),
            'test_mcc_mean': mode_data['test_mcc'].mean(),
            'test_mcc_std': mode_data['test_mcc'].std(ddof=0),
            'threshold_mean': mode_data['threshold'].mean(),
            'threshold_std': mode_data['threshold'].std(ddof=0)
        }

        # Add paper reference if available
        if dataset_name in PAPER_REFERENCE:
            row['paper_auroc'] = PAPER_REFERENCE[dataset_name].get('auroc', np.nan)
            row['paper_prauc'] = PAPER_REFERENCE[dataset_name].get('prauc', np.nan)
            row['paper_reference'] = f"Zhang et al. IJMS 2024 S1 - {dataset_name}"
        else:
            row['paper_auroc'] = np.nan
            row['paper_prauc'] = np.nan
            row['paper_reference'] = 'N/A'

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, 'E3V2_loss_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics: {summary_csv}")

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS (Test Set, Mean ± Std, ddof=0)")
    print("="*80)
    for _, row in summary_df.iterrows():
        print(f"\n{row['loss_mode'].upper()}:")
        print(f"  AUROC: {row['test_auroc_mean']:.4f} ± {row['test_auroc_std']:.4f}")
        print(f"  PRAUC: {row['test_prauc_mean']:.4f} ± {row['test_prauc_std']:.4f}")
        print(f"  F1: {row['test_f1_mean']:.4f} ± {row['test_f1_std']:.4f}")
        print(f"  Precision: {row['test_precision_mean']:.4f} ± {row['test_precision_std']:.4f}")
        print(f"  Recall: {row['test_recall_mean']:.4f} ± {row['test_recall_std']:.4f}")
        print(f"  MCC: {row['test_mcc_mean']:.4f} ± {row['test_mcc_std']:.4f}")

    # Paired differences (per fold)
    print("\n[Computing Paired Differences]")
    paired_rows = []
    for fold_data in fold_results:
        fold_num = fold_data['control']['fold']

        # weighted_bce vs control
        diff_weighted = {
            'fold': fold_num,
            'comparison': 'weighted_bce - control',
            'test_auroc_diff': fold_data['weighted_bce']['test_auroc'] - fold_data['control']['test_auroc'],
            'test_prauc_diff': fold_data['weighted_bce']['test_prauc'] - fold_data['control']['test_prauc'],
            'test_f1_diff': fold_data['weighted_bce']['test_f1'] - fold_data['control']['test_f1'],
            'test_precision_diff': fold_data['weighted_bce']['test_precision'] - fold_data['control']['test_precision'],
            'test_recall_diff': fold_data['weighted_bce']['test_recall'] - fold_data['control']['test_recall'],
            'test_mcc_diff': fold_data['weighted_bce']['test_mcc'] - fold_data['control']['test_mcc'],
            'threshold_diff': fold_data['weighted_bce']['threshold'] - fold_data['control']['threshold']
        }
        paired_rows.append(diff_weighted)

        # focal_bce vs control
        diff_focal = {
            'fold': fold_num,
            'comparison': 'focal_bce - control',
            'test_auroc_diff': fold_data['focal_bce']['test_auroc'] - fold_data['control']['test_auroc'],
            'test_prauc_diff': fold_data['focal_bce']['test_prauc'] - fold_data['control']['test_prauc'],
            'test_f1_diff': fold_data['focal_bce']['test_f1'] - fold_data['control']['test_f1'],
            'test_precision_diff': fold_data['focal_bce']['test_precision'] - fold_data['control']['test_precision'],
            'test_recall_diff': fold_data['focal_bce']['test_recall'] - fold_data['control']['test_recall'],
            'test_mcc_diff': fold_data['focal_bce']['test_mcc'] - fold_data['control']['test_mcc'],
            'threshold_diff': fold_data['focal_bce']['threshold'] - fold_data['control']['threshold']
        }
        paired_rows.append(diff_focal)

    paired_df = pd.DataFrame(paired_rows)
    paired_csv = os.path.join(output_dir, 'E3V2_loss_paired_differences.csv')
    paired_df.to_csv(paired_csv, index=False)
    print(f"Paired differences: {paired_csv}")

    # Print paired diff summary
    print("\nPaired Differences Summary:")
    for comparison in ['weighted_bce - control', 'focal_bce - control']:
        comp_data = paired_df[paired_df['comparison'] == comparison]
        print(f"\n{comparison}:")
        print(f"  AUROC: {comp_data['test_auroc_diff'].mean():.4f} ± {comp_data['test_auroc_diff'].std(ddof=0):.4f}")
        print(f"  PRAUC: {comp_data['test_prauc_diff'].mean():.4f} ± {comp_data['test_prauc_diff'].std(ddof=0):.4f}")
        print(f"  F1: {comp_data['test_f1_diff'].mean():.4f} ± {comp_data['test_f1_diff'].std(ddof=0):.4f}")
        print(f"  MCC: {comp_data['test_mcc_diff'].mean():.4f} ± {comp_data['test_mcc_diff'].std(ddof=0):.4f}")

    # ==================== Save Config ====================
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved: {config_file}")

    print("\n" + "="*80)
    print("E3v2-A LOSS COMPARISON COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E3v2-A: Loss Function Comparison with Blind Test')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['II6', 'HEK293t', 'II5', 'BE3', 'K562', 'I1', 'I2', 'CHANGE-seq'],
                        help='Dataset name')
    parser.add_argument('--feature-set', type=str, required=False,
                        help='Feature set (e3a/e3b/e3c). If not provided, uses dataset-specific default.')

    args = parser.parse_args()

    # Use dataset-specific feature set if not provided
    if args.feature_set is None:
        args.feature_set = DATASET_FEATURE_MAP.get(args.dataset, 'e3c')
        print(f"Using dataset-specific feature set: {args.feature_set}")
    else:
        if args.feature_set not in ['e3a', 'e3b', 'e3c']:
            raise ValueError(f"Invalid feature_set: {args.feature_set}. Must be e3a, e3b, or e3c.")

    train_e3v2_loss(args.dataset, args.feature_set)
