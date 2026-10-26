"""
E3 — Biological Feature Fusion Experiment

So sánh P12 baseline với P12 + biological features trên cùng split/seed/batches.

Hai chế độ:
A. p12_control: P12 hiện tại (sequence only)
B. p12_biofusion: P12 + biological feature branch

Feature sets:
- E3a: mismatch only (25 features)
- E3b: mismatch + PAM + GC (40 features)
- E3c: mismatch + PAM + GC + indel (44 features)

Mục tiêu:
- Kiểm tra biological features có cải thiện PRAUC/F1 trên II5/II6
- Giữ hiệu suất trên K562/HEK/BE3
- So sánh công bằng: cùng split, seed, sampling, epochs, checkpoint rule

Không thay đổi:
- MODEL12/P12 backbone
- BCE-Dice loss
- StableMiningGenerator replacement-control (64 pos, 64 hard, 128 norm)
- Batch 256, steps 120, epochs 35
- Adam 1e-4, ReduceLROnPlateau
- Checkpoint theo val_loss
- F1 threshold cumulative (E0), PRAUC từ continuous probs
- Kết quả: Mean ± Std của 5 fold (std ddof=0)
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import hashlib
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, precision_score, recall_score, f1_score
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
from MODEL_E3_BioFusion import Crispr_SGRU_MODEL12_Control, Crispr_SGRU_BioFusion


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


class StableMiningGenerator:
    """
    P12 baseline generator: 64 pos, 64 bio-hard, 128 normal (replacement).
    """
    def __init__(self, pos_enc, bio_hard_enc, normal_enc, batch_size=256, steps_per_epoch=120, seed=42):
        self.pos_enc = pos_enc
        self.bio_hard_enc = bio_hard_enc if len(bio_hard_enc) > 0 else normal_enc
        self.normal_enc = normal_enc
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
                b_pos = self.pos_enc[pos_idx]

                hard_idx = self.rng.choice(len(self.bio_hard_enc), self.n_hard, replace=True)
                b_hard = self.bio_hard_enc[hard_idx]

                norm_idx = self.rng.choice(len(self.normal_enc), self.n_norm, replace=True)
                b_norm = self.normal_enc[norm_idx]

                X_batch = np.concatenate([b_pos, b_hard, b_norm], axis=0)
                y_batch = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_norm), dtype=np.float32)

                p = self.rng.permutation(len(y_batch))
                yield X_batch[p], y_batch[p].reshape(-1, 1)


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
                b_pos_seq = self.pos_enc[pos_idx]
                b_pos_bio = self.pos_bio[pos_idx]

                hard_idx = self.rng.choice(len(self.bio_hard_enc), self.n_hard, replace=True)
                b_hard_seq = self.bio_hard_enc[hard_idx]
                b_hard_bio = self.bio_hard_bio[hard_idx]

                norm_idx = self.rng.choice(len(self.normal_enc), self.n_norm, replace=True)
                b_norm_seq = self.normal_enc[norm_idx]
                b_norm_bio = self.normal_bio[norm_idx]

                X_seq = np.concatenate([b_pos_seq, b_hard_seq, b_norm_seq], axis=0)
                X_bio = np.concatenate([b_pos_bio, b_hard_bio, b_norm_bio], axis=0)
                y_batch = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_norm), dtype=np.float32)

                p = self.rng.permutation(len(y_batch))
                yield (X_seq[p], X_bio[p]), y_batch[p].reshape(-1, 1)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E3 Biological Feature Fusion: P12 baseline vs P12+BioFeatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_e3_biofusion.py --dataset II5 --feature-set e3a
  python train_e3_biofusion.py --dataset II6 --feature-set e3b
  python train_e3_biofusion.py --dataset CHANGE-seq --feature-set e3c

Feature sets:
  e3a: mismatch only (25 features)
  e3b: mismatch + PAM + GC (40 features)
  e3c: mismatch + PAM + GC + indel (44 features)

Datasets:
  II5, II6, K562, BE3, hek (HEK293t), I1, I2, CHANGE-seq
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="II6",
        choices=["II5", "II6", "K562", "BE3", "hek", "I1", "I2", "CHANGE-seq"],
        help="Dataset to train on (default: II6)"
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default="e3b",
        choices=["e3a", "e3b", "e3c"],
        help="Feature set to use (default: e3b)"
    )

    args = parser.parse_args()
    data_name = args.dataset
    feature_set = args.feature_set

    dataset_path_map = {
        "BE3": os.path.join(BASE_DIR, "Datasets", "BE3.csv"),
        "I1":  os.path.join(BASE_DIR, "Datasets", "I1.csv"),
        "I2":  os.path.join(BASE_DIR, "Datasets", "I2.csv"),
        "II6": os.path.join(BASE_DIR, "Datasets", "II6.csv"),
        "II5": os.path.join(BASE_DIR, "Datasets", "II5.csv"),
        "K562": os.path.join(BASE_DIR, "Datasets", "K562.csv"),
        "hek": os.path.join(BASE_DIR, "Datasets", "HEK293t.csv"),
        "CHANGE-seq": os.path.join(BASE_DIR, "Datasets", "CHANGE-seq", "CHANGE-seq.csv"),
    }

    file_path = dataset_path_map.get(data_name)
    if file_path is None:
        print(f"[ERROR] Unknown data_name: '{data_name}'")
        sys.exit(1)

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, "weights", f"{data_name}_E3_{feature_set}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    save_dir_control = os.path.join(run_dir, "control")
    save_dir_biofusion = os.path.join(run_dir, "biofusion")
    history_dir = os.path.join(run_dir, "history")
    features_dir = os.path.join(run_dir, "features")
    predictions_dir = os.path.join(run_dir, "predictions")

    for d in [save_dir_control, save_dir_biofusion, history_dir, features_dir, predictions_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"E3 — BIOLOGICAL FEATURE FUSION EXPERIMENT")
    print(f"Dataset: {file_path}")
    print(f"Feature set: {feature_set.upper()}")
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")

    # Feature extractor
    extractor = BiologicalFeatureExtractor(feature_set=feature_set)
    feature_names = extractor.get_feature_names()
    feature_dim = extractor.get_feature_dim()

    print(f"Feature dimension: {feature_dim}")
    print(f"Feature names: {feature_names[:5]} ... ({len(feature_names)} total)\n")

    # Load data
    Negative, Positive = load_data(file_path)
    Negative = np.array(Negative, dtype=object)
    Positive = np.array(Positive, dtype=object)
    total_pos = len(Positive)
    total_neg = len(Negative)

    dataset_file_hash = compute_file_hash(file_path)
    print(f"Data: {total_pos} Pos | {total_neg} Neg")
    print(f"Dataset file hash: {dataset_file_hash}")

    # Split 15% blind test (same as P12)
    Train_Pool_Neg, Test_Neg = train_test_split(Negative, test_size=0.15, random_state=42)
    Train_Pool_Pos, Test_Pos = train_test_split(Positive, test_size=0.15, random_state=42)

    train_pool_neg_hash = get_split_hash(Train_Pool_Neg)
    test_neg_hash = get_split_hash(Test_Neg)
    train_pool_pos_hash = get_split_hash(Train_Pool_Pos)
    test_pos_hash = get_split_hash(Test_Pos)

    print(f"Train pool neg hash: {train_pool_neg_hash}")
    print(f"Test neg hash: {test_neg_hash}")
    print(f"Train pool pos hash: {train_pool_pos_hash}")
    print(f"Test pos hash: {test_pos_hash}")

    # Encode test set (sequence + bio features)
    Xtest_raw = np.vstack((Test_Neg, Test_Pos))
    np.random.seed(42)
    np.random.shuffle(Xtest_raw)
    test_source_ids = [int(row[3]) for row in Xtest_raw]

    Xtest_df = pd.DataFrame(Xtest_raw[:, :3])
    Xtest_enc = encoding(Xtest_df).reshape((-1, 1, 24, 7)).astype("float32")
    Xtest_bio = extract_features_batch(Xtest_raw.tolist(), feature_set=feature_set)
    y_test_gt = np.array([1 if float(x) > 0.0 else 0 for x in Xtest_df.iloc[:, 2]])

    print(f"Test: {len(y_test_gt)} samples, {np.sum(y_test_gt)} positives\n")

    BATCH_SIZE = 256
    STEPS_PER_EPOCH = 120
    EPOCHS = 35
    num_folds = 5

    train_pool = np.vstack((Train_Pool_Neg, Train_Pool_Pos))
    train_pool_y = np.array([0] * len(Train_Pool_Neg) + [1] * len(Train_Pool_Pos))
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_splits = list(skf.split(train_pool, train_pool_y))

    # Config - hash all three E3 files
    code_files = [
        __file__,
        os.path.join(CURRENT_DIR, "biological_features.py"),
        os.path.join(BASE_DIR, "Train", "MODEL_E3_BioFusion.py")
    ]
    code_hash_combined = hashlib.md5()
    for code_file in code_files:
        if os.path.exists(code_file):
            with open(code_file, 'rb') as f:
                code_hash_combined.update(f.read())
    code_hash = code_hash_combined.hexdigest()[:16]

    # Get TensorFlow and Keras versions safely
    tf_version = tf.__version__
    try:
        keras_version = tf.keras.__version__
    except AttributeError:
        keras_version = tf_version  # Fallback: Keras is integrated in TF 2.x

    config = {
        'dataset': data_name,
        'dataset_file_hash': dataset_file_hash,
        'train_pool_neg_hash': train_pool_neg_hash,
        'test_neg_hash': test_neg_hash,
        'train_pool_pos_hash': train_pool_pos_hash,
        'test_pos_hash': test_pos_hash,
        'feature_set': feature_set,
        'feature_dim': feature_dim,
        'feature_names': feature_names,
        'run_id': run_id,
        'total_pos': total_pos,
        'total_neg': total_neg,
        'test_size': 0.15,
        'num_folds': num_folds,
        'batch_size': BATCH_SIZE,
        'steps_per_epoch': STEPS_PER_EPOCH,
        'epochs': EPOCHS,
        'model_control': 'P12_Control',
        'model_biofusion': f'P12_BioFusion_{feature_set}',
        'loss': 'BCEDiceLoss',
        'optimizer': 'Adam_1e-4',
        'generator': 'StableMining_64pos_64hard_128norm',
        'checkpoint_rule': 'val_loss_min',
        'threshold_rule': 'F1_optimal_cumulative',
        'bio_hard_rule': 'mismatch_le_3',
        'tensorflow_version': tf_version,
        'keras_version': keras_version,
        'code_hash': code_hash,
        'fold_seeds': {},
        'init_hashes': {}
    }

    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Results storage
    results_control = []
    results_biofusion = []

    start_time = datetime.now()

    # =========================================================================
    # 5-FOLD TRAINING
    # =========================================================================
    for fold in range(num_folds):
        fold_seed = 42 + fold * 100
        config['fold_seeds'][f'fold_{fold+1}'] = fold_seed

        print(f"\n{'='*70}")
        print(f"FOLD #{fold + 1} (seed={fold_seed})")
        print(f"{'='*70}")

        train_idx, val_idx = fold_splits[fold]
        Train_raw, Val_raw = train_pool[train_idx], train_pool[val_idx]
        Train_y, Val_y = train_pool_y[train_idx], train_pool_y[val_idx]

        val_source_ids = [int(row[3]) for row in Val_raw]

        Train_Neg = Train_raw[Train_y == 0]
        Train_Pos = Train_raw[Train_y == 1]

        # Bio-hard negatives
        bio_mask = np.array([count_mismatches(r[0], r[1]) <= 3 for r in Train_Neg])
        Neg_Bio = Train_Neg[bio_mask]
        Neg_Normal = Train_Neg[~bio_mask]

        print(f"Train: {len(Train_Pos)} Pos | {len(Neg_Bio)} Bio-Hard | {len(Neg_Normal)} Normal")

        # Encoding (sequence)
        Train_Pos_enc = encoding(pd.DataFrame([row[:3] for row in Train_Pos])).reshape((-1, 1, 24, 7)).astype("float32")
        Neg_Bio_enc = encoding(pd.DataFrame([row[:3] for row in Neg_Bio])).reshape((-1, 1, 24, 7)).astype("float32") if len(Neg_Bio) > 0 else np.empty((0, 1, 24, 7), dtype="float32")
        Neg_Normal_enc = encoding(pd.DataFrame([row[:3] for row in Neg_Normal])).reshape((-1, 1, 24, 7)).astype("float32")

        Xval_enc = encoding(pd.DataFrame([row[:3] for row in Val_raw])).reshape((-1, 1, 24, 7)).astype("float32")
        y_val_gt = Val_y
        y_val_target = y_val_gt.reshape(-1, 1).astype(np.float32)

        # Extract biological features
        Train_Pos_bio = extract_features_batch(Train_Pos.tolist(), feature_set=feature_set)
        Neg_Bio_bio = extract_features_batch(Neg_Bio.tolist(), feature_set=feature_set) if len(Neg_Bio) > 0 else np.empty((0, feature_dim), dtype="float32")
        Neg_Normal_bio = extract_features_batch(Neg_Normal.tolist(), feature_set=feature_set)
        Xval_bio_raw = extract_features_batch(Val_raw.tolist(), feature_set=feature_set)

        # Fit scaler on TRAIN only
        scaler = StandardScaler()
        Train_all_bio = np.vstack([Train_Pos_bio, Neg_Bio_bio if len(Neg_Bio_bio) > 0 else Train_Pos_bio[:0], Neg_Normal_bio])
        scaler.fit(Train_all_bio)

        # Transform all splits
        Train_Pos_bio = scaler.transform(Train_Pos_bio)
        if len(Neg_Bio_bio) > 0:
            Neg_Bio_bio = scaler.transform(Neg_Bio_bio)
        Neg_Normal_bio = scaler.transform(Neg_Normal_bio)
        Xval_bio = scaler.transform(Xval_bio_raw)
        Xtest_bio_fold = scaler.transform(Xtest_bio)

        # Save feature stats
        feature_stats = {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'feature_names': feature_names
        }
        with open(os.path.join(features_dir, f"fold{fold+1}_feature_stats.json"), 'w') as f:
            json.dump(feature_stats, f, indent=2)

        # =====================================================================
        # MODE A: MODEL12 CONTROL
        # =====================================================================
        print(f"\n--- MODE A: model12_control ---")

        random.seed(fold_seed)
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        model_control = Crispr_SGRU_MODEL12_Control(learning_rate=0.0001)
        init_hash_control = get_model_weights_hash(model_control)
        control_param_count = model_control.count_params()
        print(f"Init hash: {init_hash_control}")
        print(f"Parameters: {control_param_count:,}")

        # Save control init hash
        config['init_hashes'][f'fold_{fold+1}_control'] = init_hash_control

        gen_control = StableMiningGenerator(
            pos_enc=Train_Pos_enc,
            bio_hard_enc=Neg_Bio_enc if len(Neg_Bio_enc) > 0 else Neg_Normal_enc,
            normal_enc=Neg_Normal_enc,
            batch_size=BATCH_SIZE,
            steps_per_epoch=STEPS_PER_EPOCH,
            seed=fold_seed
        )

        weight_path_control = os.path.join(save_dir_control, f"fold{fold + 1}.weights.h5")

        callbacks_control = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            ModelCheckpoint(weight_path_control, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
        ]

        print("Training...")
        history_control = model_control.fit(
            gen_control.flow(),
            validation_data=(Xval_enc, y_val_target),
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks_control,
            verbose=2
        )

        if not os.path.exists(weight_path_control):
            print(f"[ERROR] Checkpoint not found: {weight_path_control}")
            sys.exit(1)

        best_epoch_control = int(np.argmin(history_control.history['val_loss'])) + 1
        print(f"Best epoch (val_loss): {best_epoch_control}")

        # Evaluate
        model_control.load_weights(weight_path_control)

        # Validation metrics
        val_probs_control = model_control.predict(Xval_enc, verbose=0).ravel()
        thresh_control, _ = find_optimal_f1_threshold(y_val_gt, val_probs_control)
        val_pred_control = (val_probs_control >= thresh_control).astype(int)

        # Validation metrics only if both classes present
        val_metrics_control = {}
        if len(np.unique(y_val_gt)) == 2:
            fpr, tpr, _ = roc_curve(y_val_gt, val_probs_control)
            val_metrics_control['AUROC'] = auc(fpr, tpr)
            p_curve, r_curve, _ = precision_recall_curve(y_val_gt, val_probs_control)
            val_metrics_control['PRAUC'] = auc(r_curve, p_curve)
        else:
            val_metrics_control['AUROC'] = np.nan
            val_metrics_control['PRAUC'] = np.nan
        val_metrics_control['F1'] = f1_score(y_val_gt, val_pred_control, zero_division=0)
        val_metrics_control['Precision'] = precision_score(y_val_gt, val_pred_control, zero_division=0)
        val_metrics_control['Recall'] = recall_score(y_val_gt, val_pred_control, zero_division=0)
        val_metrics_control['MCC'] = matthews_corrcoef(y_val_gt, val_pred_control)

        # Test metrics
        test_probs_control = model_control.predict(Xtest_enc, verbose=0).ravel()
        test_pred_control = (test_probs_control >= thresh_control).astype(int)

        test_metrics_control = {}
        if len(np.unique(y_test_gt)) == 2:
            fpr, tpr, _ = roc_curve(y_test_gt, test_probs_control)
            test_metrics_control['AUROC'] = auc(fpr, tpr)
            p_curve, r_curve, _ = precision_recall_curve(y_test_gt, test_probs_control)
            test_metrics_control['PRAUC'] = auc(r_curve, p_curve)
        else:
            test_metrics_control['AUROC'] = np.nan
            test_metrics_control['PRAUC'] = np.nan
        test_metrics_control['F1'] = f1_score(y_test_gt, test_pred_control, zero_division=0)
        test_metrics_control['Precision'] = precision_score(y_test_gt, test_pred_control, zero_division=0)
        test_metrics_control['Recall'] = recall_score(y_test_gt, test_pred_control, zero_division=0)
        test_metrics_control['MCC'] = matthews_corrcoef(y_test_gt, test_pred_control)

        print(f"Control (Validation): AUROC={val_metrics_control['AUROC']:.4f}, PRAUC={val_metrics_control['PRAUC']:.4f}, F1={val_metrics_control['F1']:.4f}, Prec={val_metrics_control['Precision']:.4f}, Rec={val_metrics_control['Recall']:.4f}, MCC={val_metrics_control['MCC']:.4f} (thresh={thresh_control:.4f})")
        print(f"Control (Test):       AUROC={test_metrics_control['AUROC']:.4f}, PRAUC={test_metrics_control['PRAUC']:.4f}, F1={test_metrics_control['F1']:.4f}, Prec={test_metrics_control['Precision']:.4f}, Rec={test_metrics_control['Recall']:.4f}, MCC={test_metrics_control['MCC']:.4f}")

        results_control.append({
            'Fold': f'Fold_{fold+1}',
            'split': 'val',
            'Best_Epoch': best_epoch_control,
            'Threshold': thresh_control,
            'AUROC': val_metrics_control['AUROC'],
            'PRAUC': val_metrics_control['PRAUC'],
            'F1': val_metrics_control['F1'],
            'Precision': val_metrics_control['Precision'],
            'Recall': val_metrics_control['Recall'],
            'MCC': val_metrics_control['MCC']
        })

        results_control.append({
            'Fold': f'Fold_{fold+1}',
            'split': 'test',
            'Best_Epoch': best_epoch_control,
            'Threshold': thresh_control,
            'AUROC': test_metrics_control['AUROC'],
            'PRAUC': test_metrics_control['PRAUC'],
            'F1': test_metrics_control['F1'],
            'Precision': test_metrics_control['Precision'],
            'Recall': test_metrics_control['Recall'],
            'MCC': test_metrics_control['MCC']
        })

        # Save predictions
        pd.DataFrame({
            'source_id': val_source_ids,
            'y_true': y_val_gt,
            'y_prob': val_probs_control
        }).to_csv(os.path.join(predictions_dir, f"fold{fold+1}_control_val.csv"), index=False)

        pd.DataFrame({
            'source_id': test_source_ids,
            'y_true': y_test_gt,
            'y_prob': test_probs_control,
            'y_pred': test_pred_control
        }).to_csv(os.path.join(predictions_dir, f"fold{fold+1}_control_test.csv"), index=False)

        del model_control
        K.clear_session()
        gc.collect()

        # =====================================================================
        # MODE B: BIOFUSION
        # =====================================================================
        print(f"\n--- MODE B: model12_biofusion ---")

        random.seed(fold_seed)
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        model_biofusion = Crispr_SGRU_BioFusion(
            bio_feature_dim=feature_dim,
            learning_rate=0.0001
        )
        init_hash_biofusion = get_model_weights_hash(model_biofusion)
        biofusion_param_count = model_biofusion.count_params()
        print(f"Init hash: {init_hash_biofusion}")
        print(f"Parameters: {biofusion_param_count:,} (+{biofusion_param_count - control_param_count:,} vs control)")

        config['init_hashes'][f'fold_{fold+1}_biofusion'] = init_hash_biofusion

        gen_biofusion = BioFusionGenerator(
            pos_enc=Train_Pos_enc,
            pos_bio=Train_Pos_bio,
            bio_hard_enc=Neg_Bio_enc if len(Neg_Bio_enc) > 0 else Neg_Normal_enc,
            bio_hard_bio=Neg_Bio_bio if len(Neg_Bio_bio) > 0 else Neg_Normal_bio,
            normal_enc=Neg_Normal_enc,
            normal_bio=Neg_Normal_bio,
            batch_size=BATCH_SIZE,
            steps_per_epoch=STEPS_PER_EPOCH,
            seed=fold_seed
        )

        weight_path_biofusion = os.path.join(save_dir_biofusion, f"fold{fold + 1}.weights.h5")

        callbacks_biofusion = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            ModelCheckpoint(weight_path_biofusion, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
        ]

        print("Training...")
        history_biofusion = model_biofusion.fit(
            gen_biofusion.flow(),
            validation_data=((Xval_enc, Xval_bio), y_val_target),
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks_biofusion,
            verbose=2
        )

        if not os.path.exists(weight_path_biofusion):
            print(f"[ERROR] Checkpoint not found: {weight_path_biofusion}")
            sys.exit(1)

        best_epoch_biofusion = int(np.argmin(history_biofusion.history['val_loss'])) + 1
        print(f"Best epoch (val_loss): {best_epoch_biofusion}")

        # Evaluate
        model_biofusion.load_weights(weight_path_biofusion)

        # Validation metrics
        val_probs_biofusion = model_biofusion.predict((Xval_enc, Xval_bio), verbose=0).ravel()
        thresh_biofusion, _ = find_optimal_f1_threshold(y_val_gt, val_probs_biofusion)
        val_pred_biofusion = (val_probs_biofusion >= thresh_biofusion).astype(int)

        # Validation metrics only if both classes present
        val_metrics_biofusion = {}
        if len(np.unique(y_val_gt)) == 2:
            fpr, tpr, _ = roc_curve(y_val_gt, val_probs_biofusion)
            val_metrics_biofusion['AUROC'] = auc(fpr, tpr)
            p_curve, r_curve, _ = precision_recall_curve(y_val_gt, val_probs_biofusion)
            val_metrics_biofusion['PRAUC'] = auc(r_curve, p_curve)
        else:
            val_metrics_biofusion['AUROC'] = np.nan
            val_metrics_biofusion['PRAUC'] = np.nan
        val_metrics_biofusion['F1'] = f1_score(y_val_gt, val_pred_biofusion, zero_division=0)
        val_metrics_biofusion['Precision'] = precision_score(y_val_gt, val_pred_biofusion, zero_division=0)
        val_metrics_biofusion['Recall'] = recall_score(y_val_gt, val_pred_biofusion, zero_division=0)
        val_metrics_biofusion['MCC'] = matthews_corrcoef(y_val_gt, val_pred_biofusion)

        # Test metrics (using validation threshold)
        test_probs_biofusion = model_biofusion.predict((Xtest_enc, Xtest_bio_fold), verbose=0).ravel()
        test_pred_biofusion = (test_probs_biofusion >= thresh_biofusion).astype(int)

        test_metrics_biofusion = {}
        if len(np.unique(y_test_gt)) == 2:
            fpr, tpr, _ = roc_curve(y_test_gt, test_probs_biofusion)
            test_metrics_biofusion['AUROC'] = auc(fpr, tpr)
            p_curve, r_curve, _ = precision_recall_curve(y_test_gt, test_probs_biofusion)
            test_metrics_biofusion['PRAUC'] = auc(r_curve, p_curve)
        else:
            test_metrics_biofusion['AUROC'] = np.nan
            test_metrics_biofusion['PRAUC'] = np.nan
        test_metrics_biofusion['F1'] = f1_score(y_test_gt, test_pred_biofusion, zero_division=0)
        test_metrics_biofusion['Precision'] = precision_score(y_test_gt, test_pred_biofusion, zero_division=0)
        test_metrics_biofusion['Recall'] = recall_score(y_test_gt, test_pred_biofusion, zero_division=0)
        test_metrics_biofusion['MCC'] = matthews_corrcoef(y_test_gt, test_pred_biofusion)

        print(f"BioFusion (Validation): AUROC={val_metrics_biofusion['AUROC']:.4f}, PRAUC={val_metrics_biofusion['PRAUC']:.4f}, F1={val_metrics_biofusion['F1']:.4f}, Prec={val_metrics_biofusion['Precision']:.4f}, Rec={val_metrics_biofusion['Recall']:.4f}, MCC={val_metrics_biofusion['MCC']:.4f} (thresh={thresh_biofusion:.4f})")
        print(f"BioFusion (Test):       AUROC={test_metrics_biofusion['AUROC']:.4f}, PRAUC={test_metrics_biofusion['PRAUC']:.4f}, F1={test_metrics_biofusion['F1']:.4f}, Prec={test_metrics_biofusion['Precision']:.4f}, Rec={test_metrics_biofusion['Recall']:.4f}, MCC={test_metrics_biofusion['MCC']:.4f}")

        results_biofusion.append({
            'Fold': f'Fold_{fold+1}',
            'split': 'val',
            'Best_Epoch': best_epoch_biofusion,
            'Threshold': thresh_biofusion,
            'AUROC': val_metrics_biofusion['AUROC'],
            'PRAUC': val_metrics_biofusion['PRAUC'],
            'F1': val_metrics_biofusion['F1'],
            'Precision': val_metrics_biofusion['Precision'],
            'Recall': val_metrics_biofusion['Recall'],
            'MCC': val_metrics_biofusion['MCC']
        })

        results_biofusion.append({
            'Fold': f'Fold_{fold+1}',
            'split': 'test',
            'Best_Epoch': best_epoch_biofusion,
            'Threshold': thresh_biofusion,
            'AUROC': test_metrics_biofusion['AUROC'],
            'PRAUC': test_metrics_biofusion['PRAUC'],
            'F1': test_metrics_biofusion['F1'],
            'Precision': test_metrics_biofusion['Precision'],
            'Recall': test_metrics_biofusion['Recall'],
            'MCC': test_metrics_biofusion['MCC']
        })

        # Save predictions
        pd.DataFrame({
            'source_id': val_source_ids,
            'y_true': y_val_gt,
            'y_prob': val_probs_biofusion
        }).to_csv(os.path.join(predictions_dir, f"fold{fold+1}_biofusion_val.csv"), index=False)

        pd.DataFrame({
            'source_id': test_source_ids,
            'y_true': y_test_gt,
            'y_prob': test_probs_biofusion,
            'y_pred': test_pred_biofusion
        }).to_csv(os.path.join(predictions_dir, f"fold{fold+1}_biofusion_test.csv"), index=False)

        del model_biofusion
        K.clear_session()
        gc.collect()

        # Update config after fold
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    # =========================================================================
    # COMPUTE MEAN AND STD
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPUTING MEAN ± STD (VALIDATION AND TEST SPLITS)")
    print(f"{'='*70}\n")

    df_control = pd.DataFrame(results_control)
    df_biofusion = pd.DataFrame(results_biofusion)

    metrics_cols = ['Best_Epoch', 'Threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC']

    # Validate fold rows and compute Mean/Std independently for val and test.
    # The lowercase ``split`` column is part of the machine-readable schema.
    expected_folds = [f'Fold_{i + 1}' for i in range(num_folds)]
    if 'split' not in df_control.columns or 'split' not in df_biofusion.columns:
        raise RuntimeError("Internal results schema must contain lowercase 'split'")

    def summarize_by_split(df, split_name):
        split_df = df[df['split'] == split_name].copy()
        if set(split_df['Fold'].tolist()) != set(expected_folds) or len(split_df) != num_folds:
            raise RuntimeError(
                f"Expected exactly {num_folds} {split_name} fold rows, "
                f"got {len(split_df)}: {split_df['Fold'].tolist()}"
            )
        split_df = split_df.set_index('Fold').loc[expected_folds].reset_index()
        mean_row = {'Fold': 'Mean', 'split': split_name}
        std_row = {'Fold': 'Std', 'split': split_name}
        for col in metrics_cols:
            mean_row[col] = split_df[col].mean()
            std_row[col] = split_df[col].std(ddof=0)
        return split_df, mean_row, std_row

    control_split_frames = {}
    biofusion_split_frames = {}
    control_summary_rows = []
    biofusion_summary_rows = []
    for split_name in ('val', 'test'):
        control_split_df, control_mean, control_std = summarize_by_split(df_control, split_name)
        biofusion_split_df, biofusion_mean, biofusion_std = summarize_by_split(df_biofusion, split_name)
        control_split_frames[split_name] = control_split_df
        biofusion_split_frames[split_name] = biofusion_split_df
        control_summary_rows.extend([control_mean, control_std])
        biofusion_summary_rows.extend([biofusion_mean, biofusion_std])

    df_control_val = control_split_frames['val']
    df_biofusion_val = biofusion_split_frames['val']
    mean_control = next(row for row in control_summary_rows if row['split'] == 'val' and row['Fold'] == 'Mean')
    std_control = next(row for row in control_summary_rows if row['split'] == 'val' and row['Fold'] == 'Std')
    mean_biofusion = next(row for row in biofusion_summary_rows if row['split'] == 'val' and row['Fold'] == 'Mean')
    std_biofusion = next(row for row in biofusion_summary_rows if row['split'] == 'val' and row['Fold'] == 'Std')

    df_control = pd.concat([df_control, pd.DataFrame(control_summary_rows)], ignore_index=True)
    df_biofusion = pd.concat([df_biofusion, pd.DataFrame(biofusion_summary_rows)], ignore_index=True)

    # Save per-fold results
    control_csv = os.path.join(run_dir, f"E3_p12_control_per_fold.csv")
    biofusion_csv = os.path.join(run_dir, f"E3_biofusion_per_fold.csv")

    df_control.to_csv(control_csv, index=False)
    df_biofusion.to_csv(biofusion_csv, index=False)

    print("P12 Control (all folds + mean/std):")
    print(df_control.to_string(index=False))
    print(f"\nSaved: {control_csv}")

    print("\nP12 BioFusion (all folds + mean/std):")
    print(df_biofusion.to_string(index=False))
    print(f"\nSaved: {biofusion_csv}")

    # Paired differences (validation split only)
    paired_diffs = []
    for i in range(5):
        diff_row = {'Fold': f'Fold_{i+1}', 'split': 'val'}
        for col in metrics_cols:
            diff_row[col] = df_biofusion_val.iloc[i][col] - df_control_val.iloc[i][col]
        paired_diffs.append(diff_row)

    mean_diff = {'Fold': 'Mean_Diff', 'split': 'val'}
    for col in metrics_cols:
        diffs = [paired_diffs[i][col] for i in range(5)]
        mean_diff[col] = np.mean(diffs)

    paired_diffs.append(mean_diff)
    df_paired = pd.DataFrame(paired_diffs)

    paired_csv = os.path.join(run_dir, f"E3_paired_differences.csv")
    df_paired.to_csv(paired_csv, index=False)

    print("\nPaired differences (BioFusion - Control, validation split only):")
    print(df_paired.to_string(index=False))
    print(f"\nSaved: {paired_csv}")

    # Combined summary (includes both val and test)
    summary_rows = []

    for i in range(len(results_control)):
        row = results_control[i].copy()
        row['Mode'] = 'p12_control'
        summary_rows.append(row)

    summary_rows.append({
        'Fold': 'Mean',
        'split': 'val',
        'Mode': 'p12_control',
        **{k: mean_control[k] for k in metrics_cols}
    })
    summary_rows.append({
        'Fold': 'Std',
        'split': 'val',
        'Mode': 'p12_control',
        **{k: std_control[k] for k in metrics_cols}
    })
    control_test_mean = next(row for row in control_summary_rows if row['split'] == 'test' and row['Fold'] == 'Mean')
    control_test_std = next(row for row in control_summary_rows if row['split'] == 'test' and row['Fold'] == 'Std')
    summary_rows.append({
        'Fold': 'Mean',
        'split': 'test',
        'Mode': 'p12_control',
        **{k: control_test_mean[k] for k in metrics_cols}
    })
    summary_rows.append({
        'Fold': 'Std',
        'split': 'test',
        'Mode': 'p12_control',
        **{k: control_test_std[k] for k in metrics_cols}
    })

    for i in range(len(results_biofusion)):
        row = results_biofusion[i].copy()
        row['Mode'] = 'p12_biofusion'
        summary_rows.append(row)

    summary_rows.append({
        'Fold': 'Mean',
        'split': 'val',
        'Mode': 'p12_biofusion',
        **{k: mean_biofusion[k] for k in metrics_cols}
    })
    summary_rows.append({
        'Fold': 'Std',
        'split': 'val',
        'Mode': 'p12_biofusion',
        **{k: std_biofusion[k] for k in metrics_cols}
    })
    biofusion_test_mean = next(row for row in biofusion_summary_rows if row['split'] == 'test' and row['Fold'] == 'Mean')
    biofusion_test_std = next(row for row in biofusion_summary_rows if row['split'] == 'test' and row['Fold'] == 'Std')
    summary_rows.append({
        'Fold': 'Mean',
        'split': 'test',
        'Mode': 'p12_biofusion',
        **{k: biofusion_test_mean[k] for k in metrics_cols}
    })
    summary_rows.append({
        'Fold': 'Std',
        'split': 'test',
        'Mode': 'p12_biofusion',
        **{k: biofusion_test_std[k] for k in metrics_cols}
    })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(run_dir, f"E3_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    print(f"\nSaved summary: {summary_csv}")

    # Final config save
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print("E3 COMPLETE")
    print(f"{'='*70}")
    print(f"\nPRIMARY RESULT = Mean ± Std of 5 validation folds for each mode")
    print(f"  Feature selection uses VALIDATION metrics only")
    print(f"COMPARISON = Paired differences (biofusion - control) on validation")
    print(f"\nAll results include both validation and test splits:")
    print(f"  - Validation: Used for feature/threshold selection")
    print(f"  - Test: Final evaluation with validation threshold (locked)")
    print(f"\nFeature set: {feature_set.upper()} ({feature_dim} features)")
    print(f"Feature stats saved per fold in {features_dir}")
    print(f"Predictions saved in {predictions_dir}")
    print(f"\nTotal time: {datetime.now() - start_time}")
