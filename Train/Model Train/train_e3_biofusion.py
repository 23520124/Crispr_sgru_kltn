"""
E3 — Biological Feature Fusion Experiment

Train P12 + unified biological features (BioFusion only).

Feature set:
- enhanced: unified mismatch, substitution, PAM, GC and positional indel features

Mục tiêu:
- Kiểm tra biological features có cải thiện PRAUC/F1 trên II5/II6
- Giữ hiệu suất trên K562/HEK/BE3
- So sánh công bằng: cùng split, seed, sampling, epochs, checkpoint rule

Không thay đổi:
- MODEL12/P12 backbone and five-fold split protocol
- Weighted BCE-Dice loss (alpha=0.7, beta=2.0)
- BioFusionGenerator (64 pos, 16 hard, 176 normal)
- Batch size 256, 250 steps/epoch, epochs 35
- Position-wise biological channels fused before BiGRU-2
- Adam 1e-4, ReduceLROnPlateau
- Checkpoint theo minimum val_loss
- F1 threshold cumulative (E0), PRAUC from continuous probabilities
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
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
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

from biological_features import (
    BiologicalFeatureExtractor,
    extract_features_batch,
    extract_position_features_batch,
)
from MODEL_E3_BioFusion import Crispr_SGRU_BioFusion


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
    """Count aligned protospacer mismatches, excluding the terminal PAM."""
    proto1 = str(seq1).strip().upper()[:-3]
    proto2 = str(seq2).strip().upper()[:-3]
    width = max(len(proto1), len(proto2))
    if width > 21:
        raise ValueError("Expected at most 21 aligned protospacer columns before NGG PAM")
    proto1 = "-" * (width - len(proto1)) + proto1
    proto2 = "-" * (width - len(proto2)) + proto2
    return sum(1 for a, b in zip(proto1, proto2) if a != b)


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
    """Select the F1-optimal probability threshold on validation only."""
    order = np.argsort(y_probs)
    probs = y_probs[order]
    labels = y_true[order]
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))
    cum_pos = np.cumsum(labels == 1)
    unique_probs, first_indices = np.unique(probs, return_index=True)
    candidates = []

    def f1(tp, fp, fn):
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    for i, threshold in enumerate(unique_probs):
        idx = first_indices[i]
        below_pos = int(cum_pos[idx - 1]) if idx > 0 else 0
        below_neg = idx - below_pos
        candidates.append((f1(total_pos - below_pos, total_neg - below_neg, below_pos), float(threshold)))
    candidates.append((f1(0, 0, total_pos), np.inf))
    best_score = max(score for score, _ in candidates)
    return min(threshold for score, threshold in candidates if score == best_score), best_score


def positive_probability(model_output):
    """Extract the positive-class score without flattening class outputs."""
    output = np.asarray(model_output)
    if output.ndim == 2 and output.shape[1] == 1:
        return output[:, 0].astype(np.float64)
    raise ValueError(f"E3 BioFusion expects output shape (N, 1), got {output.shape}")


class BioFusionGenerator:
    """Generator yields sequence, global biology and position-wise inputs."""
    def __init__(self, pos_enc, pos_bio, pos_position,
                 bio_hard_enc, bio_hard_bio, bio_hard_position,
                 normal_enc, normal_bio, normal_position,
                 batch_size=256, steps_per_epoch=120, seed=42):
        self.pos_enc = pos_enc
        self.pos_bio = pos_bio
        self.pos_position = pos_position
        self.bio_hard_enc = bio_hard_enc if len(bio_hard_enc) > 0 else normal_enc
        self.bio_hard_bio = bio_hard_bio if len(bio_hard_bio) > 0 else normal_bio
        self.bio_hard_position = bio_hard_position if len(bio_hard_position) > 0 else normal_position
        self.normal_enc = normal_enc
        self.normal_bio = normal_bio
        self.normal_position = normal_position
        if len(self.normal_enc) == 0:
            self.normal_enc = self.bio_hard_enc
            self.normal_bio = self.bio_hard_bio
            self.normal_position = self.bio_hard_position
        if len(self.pos_enc) == 0 or len(self.bio_hard_enc) == 0 or len(self.normal_enc) == 0:
            raise ValueError("Training fold must contain positive and negative samples")
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.n_pos = 64
        self.n_hard = 16
        self.n_norm = 176

        self.rng = np.random.RandomState(seed)

    def flow(self):
        while True:
            for _ in range(self.steps_per_epoch):
                pos_idx = self.rng.choice(len(self.pos_enc), self.n_pos, replace=True)
                b_pos_seq = self.pos_enc[pos_idx]
                b_pos_bio = self.pos_bio[pos_idx]
                b_pos_position = self.pos_position[pos_idx]
                hard_idx = self.rng.choice(len(self.bio_hard_enc), self.n_hard, replace=True)
                norm_idx = self.rng.choice(len(self.normal_enc), self.n_norm, replace=True)
                X_seq = np.concatenate([b_pos_seq, self.bio_hard_enc[hard_idx], self.normal_enc[norm_idx]], axis=0)
                X_bio = np.concatenate([b_pos_bio, self.bio_hard_bio[hard_idx], self.normal_bio[norm_idx]], axis=0)
                X_position = np.concatenate([b_pos_position, self.bio_hard_position[hard_idx], self.normal_position[norm_idx]], axis=0)
                y_batch_scalar = np.array(
                    [1] * self.n_pos + [0] * (self.n_hard + self.n_norm),
                    dtype=np.int32,
                )
                y_batch = y_batch_scalar.astype(np.float32).reshape(-1, 1)

                p = self.rng.permutation(len(y_batch))
                yield (X_seq[p], X_bio[p], X_position[p]), y_batch[p]


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E3 BioFusion-only training using preprocessed datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_e3_biofusion.py --dataset II5 --feature-set enhanced
  python train_e3_biofusion.py --dataset CHANGE-seq --feature-set enhanced

Feature sets:
  enhanced: one unified biological feature set

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
        default="enhanced",
        choices=["enhanced"],
        help="Unified biological feature set (default: enhanced)"
    )

    args = parser.parse_args()
    data_name = args.dataset
    feature_set = "enhanced"

    dataset_path_map = {
        "BE3": os.path.join(BASE_DIR, "Datasets", "preprocessed", "BE3.csv"),
        "I1":  os.path.join(BASE_DIR, "Datasets", "preprocessed", "I1.csv"),
        "I2":  os.path.join(BASE_DIR, "Datasets", "preprocessed_conflict_excluded", "I2.csv"),
        "II6": os.path.join(BASE_DIR, "Datasets", "preprocessed", "II6.csv"),
        "II5": os.path.join(BASE_DIR, "Datasets", "preprocessed", "II5.csv"),
        "K562": os.path.join(BASE_DIR, "Datasets", "preprocessed", "K562.csv"),
        "hek": os.path.join(BASE_DIR, "Datasets", "preprocessed", "HEK293t.csv"),
        "CHANGE-seq": os.path.join(BASE_DIR, "Datasets", "preprocessed", "CHANGE-seq.csv"),
    }

    file_path = dataset_path_map.get(data_name)
    if file_path is None or not os.path.isfile(file_path):
        print(f"[ERROR] Preprocessed dataset not found: {file_path}")
        sys.exit(1)
    preprocessing_metadata_path = os.path.splitext(file_path)[0] + ".json"
    if not os.path.isfile(preprocessing_metadata_path):
        print(f"[ERROR] Preprocessing metadata not found: {preprocessing_metadata_path}")
        sys.exit(1)

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, "weights", f"{data_name}_E3_{feature_set}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    save_dir_biofusion = os.path.join(run_dir, "biofusion")
    history_dir = os.path.join(run_dir, "history")
    features_dir = os.path.join(run_dir, "features")
    predictions_dir = os.path.join(run_dir, "predictions")

    for d in [save_dir_biofusion, history_dir, features_dir, predictions_dir]:
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
    sequence_column_order = "preprocessed_column_0_sgrna_column_1_offtarget"
    Negative = np.array(Negative, dtype=object)
    Positive = np.array(Positive, dtype=object)
    total_pos = len(Positive)
    total_neg = len(Negative)

    dataset_file_hash = compute_file_hash(file_path)
    preprocessing_metadata_hash = compute_file_hash(preprocessing_metadata_path)
    print(f"Data: {total_pos} Pos | {total_neg} Neg")
    print(f"Dataset file hash: {dataset_file_hash}")

    # Split 15% blind test (same as P12)
    Train_Pool_Neg, Test_Neg = train_test_split(Negative, test_size=0.15, random_state=258)
    Train_Pool_Pos, Test_Pos = train_test_split(Positive, test_size=0.15, random_state=258)

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
    Xtest_position = extract_position_features_batch(Xtest_raw.tolist())
    y_test_gt = np.array([1 if float(x) > 0.0 else 0 for x in Xtest_df.iloc[:, 2]])

    print(f"Test: {len(y_test_gt)} samples, {np.sum(y_test_gt)} positives\n")

    BATCH_SIZE = 256
    EPOCHS = 35
    num_folds = 5

    train_pool = np.vstack((Train_Pool_Neg, Train_Pool_Pos))
    train_pool_y = np.array([0] * len(Train_Pool_Neg) + [1] * len(Train_Pool_Pos))
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=258)
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
        'input_file': os.path.abspath(file_path),
        'preprocessing_metadata_file': os.path.abspath(preprocessing_metadata_path),
        'preprocessing_metadata_hash': preprocessing_metadata_hash,
        'sequence_column_order': sequence_column_order,
        'source_id_policy': 'row index in the preprocessed CSV; original source row IDs are recorded in preprocessing metadata only for conflict audits',
        'dataset_file_hash': dataset_file_hash,
        'train_pool_neg_hash': train_pool_neg_hash,
        'test_neg_hash': test_neg_hash,
        'train_pool_pos_hash': train_pool_pos_hash,
        'test_pos_hash': test_pos_hash,
        'feature_set': 'enhanced',
        'feature_dim': feature_dim,
        'feature_names': feature_names,
        'position_feature_dim': 21,
        'position_feature_channels': [
            'ordered_base_pair_one_hot_16', 'mismatch', 'transition',
            'transversion', 'gap', 'cut_site_17_19'
        ],
        'position_input_used': True,
        'position_fusion': 'concatenate raw 21 position channels with raw sequence and BiGRU-1 output before BiGRU-2',
        'bio_fusion': 'global biological features use residual gated calibration at the classifier',
        'run_id': run_id,
        'total_pos': total_pos,
        'total_neg': total_neg,
        'test_size': 0.15,
        'num_folds': num_folds,
        'batch_size': BATCH_SIZE,
        'steps_per_epoch': 250,
        'epochs': EPOCHS,
        'model': f'P12_BioFusion_{feature_set}',
        'architecture_ablation': 'use raw 21-channel position input before BiGRU-2; Flatten classifier path; bio_gate uses the Keras default zero initializer',
        'bio_gate_bias_initializer': 'zeros (Keras default)',
        'loss': 'WeightedBCEDiceLoss_alpha0.7_beta2.0',
        'optimizer': 'Adam_1e-4',
        'generator': 'BioFusionGenerator_64pos_16hard_176norm',
        'checkpoint_rule': 'val_loss_min',
        'threshold_rule': 'F1_optimal_cumulative',
        'bio_hard_rule': 'protospacer_mismatch_le_3; quota 16 per batch',
        'steps_policy': 'fixed 250 steps per epoch',
        'feature_policy': 'single unified enhanced feature set; legacy names resolve to this same set for caller compatibility',
        'pam_policy': 'paper section 4.1: NGG PAM; encode only the terminal variable N base (4 features), omit invariant GG',
        'mismatch_policy': 'adds positions 1-3 and 17-19, transition/transversion counts, and maximum consecutive mismatch run',
        'indel_policy': 'uses explicit _/- positions, counts, and distance to the terminal PAM',
        'sequence_coordinate_policy': 'right-align 23/24 nt sequences to the terminal PAM; protospacer features exclude final 3 PAM bases',
        'tensorflow_version': tf_version,
        'keras_version': keras_version,
        'code_hash': code_hash,
        'fold_seeds': {},
        'init_hashes': {}
    }

    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Results storage
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
        steps_per_epoch = 250

        print(
            f"Train: {len(Train_Pos)} Pos | {len(Train_Neg)} Neg | "
            f"{steps_per_epoch} steps/epoch"
        )

        # Encoding (sequence)
        Train_Pos_enc = encoding(pd.DataFrame([row[:3] for row in Train_Pos])).reshape((-1, 1, 24, 7)).astype("float32")
        bio_mask = np.array([count_mismatches(r[0], r[1]) <= 3 for r in Train_Neg])
        Neg_Bio = Train_Neg[bio_mask]
        Neg_Normal = Train_Neg[~bio_mask]
        Neg_Bio_enc = encoding(pd.DataFrame([row[:3] for row in Neg_Bio])).reshape((-1, 1, 24, 7)).astype("float32") if len(Neg_Bio) > 0 else np.empty((0, 1, 24, 7), dtype="float32")
        Neg_Normal_enc = encoding(pd.DataFrame([row[:3] for row in Neg_Normal])).reshape((-1, 1, 24, 7)).astype("float32")
        Train_Pos_position = extract_position_features_batch(Train_Pos.tolist())
        Neg_Bio_position = extract_position_features_batch(Neg_Bio.tolist()) if len(Neg_Bio) > 0 else np.empty((0, 24, 21), dtype="float32")
        Neg_Normal_position = extract_position_features_batch(Neg_Normal.tolist())

        Xval_enc = encoding(pd.DataFrame([row[:3] for row in Val_raw])).reshape((-1, 1, 24, 7)).astype("float32")
        Xval_position = extract_position_features_batch(Val_raw.tolist())
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

        print(f"\n--- BIOFUSION ---")

        random.seed(fold_seed)
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        model_biofusion = Crispr_SGRU_BioFusion(
            bio_feature_dim=feature_dim,
            position_feature_dim=21,
            learning_rate=0.0001
        )
        init_hash_biofusion = get_model_weights_hash(model_biofusion)
        biofusion_param_count = model_biofusion.count_params()
        print(f"Init hash: {init_hash_biofusion}")
        print(f"Parameters: {biofusion_param_count:,}")

        config['init_hashes'][f'fold_{fold+1}_biofusion'] = init_hash_biofusion

        gen_biofusion = BioFusionGenerator(
            pos_enc=Train_Pos_enc,
            pos_bio=Train_Pos_bio,
            pos_position=Train_Pos_position,
            bio_hard_enc=Neg_Bio_enc,
            bio_hard_bio=Neg_Bio_bio,
            bio_hard_position=Neg_Bio_position,
            normal_enc=Neg_Normal_enc,
            normal_bio=Neg_Normal_bio,
            normal_position=Neg_Normal_position,
            batch_size=BATCH_SIZE,
            steps_per_epoch=steps_per_epoch,
            seed=fold_seed
        )

        weight_path_biofusion = os.path.join(save_dir_biofusion, f"fold{fold + 1}.weights.h5")

        callbacks_biofusion = [
            ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            ModelCheckpoint(weight_path_biofusion, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
        ]

        print("Training...")
        history_biofusion = model_biofusion.fit(
            gen_biofusion.flow(),
            validation_data=((Xval_enc, Xval_bio, Xval_position), y_val_target),
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks_biofusion,
            verbose=2
        )

        if not os.path.exists(weight_path_biofusion):
            print(f"[ERROR] Checkpoint not found: {weight_path_biofusion}")
            sys.exit(1)

        pd.DataFrame(history_biofusion.history).assign(
            epoch=np.arange(1, len(history_biofusion.epoch) + 1)
        ).to_csv(os.path.join(history_dir, f"fold{fold+1}_biofusion_history.csv"), index=False)
        best_epoch_biofusion = int(np.argmin(history_biofusion.history['val_loss'])) + 1
        print(f"Best epoch (val_loss): {best_epoch_biofusion}")

        # Evaluate
        model_biofusion.load_weights(weight_path_biofusion)

        # Validation metrics
        val_output_biofusion = model_biofusion.predict((Xval_enc, Xval_bio, Xval_position), verbose=0)
        val_probs_biofusion = positive_probability(val_output_biofusion)
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

        # Test metrics use the threshold selected on validation only.
        test_output_biofusion = model_biofusion.predict((Xtest_enc, Xtest_bio_fold, Xtest_position), verbose=0)
        test_probs_biofusion = positive_probability(test_output_biofusion)
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

    print(f"\n{'='*70}")
    print("BIOFUSION MEAN ± STD (VALIDATION AND TEST)")
    print(f"{'='*70}\n")

    df_biofusion = pd.DataFrame(results_biofusion)
    metrics_cols = ['Best_Epoch', 'Threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC']
    expected_folds = [f'Fold_{i + 1}' for i in range(num_folds)]
    summary_rows = []
    for split_name in ('val', 'test'):
        split_df = df_biofusion[df_biofusion['split'] == split_name].copy()
        if set(split_df['Fold']) != set(expected_folds) or len(split_df) != num_folds:
            raise RuntimeError(
                f"Expected exactly {num_folds} BioFusion {split_name} fold rows; "
                f"got {len(split_df)}"
            )
        split_df = split_df.set_index('Fold').loc[expected_folds].reset_index()
        mean_row = {'Fold': 'Mean', 'split': split_name}
        std_row = {'Fold': 'Std', 'split': split_name}
        for metric in metrics_cols:
            mean_row[metric] = split_df[metric].mean()
            std_row[metric] = split_df[metric].std(ddof=0)
        summary_rows.extend([mean_row, std_row])

    df_biofusion = pd.concat([df_biofusion, pd.DataFrame(summary_rows)], ignore_index=True)
    per_fold_csv = os.path.join(run_dir, "E3_biofusion_per_fold.csv")
    summary_csv = os.path.join(run_dir, "E3_biofusion_summary.csv")
    df_biofusion.to_csv(per_fold_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(df_biofusion.to_string(index=False))
    print(f"\nSaved: {per_fold_csv}")
    print(f"Saved: {summary_csv}")

    # Final config save
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print("E3 COMPLETE")
    print(f"{'='*70}")
    print(f"\nPRIMARY RESULT = BioFusion mean ± std across 5 validation folds")
    print(f"  Decision rule: positive class score > negative class score")
    print(f"\nAll results include both validation and test splits:")
    print(f"  - Validation: ranking and class-decision evaluation")
    print(f"  - Test: final evaluation with the locked MODEL.py decision rule")
    print(f"\nFeature set: {feature_set.upper()} ({feature_dim} features)")
    print(f"Feature stats saved per fold in {features_dir}")
    print(f"Predictions saved in {predictions_dir}")
    print(f"\nTotal time: {datetime.now() - start_time}")
