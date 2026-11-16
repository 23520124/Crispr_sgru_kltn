"""E3 BioFusion with the K562 20261030_213631 training configuration.

Five independent folds on the 85% development pool select checkpoints by
minimum val_loss. Pool their held-out Validation predictions to lock one
max-F1 threshold, then average the five checkpoint probabilities on the shared
15% Test. No Test inference or Test metrics run during cross-validation.
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
    """Legacy hard-pool rule used by the reference run: first 20 columns."""
    limit = min(20, len(seq1), len(seq2))
    return sum(a != b for a, b in zip(seq1[:limit], seq2[:limit]))


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
    y_true = np.asarray(y_true, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    if y_true.ndim != 1 or y_probs.shape != y_true.shape or not len(y_true):
        raise ValueError("Expected nonempty, equally sized label and probability vectors")
    if not np.isin(y_true, [0, 1]).all() or not np.isfinite(y_probs).all():
        raise ValueError("Expected binary labels and finite probabilities")
    order = np.argsort(y_probs)
    probs = y_probs[order]
    labels = y_true[order]
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))
    cum_pos = np.cumsum(labels == 1)
    unique_probs, first_indices = np.unique(probs, return_index=True)
    candidates = []

    def f1(tp, fp, fn):
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0

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
                 batch_size=256, steps_per_epoch=250, seed=42):
        if batch_size != 256:
            raise ValueError("Reference sampler requires batch_size=256 (64/16/176)")
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
def compute_metrics(y_true, probabilities, threshold):
    """Score locked predictions; never select a threshold here."""
    predicted = (probabilities >= threshold).astype(int)
    metrics = {}
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        metrics['AUROC'] = float(auc(fpr, tpr))
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        metrics['PRAUC'] = float(auc(recall, precision))
    else:
        metrics.update(AUROC=float('nan'), PRAUC=float('nan'))
    metrics.update(
        F1=float(f1_score(y_true, predicted, zero_division=0)),
        Precision=float(precision_score(y_true, predicted, zero_division=0)),
        Recall=float(recall_score(y_true, predicted, zero_division=0)),
        MCC=float(matthews_corrcoef(y_true, predicted)),
    )
    for name, true_label, predicted_label in [('TP',1,1),('FN',1,0),('FP',0,1),('TN',0,0)]:
        metrics[name] = int(np.sum((y_true == true_label) & (predicted == predicted_label)))
    return metrics


def predict_ensemble(fold_artifacts, sequence, raw_bio, position, learning_rate=1e-4):
    """Average five checkpoint probabilities, each using its own Train scaler.

    Test labels are absent. Call after CV and OOF threshold selection finish.
    """
    if len(fold_artifacts) != 5:
        raise ValueError("Ensemble requires all five fold checkpoints")
    for artifact in fold_artifacts:
        for key in ('weights_path', 'scaler_path'):
            if not os.path.isfile(artifact[key]):
                raise FileNotFoundError(artifact[key])
    probability_sum = np.zeros(len(sequence), dtype=np.float64)
    for fold_number, artifact in enumerate(fold_artifacts, start=1):
        with open(artifact['scaler_path'], encoding='utf-8') as handle:
            stats = json.load(handle)
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(stats['mean'], dtype=np.float64)
        scaler.scale_ = np.asarray(stats['std'], dtype=np.float64)
        scaler.n_features_in_ = len(scaler.mean_)
        if (scaler.mean_.shape != (raw_bio.shape[1],)
                or scaler.scale_.shape != scaler.mean_.shape
                or not np.isfinite(scaler.mean_).all()
                or not np.isfinite(scaler.scale_).all()
                or np.any(scaler.scale_ <= 0)):
            raise ValueError(f"Invalid Train scaler for fold {fold_number}")
        scaled_bio = scaler.transform(raw_bio)
        K.clear_session()
        model = Crispr_SGRU_BioFusion(
            bio_feature_dim=raw_bio.shape[1], position_feature_dim=position.shape[2],
            learning_rate=learning_rate,
        )
        try:
            model.load_weights(artifact['weights_path'])
            output = model.predict((sequence, scaled_bio, position), verbose=0)
            probabilities = positive_probability(output)
            if probabilities.shape != probability_sum.shape or not np.isfinite(probabilities).all():
                raise RuntimeError(f"Invalid ensemble probabilities for fold {fold_number}")
            probability_sum += probabilities
        finally:
            del model, scaled_bio
            K.clear_session()
            gc.collect()
    return probability_sum / len(fold_artifacts)


def main(argv=None):
    training_profile_path = os.path.join(CURRENT_DIR, "biofusion_training_profile.json")
    with open(training_profile_path, encoding="utf-8") as handle:
        training_profile = json.load(handle)
    if training_profile["selection_rule"] != "val_loss_f1":
        raise ValueError("Reference training profile must use val_loss_f1")
    if training_profile["split_protocol"] != "classwise_holdout_stratified_fivefold":
        raise ValueError("Reference training profile must use the original row-level split")
    if (training_profile["hard_negative_mode"] != "static"
            or training_profile["hard_negative_per_batch"] != 16):
        raise ValueError("Reference sampler requires static hard negatives with quota 16")
    validation_batch_size = int(training_profile["validation_batch_size"])
    if validation_batch_size < 1:
        raise ValueError("Validation batch size must be positive")
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

    parser.add_argument("--csv", default=None, help="Optional input CSV override")
    parser.add_argument("--split-seed", type=int, default=training_profile["split_seed"])
    parser.add_argument("--train-seed", type=int, default=training_profile["train_seed"])
    parser.add_argument("--epochs", type=int, default=training_profile["epochs"])
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=training_profile["learning_rate"])
    args = parser.parse_args(argv)
    canonical_name = "HEK293t" if args.dataset == "hek" else args.dataset
    if args.steps_per_epoch is None:
        args.steps_per_epoch = training_profile["steps_per_epoch"][canonical_name]
    if args.epochs < 1 or args.steps_per_epoch < 1:
        parser.error("Epochs and steps per epoch must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if not 0 <= args.split_seed <= 2**32 - 1 or not 0 <= args.train_seed <= 2**32 - 401:
        parser.error("Seeds must fit NumPy's 32-bit range, including all fold offsets")
    data_name = args.dataset
    feature_set = "enhanced"

    dataset_root = os.path.join(BASE_DIR, training_profile["dataset_directory"])
    file_path = os.path.abspath(args.csv or os.path.join(dataset_root, canonical_name + ".csv"))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Preprocessed dataset not found: {file_path}")
    preprocessing_metadata_path = os.path.splitext(file_path)[0] + ".json"
    if not os.path.isfile(preprocessing_metadata_path):
        print(f"[ERROR] Preprocessing metadata not found: {preprocessing_metadata_path}")
        sys.exit(1)

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, "weights", f"{data_name}_E3_{feature_set}_ensemble_{run_id}")
    os.makedirs(run_dir, exist_ok=False)

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
    Train_Pool_Neg, Test_Neg = train_test_split(Negative, test_size=0.15, random_state=args.split_seed)
    Train_Pool_Pos, Test_Pos = train_test_split(Positive, test_size=0.15, random_state=args.split_seed)

    train_pool_neg_hash = get_split_hash(Train_Pool_Neg)
    test_neg_hash = get_split_hash(Test_Neg)
    train_pool_pos_hash = get_split_hash(Train_Pool_Pos)
    test_pos_hash = get_split_hash(Test_Pos)

    print(f"Train pool neg hash: {train_pool_neg_hash}")
    print(f"Test neg hash: {test_neg_hash}")
    print(f"Train pool pos hash: {train_pool_pos_hash}")
    print(f"Test pos hash: {test_pos_hash}")

    # Keep Test raw until all folds finish and the OOF threshold is locked.
    Xtest_raw = np.vstack((Test_Neg, Test_Pos))
    np.random.RandomState(args.split_seed).shuffle(Xtest_raw)
    test_source_ids = [int(row[3]) for row in Xtest_raw]
    BATCH_SIZE = training_profile["batch_size"]
    EPOCHS = args.epochs
    num_folds = 5

    train_pool = np.vstack((Train_Pool_Neg, Train_Pool_Pos))
    train_pool_y = np.array([0] * len(Train_Pool_Neg) + [1] * len(Train_Pool_Pos))
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.split_seed)
    fold_splits = list(skf.split(train_pool, train_pool_y))

    # Hash the active training profile, encoder and three E3 sources.
    code_files = [
        __file__,
        training_profile_path,
        Encoder_sgRNA_off.__file__,
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
        'steps_per_epoch': args.steps_per_epoch,
        'epochs': EPOCHS,
        'model': f'P12_BioFusion_{feature_set}',
        'architecture_ablation': 'use raw 21-channel position input before BiGRU-2; Flatten classifier path; bio_gate uses the Keras default zero initializer',
        'bio_gate_bias_initializer': 'zeros (Keras default)',
        'loss': 'WeightedBCEDiceLoss_alpha0.7_beta2.0',
        'optimizer': f'Adam_{args.learning_rate}',
        'learning_rate': args.learning_rate,
        'gradient_clipping': None,
        'validation_batch_size': validation_batch_size,
        'validation_shuffle': False,
        'generator': 'BioFusionGenerator_64pos_16hard_176norm',
        'checkpoint_rule': 'val_loss_min',
        'threshold_rule': 'max F1 on pooled out-of-fold Validation probabilities; ties: lowest threshold',
        'fold_validation_threshold_rule': 'F1_optimal_cumulative',
        'learning_rate_monitor': 'val_loss_min',
        'split_seed': args.split_seed,
        'train_seed_base': args.train_seed,
        'split_protocol': training_profile['split_protocol'],
        'training_profile_file': os.path.abspath(training_profile_path),
        'training_profile_hash': compute_file_hash(training_profile_path),
        'reference_run': 'K562_E3_enhanced_20261030_213631',
        'evaluation_protocol': '85% development, five independent stratified folds; Test inference only after CV and threshold locking',
        'ensemble_rule': 'equal-weight arithmetic mean of five best-checkpoint positive probabilities, each using its own train-fitted scaler',
        'fold_seed_policy': 'train_seed_base + zero_based_fold_index * 100',
        'threshold_transfer': 'pooled OOF single-model scores estimate a threshold for the ensemble; score calibration may differ',
        'status': 'cross_validation',
        'bio_hard_rule': 'legacy first 20 sequence columns, mismatches <= 3; quota 16 per batch',
        'steps_policy': 'fixed within run; reference default 250 steps per epoch',
        'feature_policy': 'single unified enhanced feature set; legacy names resolve to this same set for caller compatibility',
        'pam_policy': 'paper section 4.1: NGG PAM; encode only the terminal variable N base (4 features), omit invariant GG',
        'mismatch_policy': 'adds positions 1-3 and 17-19, transition/transversion counts, and maximum consecutive mismatch run',
        'indel_policy': 'uses explicit _/- positions, counts, and distance to the terminal PAM',
        'sequence_coordinate_policy': 'right-align 23/24 nt sequences to the terminal PAM; protospacer features exclude final 3 PAM bases',
        'tensorflow_version': tf_version,
        'keras_version': keras_version,
        'code_hash': code_hash,
        'fold_seeds': {},
        'init_hashes': {},
        'best_selections': {}
    }

    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Results storage
    results_biofusion = []
    oof_frames = []
    fold_artifacts = []

    start_time = datetime.now()

    # =========================================================================
    # 5-FOLD TRAINING
    # =========================================================================
    for fold in range(num_folds):
        fold_seed = args.train_seed + fold * 100
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
        steps_per_epoch = args.steps_per_epoch

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

        # Transform Train and Validation with this fold's Train-fitted scaler.
        Train_Pos_bio = scaler.transform(Train_Pos_bio)
        if len(Neg_Bio_bio) > 0:
            Neg_Bio_bio = scaler.transform(Neg_Bio_bio)
        if len(Neg_Normal_bio) > 0:
            Neg_Normal_bio = scaler.transform(Neg_Normal_bio)
        Xval_bio = scaler.transform(Xval_bio_raw)

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
            learning_rate=args.learning_rate
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
            validation_batch_size=validation_batch_size,
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
        val_metrics_biofusion = compute_metrics(y_val_gt, val_probs_biofusion, thresh_biofusion)
        print(f"BioFusion (Validation): {val_metrics_biofusion} (threshold={thresh_biofusion:.6f})")

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

        fold_oof = pd.DataFrame({
            'source_id': val_source_ids,
            'fold': fold + 1,
            'y_true': y_val_gt,
            'y_prob': val_probs_biofusion,
        })
        fold_oof.to_csv(os.path.join(predictions_dir, f"fold{fold+1}_biofusion_val.csv"), index=False)
        oof_frames.append(fold_oof)
        fold_artifacts.append({
            'weights_path': weight_path_biofusion,
            'scaler_path': os.path.join(features_dir, f"fold{fold+1}_feature_stats.json"),
        })
        config['best_selections'][f'fold_{fold+1}'] = {
            'epoch': best_epoch_biofusion,
            'val_loss': float(history_biofusion.history['val_loss'][best_epoch_biofusion - 1]),
            'validation_threshold': float(thresh_biofusion),
        }
        # Release callback/generator references before constructing the next fold.
        del model_biofusion, gen_biofusion, callbacks_biofusion, history_biofusion
        del Train_Pos_enc, Neg_Bio_enc, Neg_Normal_enc, Xval_enc
        del Train_Pos_bio, Neg_Bio_bio, Neg_Normal_bio, Xval_bio, Xval_bio_raw, Train_all_bio
        del Train_Pos_position, Neg_Bio_position, Neg_Normal_position, Xval_position
        del val_output_biofusion, val_probs_biofusion, scaler
        K.clear_session()
        gc.collect()

        # Update config after fold
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    # Lock the threshold using exactly one held-out prediction per pool row.
    oof = pd.concat(oof_frames, ignore_index=True).sort_values('source_id').reset_index(drop=True)
    expected_ids = np.asarray([int(row[3]) for row in train_pool])
    if oof['source_id'].duplicated().any() or not np.array_equal(
            oof['source_id'].to_numpy(), np.sort(expected_ids)):
        raise RuntimeError("OOF predictions must cover every development row exactly once")
    if np.intersect1d(expected_ids, test_source_ids).size:
        raise RuntimeError("Development and Test source rows overlap")
    ensemble_threshold, oof_f1 = find_optimal_f1_threshold(
        oof['y_true'].to_numpy(), oof['y_prob'].to_numpy(),
    )
    config['ensemble_threshold'] = float(ensemble_threshold)
    config['oof_threshold_f1'] = float(oof_f1)
    config['oof_rows'] = len(oof)
    config['status'] = 'threshold_locked_before_test'
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    oof.to_csv(os.path.join(predictions_dir, "biofusion_oof_val.csv"), index=False)

    # First Test feature extraction/inference: all CV decisions are now locked.
    print(f"\nCV complete. Locked OOF F1 threshold: {ensemble_threshold:.6f}")
    print("Evaluating the five-checkpoint probability ensemble on Test...")
    Xtest_df = pd.DataFrame(Xtest_raw[:, :3])
    Xtest_enc = encoding(Xtest_df).reshape((-1, 1, 24, 7)).astype("float32")
    Xtest_bio = extract_features_batch(Xtest_raw.tolist(), feature_set=feature_set)
    Xtest_position = extract_position_features_batch(Xtest_raw.tolist())
    test_probs = predict_ensemble(
        fold_artifacts, Xtest_enc, Xtest_bio, Xtest_position, args.learning_rate,
    )
    y_test_gt = np.array([1 if float(x) > 0.0 else 0 for x in Xtest_df.iloc[:, 2]])
    test_metrics = compute_metrics(y_test_gt, test_probs, ensemble_threshold)
    test_pred = (test_probs >= ensemble_threshold).astype(int)
    pd.DataFrame({
        'source_id': test_source_ids,
        'y_true': y_test_gt,
        'y_prob': test_probs,
        'y_pred': test_pred,
    }).to_csv(os.path.join(predictions_dir, "ensemble_biofusion_test.csv"), index=False)

    df_biofusion = pd.DataFrame(results_biofusion)
    metrics_cols = ['Best_Epoch', 'Threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC']
    summary_rows = []
    for stat in ('Mean', 'Std'):
        row = {'Fold': stat, 'split': 'val'}
        for metric in metrics_cols:
            row[metric] = (df_biofusion[metric].mean() if stat == 'Mean'
                           else df_biofusion[metric].std(ddof=0))
        summary_rows.append(row)
    summary_rows.append({
        'Fold': 'Ensemble_5', 'split': 'test', 'Best_Epoch': np.nan,
        'Threshold': ensemble_threshold, **test_metrics,
    })
    df_biofusion.to_csv(os.path.join(run_dir, "E3_biofusion_per_fold.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(run_dir, "E3_biofusion_summary.csv"), index=False)
    config['status'] = 'complete'
    config['test_metrics'] = test_metrics
    config['ensemble_checkpoints'] = [os.path.relpath(item['weights_path'], run_dir) for item in fold_artifacts]
    config['ensemble_scalers'] = [os.path.relpath(item['scaler_path'], run_dir) for item in fold_artifacts]
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    print("\nValidation CV mean/std and ONE final Test ensemble result:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved: {run_dir}")
    print(f"Total time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
