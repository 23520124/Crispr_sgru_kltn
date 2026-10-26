"""
E2 — Normal-negative sampling without replacement

So sánh 2 chế độ generator trên cùng kiến trúc P12:
A. replacement_control: sampling có hoàn lại (P12 baseline)
B. normal_stream: normal negative không hoàn lại, positive/bio-hard giữ nguyên

Mục tiêu:
- Kiểm tra coverage tăng có cải thiện PRAUC/F1 trên II5/II6
- So sánh công bằng: cùng khởi tạo, seed, split cho 2 chế độ
- Đo coverage chi tiết theo epoch
- Checkpoint theo val_loss (E1 chưa chứng minh val_PRAUC tốt hơn)

Không thay đổi:
- MODEL12, BCE-Dice, encoding, split
- Batch 256, steps 120, epochs 35
- Adam, ReduceLROnPlateau
- F1-threshold cumulative (E0), PRAUC từ continuous probs
- Kết quả chính: Mean ± Std của 5 fold (std ddof=0)
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import hashlib
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, deque
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, precision_score, recall_score, f1_score
)

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

from MODEL12 import Crispr_SGRU, BCEDiceLoss


def load_data(file_path):
    """Load data với source row IDs."""
    Negative, Positive = [], []
    with open(file_path, 'r') as f:
        for row_id, line in enumerate(f):
            ll = line.strip().split(',')
            try:
                label_item = np.float64(ll[2])
            except (ValueError, IndexError):
                continue
            # Thêm source_id vào tuple
            row_with_id = (ll[0], ll[1], ll[2], row_id)
            if label_item == 0.0:
                Negative.append(row_with_id)
            else:
                Positive.append(row_with_id)
    return Negative, Positive


def encoding(data):
    encode = []
    for _, row in data.iterrows():
        en = Encoder_sgRNA_off.Encoder(on_seq=row[0], off_seq=row[1], with_category=True, label=row[2])
        encode.append(en.on_off_code)
    return np.array(encode)


def count_mismatches(seq1, seq2):
    limit = min(20, len(seq1), len(seq2))
    return sum(1 for a, b in zip(seq1[:limit], seq2[:limit]) if a != b)


def compute_file_hash(file_path):
    """Hash toàn bộ file dataset."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_split_hash(data_array):
    """Hash split IDs để verify reproducibility."""
    source_ids = [row[3] for row in data_array]
    return hashlib.md5(str(sorted(source_ids)).encode()).hexdigest()[:16]


def get_model_weights_hash(model):
    """Get hash of model weights for init verification."""
    weights = model.get_weights()
    combined = np.concatenate([w.flatten() for w in weights])
    return hashlib.md5(combined.tobytes()).hexdigest()


def find_optimal_f1_threshold(y_true, y_probs):
    """
    F1-optimal threshold bằng cumulative confusion matrix (E0).
    Quét: all-positive → unique probs → all-negative.
    Chọn threshold nhỏ nhất khi tie.
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


class BatchConsumedCallback(Callback):
    """
    Update coverage tracking AFTER each batch is trained.
    Sử dụng hàng đợi (deque) để xử lý trường hợp Keras prefetch nhiều batch.
    """
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.batches_trained = 0

    def on_batch_end(self, batch, logs=None):
        # Consume one batch from generator's queue
        if hasattr(self.generator, 'batch_queue') and len(self.generator.batch_queue) > 0:
            norm_idx = self.generator.batch_queue.popleft()

            # Update coverage tracking AFTER training
            for idx in norm_idx:
                self.generator.normal_usage_count[idx] += 1
                self.generator.epoch_unique_seen.add(int(idx))
                self.generator.cumulative_unique_seen.add(int(idx))
            self.generator.total_normal_draws += len(norm_idx)

        self.batches_trained += 1

    def on_epoch_begin(self, epoch, logs=None):
        # Reset epoch_unique_seen at actual epoch boundary
        self.generator.epoch_unique_seen.clear()

    def on_epoch_end(self, epoch, logs=None):
        expected = self.generator.steps_per_epoch * (epoch + 1)
        if self.batches_trained != expected:
            print(f"[WARNING] Batches trained ({self.batches_trained}) != expected ({expected})")

        # Verify queue is empty at epoch end
        if hasattr(self.generator, 'batch_queue') and len(self.generator.batch_queue) > 0:
            print(f"[WARNING] Batch queue not empty at epoch end: {len(self.generator.batch_queue)} batches remaining")


class ReplacementControlGenerator:
    """
    Generator P12 baseline: tất cả nhóm lấy có hoàn lại.
    25% pos, 25% bio-hard, 50% normal.

    Coverage tracking: sử dụng hàng đợi (deque) để đảm bảo chính xác khi Keras prefetch.
    """
    def __init__(self, pos_enc, bio_hard_enc, normal_enc, batch_size=256, steps_per_epoch=120,
                 seed_pos=42, seed_hard=42, seed_normal=42, seed_shuffle=42):
        self.pos_enc = pos_enc
        self.bio_hard_enc = bio_hard_enc if len(bio_hard_enc) > 0 else normal_enc
        self.normal_enc = normal_enc
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.n_pos = max(2, int(batch_size * 0.25))
        self.n_hard = max(2, int(batch_size * 0.25))
        self.n_norm = batch_size - (self.n_pos + self.n_hard)

        # Separate RNG for each component
        self.rng_pos = np.random.RandomState(seed_pos)
        self.rng_hard = np.random.RandomState(seed_hard)
        self.rng_normal = np.random.RandomState(seed_normal)
        self.rng_shuffle = np.random.RandomState(seed_shuffle)

        # Coverage tracking per sample ID
        self.normal_usage_count = np.zeros(len(self.normal_enc), dtype=np.int32)
        self.total_normal_draws = 0
        self.epoch_unique_seen = set()
        self.cumulative_unique_seen = set()

        # Batch queue for callback consumption (FIFO)
        self.batch_queue = deque()

    def flow(self):
        while True:
            for _ in range(self.steps_per_epoch):
                pos_idx = self.rng_pos.choice(len(self.pos_enc), self.n_pos, replace=True)
                b_pos = self.pos_enc[pos_idx]

                hard_idx = self.rng_hard.choice(len(self.bio_hard_enc), self.n_hard, replace=True)
                b_hard = self.bio_hard_enc[hard_idx]

                norm_idx = self.rng_normal.choice(len(self.normal_enc), self.n_norm, replace=True)
                b_norm = self.normal_enc[norm_idx]

                # Enqueue indices for callback to consume AFTER training
                self.batch_queue.append(norm_idx)

                X_batch = np.concatenate([b_pos, b_hard, b_norm], axis=0)
                y_batch = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_norm), dtype=np.float32)

                p = self.rng_shuffle.permutation(len(y_batch))
                yield X_batch[p], y_batch[p].reshape(-1, 1)

    def get_coverage_stats(self):
        unique_seen = len(self.cumulative_unique_seen)
        coverage = unique_seen / len(self.normal_enc) if len(self.normal_enc) > 0 else 0.0
        repetitions = self.total_normal_draws - unique_seen
        return {
            'total_normal': len(self.normal_enc),
            'total_draws': self.total_normal_draws,
            'unique_seen': unique_seen,
            'coverage': coverage,
            'repetitions': int(repetitions),
            'epoch_unique': len(self.epoch_unique_seen)
        }

    def reset_coverage(self):
        self.normal_usage_count.fill(0)
        self.total_normal_draws = 0
        self.epoch_unique_seen.clear()
        self.cumulative_unique_seen.clear()
        self.batch_queue.clear()


class NormalStreamGenerator:
    """
    Generator với normal negative không hoàn lại.
    Positive và bio-hard vẫn có hoàn lại (cùng seed với control).
    Normal: duyệt tuần tự qua permutation, shuffle khi hết.

    Coverage tracking: sử dụng hàng đợi (deque) để đảm bảo chính xác khi Keras prefetch.
    """
    def __init__(self, pos_enc, bio_hard_enc, normal_enc, batch_size=256, steps_per_epoch=120,
                 seed_pos=42, seed_hard=42, seed_normal=42, seed_shuffle=42):
        self.pos_enc = pos_enc
        self.bio_hard_enc = bio_hard_enc if len(bio_hard_enc) > 0 else normal_enc
        self.normal_enc = normal_enc
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.n_pos = max(2, int(batch_size * 0.25))
        self.n_hard = max(2, int(batch_size * 0.25))
        self.n_norm = batch_size - (self.n_pos + self.n_hard)

        # Separate RNG (same seeds as control for pos/hard/shuffle)
        self.rng_pos = np.random.RandomState(seed_pos)
        self.rng_hard = np.random.RandomState(seed_hard)
        self.rng_normal = np.random.RandomState(seed_normal)
        self.rng_shuffle = np.random.RandomState(seed_shuffle)

        # Normal stream state
        self.normal_indices = np.arange(len(self.normal_enc))
        self.rng_normal.shuffle(self.normal_indices)
        self.normal_cursor = 0
        self.normal_permutation_count = 0

        # Coverage tracking
        self.normal_usage_count = np.zeros(len(self.normal_enc), dtype=np.int32)
        self.total_normal_draws = 0
        self.epoch_unique_seen = set()
        self.cumulative_unique_seen = set()

        # Batch queue for callback consumption (FIFO)
        self.batch_queue = deque()

    def _get_normal_batch(self):
        """
        Get next n_norm samples from stream, wrap around if needed.
        Fix: copy indices trước khi shuffle, dùng while để lấy đủ qua nhiều permutation.
        """
        if len(self.normal_enc) == 0:
            raise ValueError("Normal negative pool is empty!")

        batch_idx = []
        needed = self.n_norm

        while needed > 0:
            available = len(self.normal_indices) - self.normal_cursor

            if available >= needed:
                # Đủ samples trong permutation hiện tại
                chunk = self.normal_indices[self.normal_cursor:self.normal_cursor + needed].copy()
                batch_idx.append(chunk)
                self.normal_cursor += needed
                needed = 0
            else:
                # Lấy hết phần còn lại, sau đó shuffle
                if available > 0:
                    chunk = self.normal_indices[self.normal_cursor:].copy()
                    batch_idx.append(chunk)
                    needed -= available

                # Shuffle cho permutation mới
                self.rng_normal.shuffle(self.normal_indices)
                self.normal_permutation_count += 1
                self.normal_cursor = 0

        return np.concatenate(batch_idx)

    def flow(self):
        while True:
            for _ in range(self.steps_per_epoch):
                pos_idx = self.rng_pos.choice(len(self.pos_enc), self.n_pos, replace=True)
                b_pos = self.pos_enc[pos_idx]

                hard_idx = self.rng_hard.choice(len(self.bio_hard_enc), self.n_hard, replace=True)
                b_hard = self.bio_hard_enc[hard_idx]

                norm_idx = self._get_normal_batch()
                b_norm = self.normal_enc[norm_idx]

                # Enqueue indices for callback to consume AFTER training
                self.batch_queue.append(norm_idx)

                X_batch = np.concatenate([b_pos, b_hard, b_norm], axis=0)
                y_batch = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_norm), dtype=np.float32)

                p = self.rng_shuffle.permutation(len(y_batch))
                yield X_batch[p], y_batch[p].reshape(-1, 1)

    def get_coverage_stats(self):
        unique_seen = len(self.cumulative_unique_seen)
        coverage = unique_seen / len(self.normal_enc) if len(self.normal_enc) > 0 else 0.0
        repetitions = self.total_normal_draws - unique_seen
        return {
            'total_normal': len(self.normal_enc),
            'total_draws': self.total_normal_draws,
            'unique_seen': unique_seen,
            'coverage': coverage,
            'repetitions': int(repetitions),
            'permutation_count': self.normal_permutation_count,
            'epoch_unique': len(self.epoch_unique_seen)
        }

    def reset_coverage(self):
        self.normal_usage_count.fill(0)
        self.total_normal_draws = 0
        self.epoch_unique_seen.clear()
        self.cumulative_unique_seen.clear()
        self.normal_permutation_count = 0
        self.batch_queue.clear()


class CoverageCallback(Callback):
    """Track coverage after each epoch (runs after coverage is updated by BatchConsumedCallback)."""
    def __init__(self, generator, log_list):
        super().__init__()
        self.generator = generator
        self.log_list = log_list

    def on_epoch_end(self, epoch, logs=None):
        stats = self.generator.get_coverage_stats()
        stats['epoch'] = epoch + 1
        stats['train_loss'] = logs.get('loss', 0.0)
        stats['val_loss'] = logs.get('val_loss', 0.0)
        stats['lr'] = float(K.get_value(self.model.optimizer.learning_rate))
        self.log_list.append(stats)


class ValidationPRAUCCallback(Callback):
    """Compute validation PRAUC after each epoch (for monitoring only)."""
    def __init__(self, val_x, val_y, log_list):
        super().__init__()
        self.val_x = val_x
        self.val_y = val_y.ravel()
        self.log_list = log_list

    def on_epoch_end(self, epoch, logs=None):
        val_probs = self.model.predict(self.val_x, verbose=0).ravel()
        precision, recall, _ = precision_recall_curve(self.val_y, val_probs)
        val_prauc = auc(recall, precision)
        self.log_list.append({'epoch': epoch + 1, 'val_prauc': val_prauc})
        print(f"  [Monitor] Epoch {epoch+1}: val_PRAUC={val_prauc:.4f}")


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    data_name = "BE3"  # Thay: "II5", "II6", etc.

    dataset_path_map = {
        "BE3": os.path.join(BASE_DIR, "Datasets", "BE3.csv"),
        "I1":  os.path.join(BASE_DIR, "Datasets", "I1.csv"),
        "II6": os.path.join(BASE_DIR, "Datasets", "II6.csv"),
        "II5": os.path.join(BASE_DIR, "Datasets", "II5.csv"),
        "K562": os.path.join(BASE_DIR, "Datasets", "K562.csv"),
        "hek": os.path.join(BASE_DIR, "Datasets", "HEK293t.csv"),
    }

    file_path = dataset_path_map.get(data_name)
    if file_path is None:
        print(f"[ERROR] Unknown data_name: '{data_name}'")
        sys.exit(1)

    # Create run directory with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, "weights", f"{data_name}_E2_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    save_dir_control = os.path.join(run_dir, "replacement_control")
    save_dir_stream = os.path.join(run_dir, "normal_stream")
    history_dir = os.path.join(run_dir, "history")
    coverage_dir = os.path.join(run_dir, "coverage")
    predictions_dir = os.path.join(run_dir, "predictions")

    for d in [save_dir_control, save_dir_stream, history_dir, coverage_dir, predictions_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"E2 — NORMAL-NEGATIVE COVERAGE EXPERIMENT")
    print(f"Dataset: {file_path}")
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")

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

    Xtest_raw = np.vstack((Test_Neg, Test_Pos))
    np.random.seed(42)
    np.random.shuffle(Xtest_raw)
    test_source_ids = [int(row[3]) for row in Xtest_raw]
    Xtest_df = pd.DataFrame(Xtest_raw[:, :3])  # Drop source_id for encoding
    Xtest_enc = encoding(Xtest_df).reshape((-1, 1, 24, 7)).astype("float32")
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

    # Config
    code_hash = hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:16]

    config = {
        'dataset': data_name,
        'dataset_file_hash': dataset_file_hash,
        'train_pool_neg_hash': train_pool_neg_hash,
        'test_neg_hash': test_neg_hash,
        'train_pool_pos_hash': train_pool_pos_hash,
        'test_pos_hash': test_pos_hash,
        'run_id': run_id,
        'total_pos': total_pos,
        'total_neg': total_neg,
        'test_size': 0.15,
        'num_folds': num_folds,
        'batch_size': BATCH_SIZE,
        'steps_per_epoch': STEPS_PER_EPOCH,
        'epochs': EPOCHS,
        'model': 'MODEL12',
        'loss': 'BCEDiceLoss',
        'optimizer': 'Adam_1e-4',
        'checkpoint_rule': 'val_loss_min',
        'threshold_rule': 'F1_optimal_cumulative',
        'bio_hard_rule': 'mismatch_le_3',
        'tensorflow_version': tf.__version__,
        'keras_version': tf.keras.__version__,
        'code_hash': code_hash,
        'fold_seeds': {},
        'init_hashes': {}
    }

    # Save initial config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Results storage
    results_control = []
    results_stream = []

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

        # Encoding (drop source_id)
        Train_Pos_enc = encoding(pd.DataFrame([row[:3] for row in Train_Pos])).reshape((-1, 1, 24, 7)).astype("float32")
        Neg_Bio_enc = encoding(pd.DataFrame([row[:3] for row in Neg_Bio])).reshape((-1, 1, 24, 7)).astype("float32") if len(Neg_Bio) > 0 else np.empty((0, 1, 24, 7), dtype="float32")
        Neg_Normal_enc = encoding(pd.DataFrame([row[:3] for row in Neg_Normal])).reshape((-1, 1, 24, 7)).astype("float32")

        Xval_enc = encoding(pd.DataFrame([row[:3] for row in Val_raw])).reshape((-1, 1, 24, 7)).astype("float32")
        y_val_gt = Val_y
        y_val_target = y_val_gt.reshape(-1, 1).astype(np.float32)

        # =====================================================================
        # MODE A: REPLACEMENT CONTROL
        # =====================================================================
        print(f"\n--- MODE A: replacement_control ---")

        # Set seeds
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        model_control = Crispr_SGRU(learning_rate=0.0001)
        init_hash_control = get_model_weights_hash(model_control)
        print(f"Init hash: {init_hash_control}")

        # Save init weights for stream
        init_weights_control = [w.copy() for w in model_control.get_weights()]

        gen_control = ReplacementControlGenerator(
            pos_enc=Train_Pos_enc,
            bio_hard_enc=Neg_Bio_enc if len(Neg_Bio_enc) > 0 else Neg_Normal_enc,
            normal_enc=Neg_Normal_enc,
            batch_size=BATCH_SIZE,
            steps_per_epoch=STEPS_PER_EPOCH,
            seed_pos=fold_seed,
            seed_hard=fold_seed + 1,
            seed_normal=fold_seed + 2,
            seed_shuffle=fold_seed + 3
        )

        weight_path_control = os.path.join(save_dir_control, f"{data_name}_fold{fold + 1}.weights.h5")
        coverage_log_control = []
        prauc_log_control = []

        callbacks_control = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            ModelCheckpoint(weight_path_control, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0),
            BatchConsumedCallback(gen_control),
            CoverageCallback(gen_control, coverage_log_control),
            ValidationPRAUCCallback(Xval_enc, y_val_target, prauc_log_control)
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

        # Find best epoch
        best_epoch_control = int(np.argmin(history_control.history['val_loss'])) + 1
        print(f"Best epoch (val_loss): {best_epoch_control}")

        # Save coverage history
        pd.DataFrame(coverage_log_control).to_csv(
            os.path.join(coverage_dir, f"fold{fold+1}_control_coverage.csv"), index=False)
        pd.DataFrame(prauc_log_control).to_csv(
            os.path.join(history_dir, f"fold{fold+1}_control_prauc.csv"), index=False)

        # Save normal usage counts
        normal_usage_control = pd.DataFrame({
            'normal_idx': np.arange(len(gen_control.normal_usage_count)),
            'usage_count': gen_control.normal_usage_count
        })
        normal_usage_control.to_csv(
            os.path.join(coverage_dir, f"fold{fold+1}_control_normal_usage.csv"), index=False)

        # Evaluate
        model_control.load_weights(weight_path_control)

        val_probs_control = model_control.predict(Xval_enc, verbose=0).ravel()
        thresh_control, _ = find_optimal_f1_threshold(y_val_gt, val_probs_control)

        test_probs_control = model_control.predict(Xtest_enc, verbose=0).ravel()
        test_pred_control = (test_probs_control >= thresh_control).astype(int)

        fpr, tpr, _ = roc_curve(y_test_gt, test_probs_control)
        auroc_control = auc(fpr, tpr)
        p_curve, r_curve, _ = precision_recall_curve(y_test_gt, test_probs_control)
        prauc_control = auc(r_curve, p_curve)
        f1_control = f1_score(y_test_gt, test_pred_control, zero_division=0)
        prec_control = precision_score(y_test_gt, test_pred_control, zero_division=0)
        rec_control = recall_score(y_test_gt, test_pred_control, zero_division=0)
        mcc_control = matthews_corrcoef(y_test_gt, test_pred_control)

        print(f"Control: AUROC={auroc_control:.4f}, PRAUC={prauc_control:.4f}, F1={f1_control:.4f}, Prec={prec_control:.4f}, Rec={rec_control:.4f}, MCC={mcc_control:.4f} (thresh={thresh_control:.4f})")

        results_control.append({
            'Fold': f'Fold_{fold+1}',
            'Best_Epoch': best_epoch_control,
            'Threshold': thresh_control,
            'AUROC': auroc_control,
            'PRAUC': prauc_control,
            'F1': f1_control,
            'Precision': prec_control,
            'Recall': rec_control,
            'MCC': mcc_control
        })

        # Save predictions with source IDs
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
        # MODE B: NORMAL STREAM
        # =====================================================================
        print(f"\n--- MODE B: normal_stream ---")

        # Reset seeds (same as control)
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        model_stream = Crispr_SGRU(learning_rate=0.0001)

        # Set SAME init weights as control
        model_stream.set_weights(init_weights_control)
        init_hash_stream = get_model_weights_hash(model_stream)
        print(f"Init hash: {init_hash_stream}")

        # Assert init hashes match
        if init_hash_control != init_hash_stream:
            print(f"[ERROR] Init hash mismatch! control={init_hash_control}, stream={init_hash_stream}")
            sys.exit(1)

        # Save init hash to config
        config['init_hashes'][f'fold_{fold+1}'] = init_hash_control

        gen_stream = NormalStreamGenerator(
            pos_enc=Train_Pos_enc,
            bio_hard_enc=Neg_Bio_enc if len(Neg_Bio_enc) > 0 else Neg_Normal_enc,
            normal_enc=Neg_Normal_enc,
            batch_size=BATCH_SIZE,
            steps_per_epoch=STEPS_PER_EPOCH,
            seed_pos=fold_seed,
            seed_hard=fold_seed + 1,
            seed_normal=fold_seed + 2,
            seed_shuffle=fold_seed + 3
        )

        weight_path_stream = os.path.join(save_dir_stream, f"{data_name}_fold{fold + 1}.weights.h5")
        coverage_log_stream = []
        prauc_log_stream = []

        callbacks_stream = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            ModelCheckpoint(weight_path_stream, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0),
            BatchConsumedCallback(gen_stream),
            CoverageCallback(gen_stream, coverage_log_stream),
            ValidationPRAUCCallback(Xval_enc, y_val_target, prauc_log_stream)
        ]

        print("Training...")
        history_stream = model_stream.fit(
            gen_stream.flow(),
            validation_data=(Xval_enc, y_val_target),
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks_stream,
            verbose=2
        )

        if not os.path.exists(weight_path_stream):
            print(f"[ERROR] Checkpoint not found: {weight_path_stream}")
            sys.exit(1)

        # Find best epoch
        best_epoch_stream = int(np.argmin(history_stream.history['val_loss'])) + 1
        print(f"Best epoch (val_loss): {best_epoch_stream}")

        # Save coverage history
        pd.DataFrame(coverage_log_stream).to_csv(
            os.path.join(coverage_dir, f"fold{fold+1}_stream_coverage.csv"), index=False)
        pd.DataFrame(prauc_log_stream).to_csv(
            os.path.join(history_dir, f"fold{fold+1}_stream_prauc.csv"), index=False)

        # Save normal usage counts
        normal_usage_stream = pd.DataFrame({
            'normal_idx': np.arange(len(gen_stream.normal_usage_count)),
            'usage_count': gen_stream.normal_usage_count
        })
        normal_usage_stream.to_csv(
            os.path.join(coverage_dir, f"fold{fold+1}_stream_normal_usage.csv"), index=False)

        # Evaluate
        model_stream.load_weights(weight_path_stream)

        val_probs_stream = model_stream.predict(Xval_enc, verbose=0).ravel()
        thresh_stream, _ = find_optimal_f1_threshold(y_val_gt, val_probs_stream)

        test_probs_stream = model_stream.predict(Xtest_enc, verbose=0).ravel()
        test_pred_stream = (test_probs_stream >= thresh_stream).astype(int)

        fpr, tpr, _ = roc_curve(y_test_gt, test_probs_stream)
        auroc_stream = auc(fpr, tpr)
        p_curve, r_curve, _ = precision_recall_curve(y_test_gt, test_probs_stream)
        prauc_stream = auc(r_curve, p_curve)
        f1_stream = f1_score(y_test_gt, test_pred_stream, zero_division=0)
        prec_stream = precision_score(y_test_gt, test_pred_stream, zero_division=0)
        rec_stream = recall_score(y_test_gt, test_pred_stream, zero_division=0)
        mcc_stream = matthews_corrcoef(y_test_gt, test_pred_stream)

        print(f"Stream: AUROC={auroc_stream:.4f}, PRAUC={prauc_stream:.4f}, F1={f1_stream:.4f}, Prec={prec_stream:.4f}, Rec={rec_stream:.4f}, MCC={mcc_stream:.4f} (thresh={thresh_stream:.4f})")

        results_stream.append({
            'Fold': f'Fold_{fold+1}',
            'Best_Epoch': best_epoch_stream,
            'Threshold': thresh_stream,
            'AUROC': auroc_stream,
            'PRAUC': prauc_stream,
            'F1': f1_stream,
            'Precision': prec_stream,
            'Recall': rec_stream,
            'MCC': mcc_stream
        })

        # Save predictions with source IDs
        pd.DataFrame({
            'source_id': val_source_ids,
            'y_true': y_val_gt,
            'y_prob': val_probs_stream
        }).to_csv(os.path.join(predictions_dir, f"fold{fold+1}_stream_val.csv"), index=False)

        pd.DataFrame({
            'source_id': test_source_ids,
            'y_true': y_test_gt,
            'y_prob': test_probs_stream,
            'y_pred': test_pred_stream
        }).to_csv(os.path.join(predictions_dir, f"fold{fold+1}_stream_test.csv"), index=False)

        del model_stream
        K.clear_session()
        gc.collect()

        # Update config after fold
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

    # =========================================================================
    # COMPUTE MEAN AND STD
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPUTING MEAN ± STD")
    print(f"{'='*70}\n")

    df_control = pd.DataFrame(results_control)
    df_stream = pd.DataFrame(results_stream)

    metrics_cols = ['Best_Epoch', 'Threshold', 'AUROC', 'PRAUC', 'F1', 'Precision', 'Recall', 'MCC']

    mean_control = {'Fold': 'Mean'}
    std_control = {'Fold': 'Std'}
    for col in metrics_cols:
        mean_control[col] = df_control[col].mean()
        std_control[col] = df_control[col].std(ddof=0)

    mean_stream = {'Fold': 'Mean'}
    std_stream = {'Fold': 'Std'}
    for col in metrics_cols:
        mean_stream[col] = df_stream[col].mean()
        std_stream[col] = df_stream[col].std(ddof=0)

    df_control = pd.concat([df_control, pd.DataFrame([mean_control, std_control])], ignore_index=True)
    df_stream = pd.concat([df_stream, pd.DataFrame([mean_stream, std_stream])], ignore_index=True)

    # Save per-fold results
    control_csv = os.path.join(run_dir, f"{data_name}_E2_replacement_control_per_fold.csv")
    stream_csv = os.path.join(run_dir, f"{data_name}_E2_normal_stream_per_fold.csv")

    df_control.to_csv(control_csv, index=False)
    df_stream.to_csv(stream_csv, index=False)

    print("Replacement control:")
    print(df_control.to_string(index=False))
    print(f"\nSaved: {control_csv}")

    print("\nNormal stream:")
    print(df_stream.to_string(index=False))
    print(f"\nSaved: {stream_csv}")

    # Paired differences
    paired_diffs = []
    for i in range(5):
        diff_row = {'Fold': f'Fold_{i+1}'}
        for col in metrics_cols:
            diff_row[col] = df_stream.iloc[i][col] - df_control.iloc[i][col]
        paired_diffs.append(diff_row)

    mean_diff = {'Fold': 'Mean_Diff'}
    for col in metrics_cols:
        diffs = [paired_diffs[i][col] for i in range(5)]
        mean_diff[col] = np.mean(diffs)

    paired_diffs.append(mean_diff)
    df_paired = pd.DataFrame(paired_diffs)

    paired_csv = os.path.join(run_dir, f"{data_name}_E2_paired_differences.csv")
    df_paired.to_csv(paired_csv, index=False)

    print("\nPaired differences (Stream - Control):")
    print(df_paired.to_string(index=False))
    print(f"\nSaved: {paired_csv}")

    # Combined summary
    summary_rows = []

    for i in range(5):
        row = results_control[i].copy()
        row['Mode'] = 'replacement_control'
        summary_rows.append(row)

    summary_rows.append({
        'Fold': 'Mean',
        'Mode': 'replacement_control',
        **{k: mean_control[k] for k in metrics_cols}
    })
    summary_rows.append({
        'Fold': 'Std',
        'Mode': 'replacement_control',
        **{k: std_control[k] for k in metrics_cols}
    })

    for i in range(5):
        row = results_stream[i].copy()
        row['Mode'] = 'normal_stream'
        summary_rows.append(row)

    summary_rows.append({
        'Fold': 'Mean',
        'Mode': 'normal_stream',
        **{k: mean_stream[k] for k in metrics_cols}
    })
    summary_rows.append({
        'Fold': 'Std',
        'Mode': 'normal_stream',
        **{k: std_stream[k] for k in metrics_cols}
    })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(run_dir, f"{data_name}_E2_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    print(f"\nSaved summary: {summary_csv}")

    # Save config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print("E2 COMPLETE")
    print(f"{'='*70}")
    print(f"\nPRIMARY RESULT = Mean ± Std of 5 folds for each mode")
    print(f"COMPARISON = Paired differences (stream - control)")
    print(f"\nNote: Coverage logs saved per fold in {coverage_dir}")
    print(f"Note: PRAUC logs saved per fold in {history_dir}")
    print(f"Note: Test predictions saved in {predictions_dir}")
    print(f"\nTotal time: {datetime.now() - start_time}")
