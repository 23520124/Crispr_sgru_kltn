import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, 
    matthews_corrcoef, precision_score, recall_score, f1_score
)

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

print(f"-> Thư mục gốc dự án (BASE_DIR): {BASE_DIR}")
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
    Negative, Positive, label = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            ll = line.strip().split(',')
            try:
                label_item = np.float64(ll[2])
            except (ValueError, IndexError):
                continue
            if label_item == 0.0:
                Negative.append(ll)
            else:
                Positive.append(ll)
            label.append(label_item)
    return Negative, Positive, label


def encoding(data):
    encode = []
    for _, row in data.iterrows():
        en = Encoder_sgRNA_off.Encoder(on_seq=row[0], off_seq=row[1], with_category=True, label=row[2])
        encode.append(en.on_off_code)
    return np.array(encode)


def count_mismatches(seq1, seq2):
    limit = min(20, len(seq1), len(seq2))
    return sum(1 for a, b in zip(seq1[:limit], seq2[:limit]) if a != b)


class StableMiningGenerator:
    """
    Generator ổn định:
    - 25% Mẫu dương (Lấy mẫu ngẫu nhiên có hoàn lại từ pool dương)
    - 25% Mẫu âm khó (mismatch <= 3)
    - 50% Mẫu âm ngẫu nhiên thông thường (giữ vai trò neo phân phối)
    """
    def __init__(self, pos_enc, bio_hard_neg_enc, normal_neg_enc, batch_size=256, steps_per_epoch=150):
        self.pos_enc = pos_enc
        self.bio_hard_neg_enc = bio_hard_neg_enc if len(bio_hard_neg_enc) > 0 else normal_neg_enc
        self.normal_neg_enc = normal_neg_enc
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.n_pos = max(2, int(batch_size * 0.25))
        self.n_hard = max(2, int(batch_size * 0.25))
        self.n_norm = batch_size - (self.n_pos + self.n_hard)

    def flow(self):
        while True:
            for _ in range(self.steps_per_epoch):
                pos_idx = np.random.choice(len(self.pos_enc), self.n_pos, replace=True)
                b_pos = self.pos_enc[pos_idx]

                hard_idx = np.random.choice(len(self.bio_hard_neg_enc), self.n_hard, replace=True)
                b_hard = self.bio_hard_neg_enc[hard_idx]

                norm_idx = np.random.choice(len(self.normal_neg_enc), self.n_norm, replace=True)
                b_norm = self.normal_neg_enc[norm_idx]

                X_batch = np.concatenate([b_pos, b_hard, b_norm], axis=0)
                y_batch = np.array([1.0] * self.n_pos + [0.0] * (self.n_hard + self.n_norm), dtype=np.float32)

                p = np.random.permutation(len(y_batch))
                yield X_batch[p], y_batch[p].reshape(-1, 1)


# =========================================================================
# CẤU HÌNH THỰC NGHIỆM: BE3
# =========================================================================
data_name = "I2"  # Hoặc "I1", "II6"

dataset_path_map = {
    "BE3": os.path.join(BASE_DIR, "Datasets", "BE3.csv"),
    "I1":  os.path.join(BASE_DIR, "Datasets", "I1.csv"),
    "II6": os.path.join(BASE_DIR, "Datasets", "II6.csv"),
    "II5": os.path.join(BASE_DIR, "Datasets", "II5.csv"),
    "I2":  os.path.join(BASE_DIR, "Datasets", "I2.csv"),
    "k562": os.path.join(BASE_DIR, "Datasets", "K562.csv"),
    "hek": os.path.join(BASE_DIR, "Datasets", "HEK293t.csv"),
    "change-seq": os.path.join(BASE_DIR, "Datasets", "CHANGE-seq", "CHANGE-seq.csv")
}

file_path = dataset_path_map.get(data_name)
save_dir = os.path.join(BASE_DIR, "weights", f"{data_name}_stable_bcedice")
os.makedirs(save_dir, exist_ok=True)

print(f"\n{'='*70}\nCHẠY PHIÊN BẢN ỔN ĐỊNH (BCE-DICE + STABLE SAMPLING)\nTập dữ liệu: {file_path}\n{'='*70}")

Negative, Positive, label = load_data(file_path)
Positive = np.array(Positive)
Negative = np.array(Negative)
total_pos = len(Positive)
total_neg = len(Negative)
print(f"Dữ liệu: {total_pos} Positives | {total_neg} Negatives | Tỷ lệ IR = {total_neg / max(1, total_pos):.2f}")

# Cố định 15% Blind Test
Train_Pool_Neg, Test_Neg = train_test_split(Negative, test_size=0.15, random_state=42)
Train_Pool_Pos, Test_Pos = train_test_split(Positive, test_size=0.15, random_state=42)

Xtest_raw = np.vstack((Test_Neg, Test_Pos))
np.random.seed(42)
np.random.shuffle(Xtest_raw)
Xtest_df = pd.DataFrame(Xtest_raw)
Xtest_enc = encoding(Xtest_df).reshape((-1, 1, 24, 7)).astype("float32")
y_test_gt = np.array([1 if float(x) > 0.0 else 0 for x in Xtest_df.iloc[:, 2]])

BATCH_SIZE = 256
STEPS_PER_EPOCH = 120
EPOCHS = 35
num_folds = 5

AUROC, PRAUC, F1_SCORE, PRECISION, RECALL, MCC = [], [], [], [], [], []
start_time = datetime.now()

train_pool = np.vstack((Train_Pool_Neg, Train_Pool_Pos))
train_pool_y = np.array([0] * len(Train_Pool_Neg) + [1] * len(Train_Pool_Pos))
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_splits = list(skf.split(train_pool, train_pool_y))
oof_probs = np.full(len(train_pool), np.nan, dtype=np.float32)

# =========================================================================
# VÒNG LẶP HUẤN LUYỆN 5-FOLD
# =========================================================================
for fold in range(num_folds):
    print(f"\n==================== CHẠY FOLD #{fold + 1} ====================")
    train_idx, val_idx = fold_splits[fold]
    Train_raw, Val_raw = train_pool[train_idx], train_pool[val_idx]
    Train_y, Val_y = train_pool_y[train_idx], train_pool_y[val_idx]

    Train_Neg = Train_raw[Train_y == 0]
    Train_Pos = Train_raw[Train_y == 1]

    # Phân loại mẫu âm sinh học khó (mismatches <= 3)
    bio_mask = np.array([count_mismatches(r[0], r[1]) <= 3 for r in Train_Neg])
    Neg_Bio = Train_Neg[bio_mask]
    Neg_Normal = Train_Neg[~bio_mask]

    print(f"Fold #{fold+1}: {len(Train_Pos)} Pos | {len(Neg_Bio)} Bio-Hard Neg | {len(Neg_Normal)} Normal Neg")

    Train_Pos_enc = encoding(pd.DataFrame(Train_Pos)).reshape((-1, 1, 24, 7)).astype("float32")
    Neg_Bio_enc = encoding(pd.DataFrame(Neg_Bio)).reshape((-1, 1, 24, 7)).astype("float32") if len(Neg_Bio) > 0 else np.empty((0, 1, 24, 7), dtype="float32")
    Neg_Normal_enc = encoding(pd.DataFrame(Neg_Normal)).reshape((-1, 1, 24, 7)).astype("float32")

    Xval_enc = encoding(pd.DataFrame(Val_raw)).reshape((-1, 1, 24, 7)).astype("float32")
    y_val_gt = Val_y
    y_val_target = y_val_gt.reshape(-1, 1).astype(np.float32)

    gen = StableMiningGenerator(
        pos_enc=Train_Pos_enc,
        bio_hard_neg_enc=Neg_Bio_enc if len(Neg_Bio_enc) > 0 else Neg_Normal_enc,
        normal_neg_enc=Neg_Normal_enc,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH
    )

    model = Crispr_SGRU(learning_rate=0.0001)

    weight_path = os.path.join(save_dir, f"{data_name}_fold{fold + 1}.weights.h5")
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(weight_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
    ]

    model.fit(
        gen.flow(),
        validation_data=(Xval_enc, y_val_target),
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=callbacks,
        verbose=2
    )

    model.load_weights(weight_path)

    # Dự đoán xác suất kiểm định
    val_probs = model.predict(Xval_enc, verbose=0).ravel()
    oof_probs[val_idx] = val_probs

    # Quét ngưỡng tối ưu hóa MCC an toàn (tránh sụp đổ về threshold 0.01)
    best_thresh = 0.5
    best_mcc = 0.001
    for th in np.linspace(0.10, 0.90, 81):
        preds = (val_probs >= th).astype(int)
        score = matthews_corrcoef(y_val_gt, preds)
        if score > best_mcc:
            best_mcc = score
            best_thresh = float(th)

    # Đánh giá trên Blind Test
    test_probs = model.predict(Xtest_enc, verbose=0).ravel()
    test_pred_bin = (test_probs >= best_thresh).astype(int)

    fpr, tpr, _ = roc_curve(y_test_gt, test_probs)
    fold_auroc = auc(fpr, tpr)
    p_curve, r_curve, _ = precision_recall_curve(y_test_gt, test_probs)
    fold_prauc = auc(r_curve, p_curve)
    fold_f1 = f1_score(y_test_gt, test_pred_bin, zero_division=0)
    fold_prec = precision_score(y_test_gt, test_pred_bin, zero_division=0)
    fold_rec = recall_score(y_test_gt, test_pred_bin, zero_division=0)
    fold_mcc = matthews_corrcoef(y_test_gt, test_pred_bin)

    print(f"Fold #{fold + 1} Result -> AUROC={fold_auroc:.4f}, PRAUC={fold_prauc:.4f}, F1={fold_f1:.4f}, Prec={fold_prec:.4f}, Rec={fold_rec:.4f}, MCC={fold_mcc:.4f} (Thresh={best_thresh:.2f})")

    AUROC.append(fold_auroc)
    PRAUC.append(fold_prauc)
    F1_SCORE.append(fold_f1)
    PRECISION.append(fold_prec)
    RECALL.append(fold_rec)
    MCC.append(fold_mcc)

    del (Xval_enc, y_val_gt, y_val_target, model, test_probs)
    K.clear_session()
    gc.collect()

# =========================================================================
# ĐÁNH GIÁ ENSEMBLE 5 FOLDS
# =========================================================================
print(f"\n{'='*70}\nĐÁNH GIÁ SOFT-VOTING ENSEMBLE TRÊN BLIND TEST\n{'='*70}")

ensemble_probs = np.zeros(len(Xtest_enc), dtype=np.float64)
for f in range(num_folds):
    m_ens = Crispr_SGRU()
    w_f = os.path.join(save_dir, f"{data_name}_fold{f + 1}.weights.h5")
    m_ens.load_weights(w_f)
    ensemble_probs += m_ens.predict(Xtest_enc, verbose=0).ravel()
    del m_ens
    K.clear_session()
    gc.collect()

ensemble_probs /= float(num_folds)

# Tìm ngưỡng toàn cục tối ưu trên OOF
ens_thresh = 0.5
best_oof_mcc = 0.001
for th in np.linspace(0.10, 0.90, 81):
    score = matthews_corrcoef(train_pool_y, (oof_probs >= th).astype(int))
    if score > best_oof_mcc:
        best_oof_mcc = score
        ens_thresh = float(th)

ens_pred_bin = (ensemble_probs >= ens_thresh).astype(int)

fpr, tpr, _ = roc_curve(y_test_gt, ensemble_probs)
ens_auroc = auc(fpr, tpr)
p_curve, r_curve, _ = precision_recall_curve(y_test_gt, ensemble_probs)
ens_prauc = auc(r_curve, p_curve)
ens_f1 = f1_score(y_test_gt, ens_pred_bin, zero_division=0)
ens_prec = precision_score(y_test_gt, ens_pred_bin, zero_division=0)
ens_rec = recall_score(y_test_gt, ens_pred_bin, zero_division=0)
ens_mcc = matthews_corrcoef(y_test_gt, ens_pred_bin)

print(f"\nKẾT QUẢ CUỐI CÙNG TRÊN {data_name}:")
print(f"AUROC     : {ens_auroc:.4f}")
print(f"PRAUC     : {ens_prauc:.4f}")
print(f"Precision : {ens_prec:.4f}")
print(f"Recall    : {ens_rec:.4f}")
print(f"F1-Score  : {ens_f1:.4f}")
print(f"MCC       : {ens_mcc:.4f} (Ngưỡng OOF MCC: {ens_thresh:.2f})")

results_df = pd.DataFrame({
    'Fold': [f'Fold_{i+1}' for i in range(num_folds)] + ['Mean', 'Std', 'Ensemble_5Fold'],
    'AUROC': AUROC + [np.mean(AUROC), np.std(AUROC), ens_auroc],
    'PRAUC': PRAUC + [np.mean(PRAUC), np.std(PRAUC), ens_prauc],
    'F1_Score': F1_SCORE + [np.mean(F1_SCORE), np.std(F1_SCORE), ens_f1],
    'Precision': PRECISION + [np.mean(PRECISION), np.std(PRECISION), ens_prec],
    'Recall': RECALL + [np.mean(RECALL), np.std(RECALL), ens_rec],
    'MCC': MCC + [np.mean(MCC), np.std(MCC), ens_mcc]
})

csv_out = os.path.join(save_dir, f"{data_name}_stable_results.csv")
results_df.to_csv(csv_out, index=False)
print(f"\nĐã lưu kết quả tại: {csv_out}")
print("Thời gian chạy:", datetime.now() - start_time)