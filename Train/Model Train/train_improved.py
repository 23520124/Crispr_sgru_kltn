import os
import sys

# 1. CẤU HÌNH GPU: Phải đặt trước khi import TensorFlow / Keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import random
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(BASE_DIR, "Train"))
sys.path.append(os.path.join(BASE_DIR, "Train/Model Train"))

import Encoder_sgRNA_off
from MODEL import Crispr_SGRU
import loss_generator as lg

def load_data(file_path):
    Negative, Positive, label = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]
            try:
                label_item = np.float64(ll[2])
            except ValueError:
                continue
            if label_item == 0.0:
                Negative.append(ll)
            else:
                Positive.append(ll)
            label.append(label_item)
    return Negative, Positive, label

def train_flow_asymmetric(Train_Negative, Train_Positive, batchsize, neg_ratio=2):
    """
    Generator lấy mẫu bất đối xứng 1 dương : 2 âm.
    Sử dụng padding tuần hoàn ở batch cuối để không bỏ sót mẫu âm tính.
    """
    Num_Positive = len(Train_Positive)
    Num_Negative = len(Train_Negative)
    
    pos_num = max(1, batchsize // (neg_ratio + 1))
    neg_num = pos_num * neg_ratio
    actual_batch = pos_num + neg_num
    total_batches = max(1, int(np.ceil(Num_Negative / neg_num)))
    
    i = 0
    while True:
        i += 1
        np.random.seed(i)
        idx_neg = np.random.permutation(Num_Negative)
        epoch_neg = np.resize(idx_neg, total_batches * neg_num)
        
        for b in range(total_batches):
            neg_idx = epoch_neg[b * neg_num : (b + 1) * neg_num]
            pos_idx = np.random.randint(0, Num_Positive, pos_num, dtype='int32')
            
            X_batch = np.concatenate([Train_Negative[neg_idx], Train_Positive[pos_idx]], axis=0)
            Y_batch = np.array([0] * neg_num + [1] * pos_num)
            
            shuffle_idx = np.random.permutation(actual_batch)
            X_batch = X_batch[shuffle_idx]
            Y_batch = to_categorical(Y_batch[shuffle_idx], num_classes=2)
            
            yield X_batch, Y_batch

def encoding(data):
    encode = []
    for _, row in data.iterrows():
        en = Encoder_sgRNA_off.Encoder(on_seq=row[0], off_seq=row[1], with_category=True, label=row[2])
        encode.append(en.on_off_code)
    return np.array(encode)

# --- CẤU HÌNH THỰC NGHIỆM ---
data_name = "k562"  # Thay tên dataset: "k562", "hek", "II5", "II6", "BE3", "I2", "I1", "CHANGE-seq"
dataset_path_map = {
    "hek": os.path.join(BASE_DIR, "Datasets/HEK293t.csv"),
    "k562": os.path.join(BASE_DIR, "Datasets/K562.csv"),
    "II5": os.path.join(BASE_DIR, "Datasets/II5.csv"),
    "II6": os.path.join(BASE_DIR, "Datasets/II6.csv"),
    "BE3": os.path.join(BASE_DIR, "Datasets/BE3.csv"),
    "I2":  os.path.join(BASE_DIR, "Datasets/I2.csv"),
    "I1":  os.path.join(BASE_DIR, "Datasets/I1.csv"),
    "CHANGE-seq": os.path.join(BASE_DIR, "Datasets/CHANGE-seq/CHANGE-seq.csv")
}

file_path = dataset_path_map.get(data_name)
save_dir = os.path.join(BASE_DIR, "weights", f"{data_name}_improved")
os.makedirs(save_dir, exist_ok=True)

print(f"Bắt đầu huấn luyện mô hình cải tiến trên tập: {file_path}")
Negative, Positive, label = load_data(file_path)
Positive = np.array(Positive)
Negative = np.array(Negative)

# Cố định tập Blind Test (15%) ngoài cùng, không tham gia vào bất kỳ khâu chọn ngưỡng nào
Train_Pool_Neg, Test_Neg = train_test_split(Negative, test_size=0.15, random_state=42)
Train_Pool_Pos, Test_Pos = train_test_split(Positive, test_size=0.15, random_state=42)

# Mã hóa sẵn tập Blind Test một lần duy nhất
Xtest_raw = np.vstack((Test_Neg, Test_Pos))
np.random.seed(42)
np.random.shuffle(Xtest_raw)
Xtest_df = pd.DataFrame(Xtest_raw)
Xtest_enc = encoding(Xtest_df).reshape((-1, 1, 24, 7)).astype("float64")
y_test_gt = np.array([1 if float(x) > 0.0 else 0 for x in Xtest_df.iloc[:, 2]])

BATCH_SIZE = 256
epochs = 35
num_folds = 5
neg_ratio = 2

AUROC, PRAUC, F1_SCORE, PRECISION, RECALL, MCC = [], [], [], [], [], []
start_time = datetime.now()

# StratifiedKFold trên 85% Train Pool: Mỗi mẫu validation xuất hiện đúng 1 lần
train_pool = np.vstack((Train_Pool_Neg, Train_Pool_Pos))
train_pool_y = np.array([0] * len(Train_Pool_Neg) + [1] * len(Train_Pool_Pos))
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_splits = list(skf.split(train_pool, train_pool_y))
oof_probs = np.full(len(train_pool), np.nan, dtype=np.float32)

# =========================================================================
# VÒNG LẶP HUẤN LUYỆN 5-FOLD
# =========================================================================
for fold in range(num_folds):
    print(f"\n==================== RUNNING IMPROVED FOLD #{fold + 1} ====================")
    train_idx, val_idx = fold_splits[fold]
    Train_raw, Val_raw = train_pool[train_idx], train_pool[val_idx]
    Train_y, Val_y = train_pool_y[train_idx], train_pool_y[val_idx]

    # Tách âm/dương cho tập Train để đưa vào generator bất đối xứng
    Train_Neg = Train_raw[Train_y == 0]
    Train_Pos = Train_raw[Train_y == 1]
    Train_Pos_enc = encoding(pd.DataFrame(Train_Pos)).reshape((-1, 1, 24, 7)).astype("float64")
    Train_Neg_enc = encoding(pd.DataFrame(Train_Neg)).reshape((-1, 1, 24, 7)).astype("float64")

    # Mã hóa tập Validation trực tiếp 1 lần, bảo toàn tuyệt đối thứ tự khớp với val_idx
    Xval_enc = encoding(pd.DataFrame(Val_raw)).reshape((-1, 1, 24, 7)).astype("float64")
    y_val_gt = Val_y
    y_val_cat = to_categorical(y_val_gt, num_classes=2)

    pos_per_batch = max(1, BATCH_SIZE // (neg_ratio + 1))
    neg_per_batch = pos_per_batch * neg_ratio
    NUM_BATCH = max(1, int(np.ceil(len(Train_Neg_enc) / neg_per_batch)))

    model = Crispr_SGRU()
    model.compile(
        loss=lg.focal_dice_loss(alpha=0.75, gamma=1.0, weight_dice=0.5),
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    weight_path = os.path.join(save_dir, f"{data_name}_improved_fold{fold + 1}.weights.h5")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(
        weight_path, monitor='val_loss', mode='min', save_best_only=True,
        save_weights_only=True, verbose=0
    )

    model.fit(
        train_flow_asymmetric(Train_Neg_enc, Train_Pos_enc, BATCH_SIZE, neg_ratio=neg_ratio),
        validation_data=(Xval_enc, y_val_cat),
        epochs=epochs,
        steps_per_epoch=NUM_BATCH,
        callbacks=[reduce_lr, checkpoint],
        verbose=2
    )

    # Nạp lại checkpoint đạt val_loss thấp nhất
    model.load_weights(weight_path)

    # 1. Tối ưu ngưỡng cắt trên VALIDATION SET (Loại bỏ rò rỉ dữ liệu)
    val_probs = model.predict(Xval_enc, verbose=0)[:, 1]
    best_thresh = 0.5
    best_val_f1 = -1.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        val_preds = (val_probs >= thresh).astype(int)
        score = f1_score(y_val_gt, val_preds, zero_division=0)
        if score > best_val_f1:
            best_val_f1 = score
            best_thresh = thresh

    # Lưu xác suất OOF vào đúng chỉ mục toàn cục
    oof_probs[val_idx] = val_probs

    # 2. Đánh giá mô hình fold hiện tại trên BLIND TEST bằng ngưỡng cố định từ Validation
    test_probs = model.predict(Xtest_enc, verbose=0)[:, 1]
    test_pred_binary = (test_probs >= best_thresh).astype(int)

    fpr, tpr, _ = roc_curve(y_test_gt, test_probs)
    fold_auroc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_gt, test_probs)
    fold_prauc = auc(recall_curve, precision_curve)
    fold_f1 = f1_score(y_test_gt, test_pred_binary, zero_division=0)
    fold_prec = precision_score(y_test_gt, test_pred_binary, zero_division=0)
    fold_rec = recall_score(y_test_gt, test_pred_binary, zero_division=0)
    fold_mcc = matthews_corrcoef(y_test_gt, test_pred_binary)

    print(f"Fold #{fold + 1} Test Result -> AUROC={fold_auroc:.3f}, PRAUC={fold_prauc:.3f}, F1={fold_f1:.3f}, Prec={fold_prec:.3f}, Rec={fold_rec:.3f}, MCC={fold_mcc:.3f} (Thresh={best_thresh:.2f})")

    AUROC.append(fold_auroc)
    PRAUC.append(fold_prauc)
    F1_SCORE.append(fold_f1)
    PRECISION.append(fold_prec)
    RECALL.append(fold_rec)
    MCC.append(fold_mcc)

    # Dọn dẹp tài nguyên
    del (Xval_enc, y_val_gt, y_val_cat, val_probs, model, test_probs,
         Train_Pos_enc, Train_Neg_enc, Train_raw, Val_raw, Train_y, Val_y)
    K.clear_session()
    gc.collect()

# =========================================================================
# ĐÁNH GIÁ SOFT-VOTING ENSEMBLE KẾT HỢP CẢ 5 CHECKPOINTS
# =========================================================================
print("\n" + "="*60)
print("TIẾN HÀNH ĐÁNH GIÁ SOFT-VOTING ENSEMBLE TỪ 5 MÔ HÌNH FOLD")
print("="*60)

ensemble_probs = np.zeros(len(Xtest_enc))
for f in range(num_folds):
    m_ens = Crispr_SGRU()
    w_file = os.path.join(save_dir, f"{data_name}_improved_fold{f + 1}.weights.h5")
    m_ens.load_weights(w_file)
    ensemble_probs += m_ens.predict(Xtest_enc, verbose=0)[:, 1]
    del m_ens
    K.clear_session()
    gc.collect()

ensemble_probs /= num_folds

# Tìm ngưỡng tối ưu cho Ensemble dựa trên toàn bộ phân phối OOF
ens_thresh = 0.5
best_oof_f1 = -1.0
for thresh in np.arange(0.1, 0.9, 0.02):
    score = f1_score(train_pool_y, (oof_probs >= thresh).astype(int), zero_division=0)
    if score > best_oof_f1:
        best_oof_f1 = score
        ens_thresh = float(thresh)

ens_pred_binary = (ensemble_probs >= ens_thresh).astype(int)

fpr, tpr, _ = roc_curve(y_test_gt, ensemble_probs)
ens_auroc = auc(fpr, tpr)
precision_curve, recall_curve, _ = precision_recall_curve(y_test_gt, ensemble_probs)
ens_prauc = auc(recall_curve, precision_curve)
ens_f1 = f1_score(y_test_gt, ens_pred_binary, zero_division=0)
ens_prec = precision_score(y_test_gt, ens_pred_binary, zero_division=0)
ens_rec = recall_score(y_test_gt, ens_pred_binary, zero_division=0)
ens_mcc = matthews_corrcoef(y_test_gt, ens_pred_binary)

print(f"KẾT QUẢ SOFT-VOTING ENSEMBLE -> AUROC: {ens_auroc:.3f} | PRAUC: {ens_prauc:.3f} | F1: {ens_f1:.3f} | Prec: {ens_prec:.3f} | Rec: {ens_rec:.3f} | MCC: {ens_mcc:.3f} (Ngưỡng OOF: {ens_thresh:.2f})")

# =========================================================================
# LẬP BẢNG TỔNG KẾT VÀ XUẤT FILE CSV
# =========================================================================
results_df = pd.DataFrame({
    'Fold': [f'Fold_{i+1}' for i in range(num_folds)] + ['Mean', 'Std', 'Ensemble_5Fold'],
    'AUROC': AUROC + [np.mean(AUROC), np.std(AUROC), ens_auroc],
    'PRAUC': PRAUC + [np.mean(PRAUC), np.std(PRAUC), ens_prauc],
    'F1_Score': F1_SCORE + [np.mean(F1_SCORE), np.std(F1_SCORE), ens_f1],
    'Precision': PRECISION + [np.mean(PRECISION), np.std(PRECISION), ens_prec],
    'Recall': RECALL + [np.mean(RECALL), np.std(RECALL), ens_rec],
    'MCC': MCC + [np.mean(MCC), np.std(MCC), ens_mcc]
})

csv_out = os.path.join(save_dir, f"{data_name}_improved_results.csv")
results_df.to_csv(csv_out, index=False)

print("\n" + "="*60)
print(f"ĐÃ LƯU BẢNG KẾT QUẢ TẠI: {csv_out}")
print("Thời gian chạy:", datetime.now() - start_time)
print("="*60)