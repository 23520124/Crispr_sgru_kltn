import os
import sys

# 1. THIẾT LẬP ĐƯỜNG DẪN ĐỒNG BỘ VỚI train_improved.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

for p in [CURRENT_DIR, os.path.join(BASE_DIR, "Train"), BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import random
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, precision_score, recall_score, matthews_corrcoef
from tensorflow.keras.optimizers import Adam

import Encoder_sgRNA_off
from MODEL import Crispr_SGRU

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 2. HÀM MẤT MÁT KẾT HỢP FOCAL - DICE LOSS
# ==========================================
def focal_dice_loss(y_true, y_pred, gamma=1.0, alpha=0.25, smooth=1e-6):
    y_pos_true = y_true[:, 1]
    y_pos_pred = K.clip(y_pred[:, 1], 1e-7, 1.0 - 1e-7)
    
    pt_1 = tf.where(tf.equal(y_pos_true, 1.0), y_pos_pred, tf.ones_like(y_pos_pred))
    pt_0 = tf.where(tf.equal(y_pos_true, 0.0), y_pos_pred, tf.zeros_like(y_pos_pred))
    focal = - (alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1) +
               (1.0 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0))
    focal_loss = K.mean(focal)
    
    intersection = K.sum(y_pos_true * y_pos_pred)
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (K.sum(y_pos_true) + K.sum(y_pos_pred) + smooth)
    
    return focal_loss + dice_loss

# ==========================================
# 3. ĐỌC FILE CSV VÀ MÃ HÓA
# ==========================================
def load_data(file_path):
    Negative = []
    Positive = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ll = line.strip().split(',')
            try:
                label_item = np.float64(ll[2])
                if label_item == 0.0:
                    Negative.append(ll)
                else:
                    Positive.append(ll)
            except (ValueError, IndexError):
                continue  # Bỏ qua dòng tiêu đề nếu có
    return np.array(Negative), np.array(Positive)

def encoding_df(data_df):
    encode = []
    for _, row in data_df.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        label = row[2]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        encode.append(en.on_off_code)
    arr = np.array(encode, dtype=np.float32)
    return arr.reshape((len(arr), 1, 24, 7))

# ==========================================
# 4. GENERATOR TỶ LỆ 1:1 THEO BÀI BÁO TÁC GIẢ
# ==========================================
def flow_1to1(neg_data, pos_data, batch_size=256, is_train=True):
    num_neg = len(neg_data)
    num_pos = len(pos_data)
    half_batch = batch_size // 2  # 128 âm, 128 dương
    
    neg_indices = np.arange(num_neg)
    if is_train:
        np.random.shuffle(neg_indices)
    
    pos_labels = np.ones(half_batch, dtype=np.int32)
    neg_labels = np.zeros(half_batch, dtype=np.int32)
    batch_y = to_categorical(np.concatenate([pos_labels, neg_labels]), num_classes=2)
    
    idx = 0
    while True:
        if idx + half_batch > num_neg:
            if is_train:
                np.random.shuffle(neg_indices)
            idx = 0
            
        cur_neg_idx = neg_indices[idx : idx + half_batch]
        batch_neg_x = neg_data[cur_neg_idx]
        idx += half_batch
        
        cur_pos_idx = np.random.choice(num_pos, half_batch, replace=True)
        batch_pos_x = pos_data[cur_pos_idx]
        
        batch_x = np.concatenate([batch_pos_x, batch_neg_x], axis=0)
        yield batch_x, batch_y

# ==========================================
# 5. CHƯƠNG TRÌNH HUẤN LUYỆN
# ==========================================
if __name__ == "__main__":
    # Cấu hình tập dữ liệu
    data_name = "k562"  # "k562", "I1", "I2", "II5", "II6", "BE3", "hek"

    dataset_path_map = {
        "hek": os.path.join(BASE_DIR, "Datasets/HEK293t.csv"),
        "k562": os.path.join(BASE_DIR, "Datasets/K562.csv"),
        "II5": os.path.join(BASE_DIR, "Datasets/II5.csv"),
        "II6": os.path.join(BASE_DIR, "Datasets/II6.csv"),
        "BE3": os.path.join(BASE_DIR, "Datasets/BE3.csv"),
        "I2": os.path.join(BASE_DIR, "Datasets/I2.csv"),
        "I1": os.path.join(BASE_DIR, "Datasets/I1.csv"),
        "CHANGE-seq": os.path.join(BASE_DIR, "Datasets/CHANGE-seq/CHANGE-seq.csv"),
    }

    file_path = dataset_path_map[data_name]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    output_dir = os.path.join(BASE_DIR, "weights", f"{data_name}_improved2")
    os.makedirs(output_dir, exist_ok=True)
    results_csv_path = os.path.join(output_dir, f"{data_name}_improved2_results.csv")

    print(f"\n=======================================================")
    print(f"BẮT ĐẦU CHẠY IMPROVED2 TRÊN: {data_name.upper()}")
    print(f"File dữ liệu: {file_path}")
    print(f"Đặc tính: Tích lũy 150 Epochs (No Reset) + Lấy mẫu 1:1 + Focal-Dice")
    print(f"=======================================================\n")

    Negative, Positive = load_data(file_path)
    print(f"Mẫu nạp thành công: Dương={len(Positive)} | Âm={len(Negative)}")

    # Phân chia cố định 1 lần duy nhất theo code tác giả
    Train_Val_Neg, Test_Neg = train_test_split(Negative, test_size=0.15, random_state=42)
    Train_Val_Pos, Test_Pos = train_test_split(Positive, test_size=0.15, random_state=42)

    Train_Neg, Val_Neg = train_test_split(Train_Val_Neg, test_size=0.2, random_state=42)
    Train_Pos, Val_Pos = train_test_split(Train_Val_Pos, test_size=0.2, random_state=42)

    Xtest_raw = np.vstack((Test_Neg, Test_Pos))
    np.random.seed(42)
    np.random.shuffle(Xtest_raw)
    y_test_gt = np.array([1 if float(x[2]) > 0.0 else 0 for x in Xtest_raw])

    print("Đang mã hóa dữ liệu...")
    Train_Pos_enc = encoding_df(pd.DataFrame(Train_Pos))
    Train_Neg_enc = encoding_df(pd.DataFrame(Train_Neg))
    Val_Pos_enc   = encoding_df(pd.DataFrame(Val_Pos))
    Val_Neg_enc   = encoding_df(pd.DataFrame(Val_Neg))
    Xtest_enc     = encoding_df(pd.DataFrame(Xtest_raw))
    print("Mã hóa hoàn tất!\n")

    # Khởi tạo mô hình ngoài vòng lặp (giữ nguyên để học dồn 150 epochs)
    model = Crispr_SGRU()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=focal_dice_loss, metrics=['accuracy'])

    BATCH_SIZE = 256
    EPOCHS_PER_STAGE = 30
    NUM_STAGES = 5
    steps_per_epoch = max(1, len(Train_Neg) // (BATCH_SIZE // 2))

    records = []
    start_time = datetime.now()

    for stage in range(NUM_STAGES):
        accumulated_epochs = (stage + 1) * EPOCHS_PER_STAGE
        print(f"\n>>> ĐANG CHẠY CHẶNG #{stage + 1} (Tích lũy đến Epoch {accumulated_epochs}/150)...")

        train_gen = flow_1to1(Train_Neg_enc, Train_Pos_enc, batch_size=BATCH_SIZE, is_train=True)
        val_gen   = flow_1to1(Val_Neg_enc, Val_Pos_enc, batch_size=BATCH_SIZE, is_train=False)

        model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS_PER_STAGE,
            validation_data=val_gen,
            validation_steps=1,
            verbose=2
        )

        weight_path = os.path.join(output_dir, f"{data_name}_stage{stage+1}_ep{accumulated_epochs}.weights.h5")
        model.save_weights(weight_path)

        y_test_pred = model.predict(Xtest_enc, verbose=0)
        y_test_prob = y_test_pred[:, 1]

        # Ngưỡng cố định 0.5 theo code tác giả: int(i[1] > i[0])
        y_pred_binary = (y_test_prob > 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_test_gt, y_test_prob)
        auroc = auc(fpr, tpr)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test_gt, y_test_prob)
        prauc = auc(rec_curve, prec_curve)

        f1 = f1_score(y_test_gt, y_pred_binary, zero_division=0)
        prec = precision_score(y_test_gt, y_pred_binary, zero_division=0)
        rec = recall_score(y_test_gt, y_pred_binary, zero_division=0)
        mcc = matthews_corrcoef(y_test_gt, y_pred_binary)

        print(f"-> Chặng #{stage + 1} ({accumulated_epochs} eps): AUROC={auroc:.4f} | PRAUC={prauc:.4f} | F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | MCC={mcc:.4f}")

        records.append({
            "Stage": f"Stage_{stage+1}_{accumulated_epochs}eps",
            "AUROC": auroc,
            "PRAUC": prauc,
            "F1_Score": f1,
            "Precision": prec,
            "Recall": rec,
            "MCC": mcc
        })

    df_results = pd.DataFrame(records)

    mean_row = {"Stage": "Mean"}
    std_row = {"Stage": "Std"}
    for col in ["AUROC", "PRAUC", "F1_Score", "Precision", "Recall", "MCC"]:
        mean_row[col] = df_results[col].mean()
        std_row[col] = df_results[col].std()

    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    df_results.to_csv(results_csv_path, index=False)

    total_time = datetime.now() - start_time
    print("\n" + "="*60)
    print(f"HOÀN THÀNH HUẤN LUYỆN TÍCH LŨY 150 EPOCHS!")
    print(f"Kết quả lưu tại: {results_csv_path}")
    print(f"Tổng thời gian chạy: {total_time}")
    print("="*60)