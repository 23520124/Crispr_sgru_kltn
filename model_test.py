import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as Shuffle

import sys
sys.path.append(os.path.abspath("Train"))

from MODEL import Crispr_SGRU
import Encoder_sgRNA_off

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_data(file_path):
    Negative = []
    Positive = []
    label = []
    with open(file_path, 'r') as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]
            label_item = np.float64(ll[2])
            if label_item == 0.0:
                Negative.append(ll)
            else:
                Positive.append(ll)
            label.append(label_item)
    return Negative, Positive, label

def encoding_xtest(Xtest_df):
    final_code = []
    for idx, row in Xtest_df.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        label_val = row[2]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label_val)
        final_code.append(en.on_off_code)
    return np.array(final_code)

# 1. Tải và phân chia tập dữ liệu kiểm thử
# Dataset dùng để test (có thể khác với dataset train)
test_dataset_name = "BE3"  # Options: k562, hek, BE3, CHANGEseq, II5, II6

# Map dataset name to file path
dataset_map = {
    "k562": "Datasets/K562.csv",
    "hek": "Datasets/HEK293t.csv",
    "BE3": "Datasets/BE3.csv",
    "CHANGEseq": "Datasets/CHANGE-seq/CHANGE-seq.csv",
    "II5": "Datasets/II5.csv",
    "II6": "Datasets/II6.csv"
}

if test_dataset_name not in dataset_map:
    print(f"ERROR: Unknown dataset name '{test_dataset_name}'")
    print(f"Available options: {list(dataset_map.keys())}")
    exit(1)

dataset_path = dataset_map[test_dataset_name]
print(f"Loading TEST dataset: {test_dataset_name}")
print(f"File path: {dataset_path}")

Negative, Positive, label = load_data(dataset_path)
Negative = Shuffle(Negative, random_state=42)
Positive = Shuffle(Positive, random_state=42)

_, Test_Negative = train_test_split(Negative, test_size=0.15, random_state=42)
_, Test_Positive = train_test_split(Positive, test_size=0.15, random_state=42)

Xtest = np.vstack((Test_Negative, Test_Positive))
Xtest = Shuffle(Xtest, random_state=42)
Xtest_df = pd.DataFrame(Xtest)

print(f"Test set size: {len(Xtest_df)}")

# 2. Mã hóa chuỗi sang ma trận đầu vào
print("Encoding sequences...")
Xtest_encoded = encoding_xtest(Xtest_df)
if len(Xtest_encoded.shape) == 3:
    Xtest_encoded = np.expand_dims(Xtest_encoded, axis=1)

print(f"Encoded shape: {Xtest_encoded.shape}")

# 3. Tải mô hình và trọng số đã huấn luyện (ensemble 5 folds)
print("Building model...")

# Weights từ dataset nào? (có thể khác với test dataset)
weights_dataset_name = "BE3"  # Options: k562, hek, BE3, CHANGEseq, II5, II6

# Tìm tất cả weights files cho dataset này
weights_folder = f"weights/{weights_dataset_name}"
if os.path.exists(weights_folder):
    weight_files = sorted([f for f in os.listdir(weights_folder) if f.endswith('.h5')])
    if not weight_files:
        print(f"ERROR: No .h5 files found in {weights_folder}")
        exit(1)

    print(f"Found {len(weight_files)} weight files in {weights_folder}/")
    print(f"Using ensemble prediction with {len(weight_files)} models")
    print(f"Weights trained on: {weights_dataset_name}")
    print(f"Testing on: {test_dataset_name}")
    if weights_dataset_name == test_dataset_name:
        print("WARNING: Same dataset for train and test - results may be overfitted!")
    print()

    # Dự đoán với ensemble (trung bình của tất cả folds)
    all_predictions = []
    for i, weight_file in enumerate(weight_files):
        weight_path = os.path.join(weights_folder, weight_file)
        print(f"  [{i+1}/{len(weight_files)}] Loading {weight_file}...")

        model = Crispr_SGRU()
        model.load_weights(weight_path)

        # Dự đoán với model này
        y_pred_prob = model.predict(Xtest_encoded, verbose=0)
        all_predictions.append(y_pred_prob)

    # Trung bình các predictions từ tất cả folds
    y_pred_prob = np.mean(all_predictions, axis=0)

else:
    print(f"ERROR: Weights folder not found at {weights_folder}")
    print("Available datasets in weights folder:")
    if os.path.exists("weights"):
        available = [d for d in os.listdir("weights") if os.path.isdir(os.path.join("weights", d))]
        for d in available:
            print(f"  - {d}")
    exit(1)

# 4. Đánh giá kết quả
print("\nEvaluating predictions...")
y_prob = y_pred_prob[:, 1]
y_pred = (y_prob > 0.5).astype(int)

y_test = [1 if float(val) > 0.0 else 0 for val in Xtest_df.iloc[:, 2]]

# Tính toán các metrics
fpr, tpr, _ = roc_curve(y_test, y_prob)
auroc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
prauc = auc(recall, precision)
f1score = f1_score(y_test, y_pred)
precision_scores = precision_score(y_test, y_pred)
recall_scores = recall_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print("AUROC=%.3f, PRAUC=%.3f, F1score=%.3f, Precision=%.3f, Recall=%.3f, MCC=%.3f" %
      (auroc, prauc, f1score, precision_scores, recall_scores, mcc))
print("="*60)