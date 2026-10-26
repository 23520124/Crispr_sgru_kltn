"""Small E3 ablation: add an explicit per-position mismatch interaction branch."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import tensorflow as tf
from sklearn.metrics import (
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Train"))

from biological_features import extract_features_batch
from MODEL_E3_BioFusion import BCEDiceLoss
from train_e3_biofusion import count_mismatches, encoding, load_data


DATASET_PATHS = {
    "K562": os.path.join(BASE_DIR, "Datasets", "K562.csv"),
    "HEK293t": os.path.join(BASE_DIR, "Datasets", "HEK293t.csv"),
    "II5": os.path.join(BASE_DIR, "Datasets", "II5.csv"),
    "II6": os.path.join(BASE_DIR, "Datasets", "II6.csv"),
    "BE3": os.path.join(BASE_DIR, "Datasets", "BE3.csv"),
    "I1": os.path.join(BASE_DIR, "Datasets", "I1.csv"),
    "I2": os.path.join(BASE_DIR, "Datasets", "I2.csv"),
    "CHANGE-seq": os.path.join(BASE_DIR, "Datasets", "CHANGE-seq", "CHANGE-seq.csv"),
}


def mismatch_interaction(rows, mode="substitution"):
    """Encode position-wise mismatch, substitution, or full base-pair states."""
    base_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    if mode == "mask":
        result = np.zeros((len(rows), 20, 1), dtype="float32")
        for n, row in enumerate(rows):
            sg, off = str(row[0]).upper(), str(row[1]).upper()
            for pos, (a, b) in enumerate(zip(sg[:20], off[:20])):
                if a != b:
                    result[n, pos, 0] = 1.0
        return result
    result = np.zeros((len(rows), 20, 16), dtype="float32")
    for n, row in enumerate(rows):
        sg, off = str(row[0]).upper(), str(row[1]).upper()
        for pos, (a, b) in enumerate(zip(sg[:20], off[:20])):
            if (mode == "pairwise_gru" or a != b) and a in base_index and b in base_index:
                result[n, pos, base_index[a] * 4 + base_index[b]] = 1.0
    return result


def build_model(bio_dim, interaction="mask", learning_rate=1e-4):
    sequence_input = Input(shape=(1, 24, 7), name="sequence_input")
    c1 = layers.Conv2D(10, (1, 1), padding="same", activation="relu")(sequence_input)
    c2 = layers.Conv2D(10, (1, 2), padding="same", activation="relu")(sequence_input)
    c3 = layers.Conv2D(10, (1, 3), padding="same", activation="relu")(sequence_input)
    c4 = layers.Conv2D(10, (1, 5), padding="same", activation="relu")(sequence_input)
    x0 = layers.Concatenate(axis=-1)([c1, c2, c3, c4])
    x0 = layers.Reshape((24, 40))(x0)
    g0 = layers.Bidirectional(layers.GRU(30, return_sequences=True))(x0)
    raw = layers.Reshape((24, 7))(sequence_input)
    x = layers.Concatenate(axis=2)([raw, g0])
    g1 = layers.Bidirectional(layers.GRU(20, return_sequences=True))(x)
    x = layers.Concatenate(axis=2)([g0, g1])
    g2 = layers.Bidirectional(layers.GRU(10, return_sequences=True))(x)
    x = layers.Concatenate(axis=-1)([g0, g1, g2])
    x = layers.Flatten()(x)
    seq_repr = layers.Dense(128, activation="relu")(x)
    seq_repr = layers.Dense(64, activation="relu", name="seq_repr")(seq_repr)
    seq_repr = layers.Dropout(0.35)(seq_repr)

    bio_input = Input(shape=(bio_dim,), name="bio_input")
    bio_repr = layers.Dense(64, activation="relu", name="bio_dense")(bio_input)
    bio_repr = layers.LayerNormalization(name="bio_norm")(bio_repr)
    bio_repr = layers.Dropout(0.2, name="bio_dropout")(bio_repr)

    interaction_dim = 1 if interaction == "mask" else 16
    mismatch_input = Input(shape=(20, interaction_dim), name="mismatch_input")
    if interaction == "pairwise_gru":
        mismatch_repr = layers.Conv1D(16, 3, padding="same", activation="relu", name="pair_conv")(mismatch_input)
        mismatch_repr = layers.Bidirectional(layers.GRU(8, return_sequences=False), name="pair_bigru")(mismatch_repr)
        mismatch_repr = layers.LayerNormalization(name="pair_norm")(mismatch_repr)
        mismatch_repr = layers.Dropout(0.1, name="pair_dropout")(mismatch_repr)
    else:
        mismatch_repr = layers.Flatten(name="mismatch_flatten")(mismatch_input)
        mismatch_units = 8 if interaction == "mask" else 32
        mismatch_repr = layers.Dense(mismatch_units, activation="relu", name="mismatch_dense")(mismatch_repr)
        mismatch_repr = layers.LayerNormalization(name="mismatch_norm")(mismatch_repr)
        mismatch_repr = layers.Dropout(0.1, name="mismatch_dropout")(mismatch_repr)

    fusion = layers.Concatenate(name="fusion")([seq_repr, bio_repr, mismatch_repr])
    fusion = layers.Dense(64, activation="relu", name="fusion_dense")(fusion)
    fusion = layers.Dropout(0.3)(fusion)
    output = layers.Dense(1, activation="sigmoid", name="prob_output")(fusion)
    model = Model(
        inputs=[sequence_input, bio_input, mismatch_input],
        outputs=output,
        name="MODEL12_BioFusion_MismatchAblation",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=False),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"), tf.keras.metrics.AUC(curve="ROC", name="auroc")],
    )
    return model


class MismatchBioFusionGenerator:
    def __init__(self, pos_seq, pos_bio, pos_mm, hard_seq, hard_bio, hard_mm,
                 norm_seq, norm_bio, norm_mm, steps=120, seed=42):
        self.pos = (pos_seq, pos_bio, pos_mm)
        self.hard = (hard_seq, hard_bio, hard_mm)
        self.norm = (norm_seq, norm_bio, norm_mm)
        self.steps = steps
        self.rng = np.random.RandomState(seed)

    def flow(self):
        while True:
            for _ in range(self.steps):
                pi = self.rng.choice(len(self.pos[0]), 64, replace=True)
                hi = self.rng.choice(len(self.hard[0]), 64, replace=True)
                ni = self.rng.choice(len(self.norm[0]), 128, replace=True)
                seq = np.concatenate([self.pos[0][pi], self.hard[0][hi], self.norm[0][ni]])
                bio = np.concatenate([self.pos[1][pi], self.hard[1][hi], self.norm[1][ni]])
                mm = np.concatenate([self.pos[2][pi], self.hard[2][hi], self.norm[2][ni]])
                y = np.concatenate([np.ones(64), np.zeros(192)]).astype("float32")
                order = self.rng.permutation(256)
                yield (seq[order], bio[order], mm[order]), y[order].reshape(-1, 1)


def metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
    return {
        "AUROC": float(roc_auc_score(y_true, probs)),
        "PRAUC": float(auc(recall_curve, precision_curve)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, pred)),
    }


def best_f1_threshold(y_true, probs):
    candidates = np.unique(probs)
    scores = [(f1_score(y_true, probs >= t, zero_division=0), float(t)) for t in candidates]
    return min(((-score, threshold) for score, threshold in scores))[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="II5", choices=sorted(DATASET_PATHS))
    parser.add_argument("--feature-set", default="e3b", choices=["e3a", "e3b", "e3c"])
    parser.add_argument("--interaction", default="mask", choices=["mask", "substitution", "pairwise_gru"])
    args = parser.parse_args()
    path = DATASET_PATHS[args.dataset]
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, "weights", f"{args.dataset}_E3_mismatch_{args.interaction}_{args.feature_set}_{timestamp}")
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=False)
    neg, pos = load_data(path)
    neg, pos = np.array(neg, dtype=object), np.array(pos, dtype=object)
    train_neg, test_neg = train_test_split(neg, test_size=0.15, random_state=42)
    train_pos, test_pos = train_test_split(pos, test_size=0.15, random_state=42)
    test_raw = np.vstack([test_neg, test_pos])
    rng = np.random.RandomState(42)
    test_raw = test_raw[rng.permutation(len(test_raw))]
    test_enc = encoding(pd.DataFrame(test_raw[:, :3])).reshape((-1, 1, 24, 7)).astype("float32")
    test_bio_raw = extract_features_batch(test_raw.tolist(), feature_set=args.feature_set)
    test_mm = mismatch_interaction(test_raw, args.interaction)
    test_y = np.array([int(float(x[2]) > 0) for x in test_raw])

    pool = np.vstack([train_neg, train_pos])
    pool_y = np.array([0] * len(train_neg) + [1] * len(train_pos))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(pool, pool_y))
    results = []
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        seed = 42 + fold * 100
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tr, va = pool[tr_idx], pool[va_idx]
        tr_y, va_y = pool_y[tr_idx], pool_y[va_idx]
        tr_pos, tr_neg = tr[tr_y == 1], tr[tr_y == 0]
        hard_mask = np.array([count_mismatches(r[0], r[1]) <= 3 for r in tr_neg])
        hard, normal = tr_neg[hard_mask], tr_neg[~hard_mask]
        if len(hard) == 0 or len(normal) == 0:
            raise ValueError(f"Fold {fold} lacks hard or normal negatives")

        def enc(rows):
            return encoding(pd.DataFrame([r[:3] for r in rows])).reshape((-1, 1, 24, 7)).astype("float32")
        pos_seq, hard_seq, norm_seq, va_seq = map(enc, [tr_pos, hard, normal, va])
        pos_bio = extract_features_batch(tr_pos.tolist(), feature_set=args.feature_set)
        hard_bio = extract_features_batch(hard.tolist(), feature_set=args.feature_set)
        norm_bio = extract_features_batch(normal.tolist(), feature_set=args.feature_set)
        va_bio = extract_features_batch(va.tolist(), feature_set=args.feature_set)
        scaler = StandardScaler().fit(np.vstack([pos_bio, hard_bio, norm_bio]))
        pos_bio, hard_bio = scaler.transform(pos_bio), scaler.transform(hard_bio)
        norm_bio, va_bio = scaler.transform(norm_bio), scaler.transform(va_bio)
        pos_mm = mismatch_interaction(tr_pos, args.interaction)
        hard_mm = mismatch_interaction(hard, args.interaction)
        norm_mm = mismatch_interaction(normal, args.interaction)
        va_mm = mismatch_interaction(va, args.interaction)
        model = build_model(pos_bio.shape[1], interaction=args.interaction)
        weight_path = os.path.join(run_dir, f"fold{fold}.weights.h5")
        model.fit(
            MismatchBioFusionGenerator(pos_seq, pos_bio, pos_mm, hard_seq, hard_bio, hard_mm, norm_seq, norm_bio, norm_mm, seed=seed).flow(),
            validation_data=((va_seq, va_bio, va_mm), va_y.astype("float32").reshape(-1, 1)),
            epochs=35, steps_per_epoch=120,
            callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6), ModelCheckpoint(weight_path, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True)],
            verbose=1,
        )
        model.load_weights(weight_path)
        va_probs = model.predict((va_seq, va_bio, va_mm), verbose=0).ravel()
        test_bio = scaler.transform(test_bio_raw)
        test_probs = model.predict((test_enc, test_bio, test_mm), verbose=0).ravel()
        threshold = best_f1_threshold(va_y, va_probs)
        for split, y, probs in [("val", va_y, va_probs), ("test", test_y, test_probs)]:
            row = {"Fold": fold, "Split": split, "Threshold": threshold, **metrics(y, probs, threshold)}
            results.append(row)
            ids = [int(r[3]) for r in (va if split == "val" else test_raw)]
            pd.DataFrame({"source_id": ids, "y_true": y, "y_prob": probs}).to_csv(os.path.join(pred_dir, f"fold{fold}_{split}.csv"), index=False)
        tf.keras.backend.clear_session()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, "per_fold_results.csv"), index=False)
    summary = []
    for split in ["val", "test"]:
        part = df[df.Split == split]
        summary.append({"Split": split, "Statistic": "Mean", **part.select_dtypes("number").mean().to_dict()})
        summary.append({"Split": split, "Statistic": "Std", **part.select_dtypes("number").std(ddof=0).to_dict()})
    pd.DataFrame(summary).to_csv(os.path.join(run_dir, "summary_results.csv"), index=False)
    with open(os.path.join(run_dir, "config.json"), "w") as h:
        ablation = {
            "mask": "20-position mismatch mask with Dense branch",
            "substitution": "20-position x 16 mismatch substitution one-hot with Dense branch",
            "pairwise_gru": "20-position x 16 full sgRNA-DNA base-pair one-hot with Conv1D-BiGRU branch",
        }[args.interaction]
        json.dump({"dataset": args.dataset, "feature_set": args.feature_set, "interaction": args.interaction, "ablation": ablation, "protocol": "E3 15-percent fixed test + 5-fold stratified CV", "epochs": 35, "steps_per_epoch": 120, "loss": "BCEDice", "sampling": "64 positive + 64 hard negative + 128 normal negative"}, h, indent=2)
    print(df[df.Split == "val"].to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
