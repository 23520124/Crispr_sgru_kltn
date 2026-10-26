"""E3 BioFusion ablation with a same-sgRNA pairwise ranking objective."""

import argparse
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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Train"))

from biological_features import extract_features_batch
from MODEL_E3_BioFusion import Crispr_SGRU_BioFusion
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


class SameGuidePairGenerator:
    """Keeps E3's batch size and pairs positives with nearest same-guide negatives."""

    def __init__(self, seq, bio, rows, labels, steps=120, seed=42):
        self.seq = seq
        self.bio = bio
        self.rows = rows
        self.labels = labels
        self.steps = steps
        self.rng = np.random.RandomState(seed)
        self.pos_ids = np.flatnonzero(labels == 1)
        self.neg_ids = np.flatnonzero(labels == 0)
        self.normal_ids = np.array(
            [i for i in self.neg_ids if count_mismatches(rows[i][0], rows[i][1]) > 3],
            dtype=int,
        )
        if len(self.pos_ids) == 0 or len(self.neg_ids) == 0 or len(self.normal_ids) == 0:
            raise ValueError("Training fold needs positives, negatives, and normal negatives")
        self.neg_by_guide = {}
        for i in self.neg_ids:
            self.neg_by_guide.setdefault(str(rows[i][0]), []).append(i)
        self.neg_by_guide = {k: np.asarray(v, dtype=int) for k, v in self.neg_by_guide.items()}
        self.pos_by_guide = {}
        for i in self.pos_ids:
            self.pos_by_guide.setdefault(str(rows[i][0]), []).append(i)
        self.unmatched_positive_guides = sorted(
            guide for guide in self.pos_by_guide if guide not in self.neg_by_guide
        )
        if self.unmatched_positive_guides:
            raise ValueError(
                "Cannot form same-guide positive/negative pairs for guides: "
                + ", ".join(self.unmatched_positive_guides)
            )
        self.signatures = {
            i: {
                (pos, str(a), str(b))
                for pos, (a, b) in enumerate(zip(str(rows[i][0])[:20], str(rows[i][1])[:20]), 1)
                if a != b
            }
            for i in list(self.pos_ids) + list(self.neg_ids)
        }
        self.nearest_neg_by_positive = {}
        for pos_i in self.pos_ids:
            guide = str(rows[pos_i][0])
            candidates = self.neg_by_guide[guide]
            distances = [
                (len(self.signatures[pos_i] ^ self.signatures[int(neg_i)]), int(neg_i))
                for neg_i in candidates
            ]
            distances.sort(key=lambda item: (item[0], item[1]))
            self.nearest_neg_by_positive[int(pos_i)] = np.asarray(
                [neg_i for _, neg_i in distances[:min(32, len(distances))]],
                dtype=int,
            )

    def flow(self):
        while True:
            for _ in range(self.steps):
                pos_idx = self.rng.choice(self.pos_ids, 64, replace=True)
                pair_neg_idx = []
                for pos_i in pos_idx:
                    candidates = self.nearest_neg_by_positive[int(pos_i)]
                    pair_neg_idx.append(self.rng.choice(candidates))
                pair_neg_idx = np.asarray(pair_neg_idx, dtype=int)
                normal_idx = self.rng.choice(self.normal_ids, 128, replace=True)
                batch_idx = np.concatenate([pos_idx, pair_neg_idx, normal_idx])
                y = np.concatenate([np.ones(64), np.zeros(192)]).astype("float32")
                yield (self.seq[batch_idx], self.bio[batch_idx]), y.reshape(-1, 1)


class PairwiseTrainer(tf.keras.Model):
    def __init__(self, base_model, ranking_weight=0.1):
        super().__init__(name="E3_SameGuidePairwise")
        self.base_model = base_model
        self.ranking_weight = float(ranking_weight)
        self.base_loss = base_model.loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.base_loss_tracker = tf.keras.metrics.Mean(name="bce_dice")
        self.rank_loss_tracker = tf.keras.metrics.Mean(name="pairwise_loss")
        self.pr_auc = tf.keras.metrics.AUC(curve="PR", name="pr_auc")
        self.auroc = tf.keras.metrics.AUC(curve="ROC", name="auroc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.base_loss_tracker, self.rank_loss_tracker, self.pr_auc, self.auroc]

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    @staticmethod
    def pair_loss(probs):
        eps = tf.keras.backend.epsilon()
        clipped = tf.clip_by_value(probs, eps, 1.0 - eps)
        logits = tf.math.log(clipped / (1.0 - clipped))
        pos_logits = logits[:64]
        paired_neg_logits = logits[64:128]
        return tf.reduce_mean(tf.nn.softplus(-(pos_logits - paired_neg_logits)))

    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            probs = self(inputs, training=True)
            base = self.base_loss(labels, probs)
            pairwise = self.pair_loss(probs)
            total = base + self.ranking_weight * pairwise
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(total)
        self.base_loss_tracker.update_state(base)
        self.rank_loss_tracker.update_state(pairwise)
        self.pr_auc.update_state(labels, probs)
        self.auroc.update_state(labels, probs)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, labels = data
        probs = self(inputs, training=False)
        base = self.base_loss(labels, probs)
        self.loss_tracker.update_state(base)
        self.base_loss_tracker.update_state(base)
        self.pr_auc.update_state(labels, probs)
        self.auroc.update_state(labels, probs)
        return {
            "loss": self.loss_tracker.result(),
            "bce_dice": self.base_loss_tracker.result(),
            "pr_auc": self.pr_auc.result(),
            "auroc": self.auroc.result(),
        }


def exact_metrics(y, probs, threshold):
    p_curve, r_curve, _ = precision_recall_curve(y, probs)
    pred = (probs >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, probs)),
        "PRAUC": float(auc(r_curve, p_curve)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
    }


def f1_threshold(y, probs):
    candidates = np.unique(probs)
    scored = [(f1_score(y, probs >= t, zero_division=0), float(t)) for t in candidates]
    return min(((-score, threshold) for score, threshold in scored))[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="II5", choices=sorted(DATASET_PATHS))
    parser.add_argument("--feature-set", default="e3b", choices=["e3a", "e3b", "e3c"])
    parser.add_argument("--ranking-weight", type=float, default=0.1)
    args = parser.parse_args()
    dataset_path = DATASET_PATHS[args.dataset]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        BASE_DIR,
        "weights",
        f"{args.dataset}_E3_pairwise_{args.feature_set}_{timestamp}",
    )
    predictions_dir = os.path.join(run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=False)

    negative, positive = load_data(dataset_path)
    negative = np.asarray(negative, dtype=object)
    positive = np.asarray(positive, dtype=object)
    train_pool_neg, test_neg = train_test_split(negative, test_size=0.15, random_state=42)
    train_pool_pos, test_pos = train_test_split(positive, test_size=0.15, random_state=42)

    test_raw = np.vstack([test_neg, test_pos])
    np.random.seed(42)
    np.random.shuffle(test_raw)
    test_frame = pd.DataFrame(test_raw[:, :3])
    test_seq = encoding(test_frame).reshape((-1, 1, 24, 7)).astype("float32")
    test_bio_raw = extract_features_batch(test_raw.tolist(), feature_set=args.feature_set)
    test_y = np.asarray([int(float(r[2]) > 0) for r in test_raw], dtype="int32")

    train_pool = np.vstack([train_pool_neg, train_pool_pos])
    train_pool_y = np.asarray([0] * len(train_pool_neg) + [1] * len(train_pool_pos), dtype="int32")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train_pool, train_pool_y))
    results = []
    config = {
        "dataset": args.dataset,
        "feature_set": args.feature_set,
        "protocol": "E3 fixed 15% test and 5-fold stratified CV on remaining 85%",
        "architecture": "Original E3 BioFusion; no added feature branch",
        "objective": "BCE-Dice + ranking_weight * pairwise_softplus_same_sgRNA",
        "ranking_weight": args.ranking_weight,
        "batch_composition": "64 positive + 64 nearest same-guide negative pairs + 128 normal negatives",
        "pair_negative_policy": "Sample among the 32 nearest same-guide training negatives by symmetric difference of position/base-substitution signatures",
        "epochs": 35,
        "steps_per_epoch": 120,
        "checkpoint_rule": "minimum validation BCE-Dice loss, matching E3 checkpoint rule",
        "test_used_for_selection": False,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        seed = 42 + (fold - 1) * 100
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        train_rows, val_rows = train_pool[train_idx], train_pool[val_idx]
        train_y, val_y = train_pool_y[train_idx], train_pool_y[val_idx]

        train_seq = encoding(pd.DataFrame(train_rows[:, :3])).reshape((-1, 1, 24, 7)).astype("float32")
        val_seq = encoding(pd.DataFrame(val_rows[:, :3])).reshape((-1, 1, 24, 7)).astype("float32")
        train_bio_raw = extract_features_batch(train_rows.tolist(), feature_set=args.feature_set)
        val_bio_raw = extract_features_batch(val_rows.tolist(), feature_set=args.feature_set)
        scaler = StandardScaler().fit(train_bio_raw)
        train_bio = scaler.transform(train_bio_raw).astype("float32")
        val_bio = scaler.transform(val_bio_raw).astype("float32")
        fold_test_bio = scaler.transform(test_bio_raw).astype("float32")

        base_model = Crispr_SGRU_BioFusion(
            bio_feature_dim=train_bio.shape[1],
            learning_rate=0.0001,
        )
        model = PairwiseTrainer(base_model, ranking_weight=args.ranking_weight)
        model.compile(optimizer=base_model.optimizer, run_eagerly=False)
        model((train_seq[:1], train_bio[:1]), training=False)
        checkpoint = os.path.join(run_dir, f"fold{fold}.weights.h5")
        generator = SameGuidePairGenerator(
            train_seq,
            train_bio,
            train_rows,
            train_y,
            steps=120,
            seed=seed,
        )
        history = model.fit(
            generator.flow(),
            validation_data=((val_seq, val_bio), val_y.astype("float32").reshape(-1, 1)),
            epochs=35,
            steps_per_epoch=120,
            callbacks=[
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
                ModelCheckpoint(
                    checkpoint,
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=0,
                ),
            ],
            verbose=1,
        )
        model.load_weights(checkpoint)
        val_probs = base_model.predict((val_seq, val_bio), verbose=0).ravel()
        test_probs = base_model.predict((test_seq, fold_test_bio), verbose=0).ravel()
        threshold = f1_threshold(val_y, val_probs)
        for split, rows, y, probs in (
            ("val", val_rows, val_y, val_probs),
            ("test", test_raw, test_y, test_probs),
        ):
            metrics = exact_metrics(y, probs, threshold)
            results.append({"Fold": fold, "Split": split, "Threshold": threshold, **metrics})
            pd.DataFrame(
                {
                    "source_id": [int(r[3]) for r in rows],
                    "y_true": y,
                    "y_prob": probs,
                }
            ).to_csv(os.path.join(predictions_dir, f"fold{fold}_{split}.csv"), index=False)
        config.setdefault("folds", {})[str(fold)] = {
            "seed": seed,
            "best_epoch_by_val_loss": int(np.argmin(history.history["val_loss"])) + 1,
            "same_guide_positive_count": len(generator.pos_ids),
            "train_positive_count": int(np.sum(train_y == 1)),
            "val_positive_count": int(np.sum(val_y == 1)),
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        tf.keras.backend.clear_session()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(run_dir, "per_fold_results.csv"), index=False)
    summary = []
    for split in ("val", "test"):
        part = results_df[results_df["Split"] == split]
        numeric = part.select_dtypes(include="number")
        summary.append({"Split": split, "Statistic": "Mean", **numeric.mean().to_dict()})
        summary.append({"Split": split, "Statistic": "Std", **numeric.std(ddof=0).to_dict()})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(run_dir, "summary_results.csv"), index=False)
    print(results_df[results_df["Split"] == "val"].to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
