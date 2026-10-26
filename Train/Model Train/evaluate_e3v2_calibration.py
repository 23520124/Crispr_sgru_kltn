#!/usr/bin/env python3
"""E3v2-D: validation-only probability calibration and threshold stability.

This evaluator consumes one completed E3v2-C directory and a fixed checkpoint
strategy (val_loss or val_prauc) specified via --checkpoint-strategy.  It does
not train a model or select between checkpoint strategies.  It reconstructs the
C split and training-fold scaler, loads the specified checkpoint strategy, and
fits raw/Platt/isotonic calibration using validation predictions only.

Test predictions are produced only after the calibrator method and per-fold
thresholds have been fixed on validation data.
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
possible_roots = [
    CURRENT_DIR,
    os.path.abspath(os.path.join(CURRENT_DIR, "..")),
    os.path.abspath(os.path.join(CURRENT_DIR, "../..")),
    os.path.abspath(os.path.join(CURRENT_DIR, "../../..")),
]
BASE_DIR = next(
    (candidate for candidate in possible_roots if os.path.exists(os.path.join(candidate, "Datasets"))),
    os.path.abspath(os.path.join(CURRENT_DIR, "../..")),
)
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(BASE_DIR, "Train"))
sys.path.append(os.path.join(BASE_DIR, "Train", "Model Train"))

import Encoder_sgRNA_off  # noqa: E402
from biological_features import extract_features_batch  # noqa: E402
from MODEL_E3V2_Loss import Crispr_SGRU_BioFusion_Loss  # noqa: E402


RANDOM_SEED = 42
BATCH_SIZE = 256
N_FOLDS = 5
THRESHOLD_TOLERANCE = 1e-12
CALIBRATION_TIE_TOLERANCE = 1e-12
CALIBRATION_TIE_ORDER = ("raw", "platt", "isotonic")
CALIBRATION_METHODS = ("raw", "platt", "isotonic")

DATASET_FEATURE_MAP = {
    "II6": "e3a",
    "HEK293t": "e3c",
    "II5": "e3b",
    "BE3": "e3c",
    "K562": "e3a",
}

PAPER_REFERENCE = {
    "K562": {"auroc": 1.000, "prauc": 0.941},
    "HEK293t": {"auroc": 0.998, "prauc": 0.801},
    "BE3": {"auroc": 0.995, "prauc": 0.523},
    "II5": {"auroc": 0.998, "prauc": 0.521},
    "II6": {"auroc": 0.996, "prauc": 0.442},
}


def load_data(file_path):
    negative, positive = [], []
    with open(file_path, "r") as handle:
        for row_id, line in enumerate(handle):
            fields = line.strip().split(",")
            try:
                label = float(fields[2])
            except (IndexError, ValueError):
                continue
            row = (fields[0], fields[1], fields[2], row_id)
            (positive if label != 0.0 else negative).append(row)
    return negative, positive


def encode_dataframe(dataframe):
    encoded = []
    for _, row in dataframe.iterrows():
        encoder = Encoder_sgRNA_off.Encoder(
            on_seq=row["sgRNA"],
            off_seq=row["off"],
            with_category=True,
            label=row["label"],
        )
        encoded.append(encoder.on_off_code)
    return np.asarray(encoded).reshape((-1, 1, 24, 7)).astype("float32")


def count_mismatches(seq1, seq2):
    limit = min(20, len(seq1), len(seq2))
    return sum(a != b for a, b in zip(seq1[:limit], seq2[:limit]))


def split_hash(rows):
    return hashlib.md5(str(sorted(row[3] for row in rows)).encode()).hexdigest()[:16]


def file_hash(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path):
    return file_hash(path)


def identify_training_pools(train_df):
    positives = train_df[train_df["label"].astype(float) != 0.0].copy()
    negatives = train_df[train_df["label"].astype(float) == 0.0].copy()
    hard_mask = negatives.apply(
        lambda row: count_mismatches(row["sgRNA"], row["off"]) <= 3, axis=1
    )
    hard = negatives[hard_mask].copy()
    normal = negatives[~hard_mask].copy()
    if len(hard) == 0:
        # Match E3v2-C's explicit fallback.  The duplicate pool is relevant
        # to the scaler, so preserve it here rather than silently changing it.
        hard = negatives.copy()
        normal = negatives.copy()
    return positives, hard, normal


def prepare_fold_features(train_df, val_df, test_df, feature_set):
    train_pos, train_hard, train_normal = identify_training_pools(train_df)
    pool_frames = [train_pos, train_hard, train_normal]

    pool_features = []
    for frame in pool_frames:
        tuples = list(frame[["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None))
        pool_features.append(extract_features_batch(tuples, feature_set=feature_set))

    val_tuples = list(val_df[["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None))
    test_tuples = list(test_df[["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None))
    val_features = extract_features_batch(val_tuples, feature_set=feature_set)
    test_features = extract_features_batch(test_tuples, feature_set=feature_set)

    scaler_input = np.vstack(pool_features)
    scaler = StandardScaler().fit(scaler_input)
    val_scaled = scaler.transform(val_features).astype("float32")
    test_scaled = scaler.transform(test_features).astype("float32")
    return val_scaled, test_scaled, int(scaler_input.shape[1])


def threshold_curve(y_true, probabilities, tolerance=THRESHOLD_TOLERANCE):
    """Return all cumulative breakpoint candidates and the highest F1 tie."""
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        raise ValueError("threshold selection requires both validation classes")

    unique = np.unique(probabilities)
    candidates = [0.0]
    candidates.extend(float(value) for value in unique)
    max_probability = float(unique.max())
    all_negative_threshold = float(np.nextafter(max_probability, np.inf))
    candidates.append(all_negative_threshold)
    candidates = sorted(set(candidates))

    rows = []
    best_f1 = -np.inf
    best_threshold = candidates[0]
    for threshold in candidates:
        predicted = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        denominator = 2 * tp + fp + fn
        f1 = float(2 * tp / denominator) if denominator else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "F1": f1,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
            }
        )
        if f1 > best_f1 + tolerance or (
            abs(f1 - best_f1) <= tolerance and threshold > best_threshold
        ):
            best_f1 = f1
            best_threshold = float(threshold)
    return float(best_threshold), float(best_f1), pd.DataFrame(rows)


def safe_auc_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    if np.unique(y_true).size == 2:
        auroc = float(roc_auc_score(y_true, probabilities))
        prauc = float(average_precision_score(y_true, probabilities))
        mcc = float(matthews_corrcoef(y_true, predicted))
    else:
        auroc = float("nan")
        prauc = float("nan")
        mcc = float("nan")

    return {
        "AUROC": auroc,
        "PRAUC": prauc,
        "F1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "MCC": mcc,
        "Brier": float(brier_score_loss(y_true, probabilities)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def fit_calibrator(method, val_probabilities, val_labels, min_class_count):
    if method == "raw":
        return None, {"status": "ok", "reason": "identity calibration"}

    counts = np.bincount(np.asarray(val_labels, dtype=np.int32), minlength=2)
    if counts[0] < min_class_count or counts[1] < min_class_count:
        return None, {
            "status": "failed",
            "reason": f"class_count_below_minimum: negative={counts[0]}, positive={counts[1]}, minimum={min_class_count}",
        }

    try:
        x = np.asarray(val_probabilities, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(val_labels, dtype=np.int32)
        if method == "platt":
            calibrator = LogisticRegression(
                solver="lbfgs", C=1e6, max_iter=1000, random_state=RANDOM_SEED
            )
            calibrator.fit(x, y)
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(x.ravel(), y)
        else:
            raise ValueError(f"unknown calibration method: {method}")
        return calibrator, {"status": "ok", "reason": ""}
    except Exception as exc:  # record failure; never substitute another method
        return None, {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def apply_calibrator(method, calibrator, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if method == "raw":
        return probabilities.copy()
    if method == "platt":
        return np.asarray(calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1], dtype=np.float64)
    if method == "isotonic":
        return np.asarray(calibrator.predict(probabilities), dtype=np.float64)
    raise ValueError(f"unknown calibration method: {method}")


def resolve_checkpoint_dir(dataset, feature_set, requested):
    if requested:
        # Try as absolute path first
        path = os.path.abspath(requested)
        if os.path.isdir(path):
            return path
        # Try relative to BASE_DIR if starts with "weights"
        if requested.startswith("weights"):
            path = os.path.join(BASE_DIR, requested)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(f"checkpoint directory not found: {requested} (tried {os.path.abspath(requested)} and {os.path.join(BASE_DIR, requested) if requested.startswith('weights') else 'N/A'})")
    pattern = os.path.join(BASE_DIR, "weights", f"{dataset}_E3V2_checkpoint_{feature_set}_*")
    candidates = sorted(path for path in glob.glob(pattern) if os.path.isdir(path))
    if not candidates:
        raise FileNotFoundError(f"no E3v2-C directory matches {pattern}")
    return candidates[-1]


def metric_summary(rows, prefix):
    metrics = ("AUROC", "PRAUC", "F1", "Precision", "Recall", "MCC", "Brier", "TP", "FP", "FN", "TN")
    output = {}
    valid = [row for row in rows if row.get("status") == "ok" and row.get(f"{prefix}_F1") is not None]
    output[f"{prefix}_n_successful_folds"] = len(valid)
    for metric in metrics:
        values = [row.get(f"{prefix}_{metric}") for row in valid]
        values = [float(value) for value in values if np.isfinite(value)]
        output[f"{prefix}_{metric}_mean"] = float(np.mean(values)) if values else float("nan")
        output[f"{prefix}_{metric}_std"] = float(np.std(values, ddof=0)) if values else float("nan")
    return output


def evaluate_e3v2_calibration(dataset, feature_set=None, checkpoint_dir=None, checkpoint_strategy=None, min_class_count=2):
    expected_feature_set = DATASET_FEATURE_MAP.get(dataset)
    if expected_feature_set is None:
        raise ValueError(f"unknown dataset: {dataset}")
    if feature_set is not None and feature_set != expected_feature_set:
        raise ValueError(f"feature set for {dataset} is fixed to {expected_feature_set}")
    feature_set = expected_feature_set

    if checkpoint_strategy is None:
        raise ValueError("--checkpoint-strategy is required; must be 'val_loss' or 'val_prauc'")
    if checkpoint_strategy not in ("val_loss", "val_prauc"):
        raise ValueError(f"checkpoint_strategy must be 'val_loss' or 'val_prauc', got {checkpoint_strategy}")

    checkpoint_dir = resolve_checkpoint_dir(dataset, feature_set, checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as handle:
        checkpoint_config = json.load(handle)
    if checkpoint_config.get("dataset") != dataset or checkpoint_config.get("feature_set") != feature_set:
        raise ValueError("checkpoint config does not match requested dataset/feature set")
    loss_mode = checkpoint_config["loss_mode"]
    sampling_mode = checkpoint_config["sampling_mode"]

    data_path = os.path.join(BASE_DIR, "Datasets", f"{dataset}.csv")
    if checkpoint_config.get("dataset_file_hash") and file_hash(data_path) != checkpoint_config["dataset_file_hash"]:
        raise ValueError("dataset hash differs from E3v2-C; refusing to evaluate mismatched data")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "weights", f"{dataset}_E3V2_calibration_{feature_set}_{timestamp}")
    calibrators_dir = os.path.join(output_dir, "calibrators")
    predictions_dir = os.path.join(output_dir, "predictions")
    curves_dir = os.path.join(output_dir, "threshold_curves")
    histories_dir = os.path.join(output_dir, "histories")
    for path in (calibrators_dir, predictions_dir, curves_dir, histories_dir):
        os.makedirs(path, exist_ok=True)

    negative, positive = load_data(data_path)
    pos_train_val, pos_test = train_test_split(positive, test_size=0.15, random_state=RANDOM_SEED)
    neg_train_val, neg_test = train_test_split(negative, test_size=0.15, random_state=RANDOM_SEED)
    train_val_data = pos_train_val + neg_train_val
    test_data = pos_test + neg_test
    random.seed(RANDOM_SEED)
    random.shuffle(train_val_data)
    random.shuffle(test_data)

    test_df = pd.DataFrame(test_data, columns=["sgRNA", "off", "label", "source_id"])
    expected_test_hash = checkpoint_config.get("split_hashes", {}).get("test")
    if expected_test_hash and split_hash(test_data) != expected_test_hash:
        raise ValueError("test split hash differs from E3v2-C; refusing to evaluate mismatched split")
    test_enc = encode_dataframe(test_df)
    test_labels = test_df["label"].astype("float32").to_numpy()
    test_tuples = list(test_df[["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None))
    test_bio = extract_features_batch(test_tuples, feature_set=feature_set)

    train_val_df = pd.DataFrame(train_val_data, columns=["sgRNA", "off", "label", "source_id"])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    labels = train_val_df["label"].astype(float)

    fold_contexts = []
    checkpoint_strategy_predictions = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, labels), start=1):
        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()
        expected_train_hash = checkpoint_config.get("split_hashes", {}).get(f"fold_{fold}_train")
        expected_val_hash = checkpoint_config.get("split_hashes", {}).get(f"fold_{fold}_val")
        actual_train_hash = split_hash(list(train_df[["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None)))
        actual_val_hash = split_hash(list(val_df[["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None)))
        if expected_train_hash and actual_train_hash != expected_train_hash:
            raise ValueError(f"fold {fold} train split hash differs from E3v2-C")
        if expected_val_hash and actual_val_hash != expected_val_hash:
            raise ValueError(f"fold {fold} validation split hash differs from E3v2-C")
        val_enc = encode_dataframe(val_df)
        val_labels = val_df["label"].astype("float32").to_numpy()
        val_bio, test_bio_scaled, bio_dim = prepare_fold_features(train_df, val_df, test_df, feature_set)
        fold_contexts.append({
            "fold": fold,
            "train_df": train_df,
            "val_df": val_df,
            "val_enc": val_enc,
            "val_labels": val_labels,
            "val_bio": val_bio,
            "test_bio": test_bio_scaled,
            "bio_dim": bio_dim,
        })

        # Load only the fixed checkpoint strategy specified by user
        weight_path = os.path.join(checkpoint_dir, "checkpoints", checkpoint_strategy, f"fold{fold}_best.weights.h5")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"missing checkpoint: {weight_path}")
        random.seed(RANDOM_SEED + fold)
        np.random.seed(RANDOM_SEED + fold)
        tf.random.set_seed(RANDOM_SEED + fold)
        model = Crispr_SGRU_BioFusion_Loss(
            bio_feature_dim=bio_dim, loss_mode=loss_mode, learning_rate=0.0001
        )
        model.load_weights(weight_path)
        val_raw = model.predict([val_enc, val_bio], verbose=0).ravel().astype(np.float64)
        checkpoint_strategy_predictions[(fold, checkpoint_strategy)] = {"val_raw": val_raw, "model": model}

    calibration_records = []
    calibrator_objects = {}
    for context in fold_contexts:
        fold = context["fold"]
        selected = checkpoint_strategy_predictions[(fold, checkpoint_strategy)]
        val_raw = selected["val_raw"]
        val_labels = context["val_labels"]
        for method in CALIBRATION_METHODS:
            calibrator, status = fit_calibrator(method, val_raw, val_labels, min_class_count)
            metadata = {
                "fold": fold,
                "method": method,
                "status": status["status"],
                "reason": status["reason"],
                "fit_split": "validation_only",
                "checkpoint_strategy": checkpoint_strategy,
                "minimum_class_count": min_class_count,
            }
            metadata_path = os.path.join(calibrators_dir, f"fold{fold}_{method}.json")
            with open(metadata_path, "w") as handle:
                json.dump(metadata, handle, indent=2)
            if status["status"] != "ok":
                calibration_records.append({
                    "fold": fold,
                    "calibration_method": method,
                    "status": "failed",
                    "reason": status["reason"],
                    "val_metrics": None,
                    "calibrator": None,
                })
                continue

            if method != "raw":
                with open(os.path.join(calibrators_dir, f"fold{fold}_{method}.pkl"), "wb") as handle:
                    pickle.dump(calibrator, handle, protocol=pickle.HIGHEST_PROTOCOL)
            val_calibrated = apply_calibrator(method, calibrator, val_raw)
            threshold, _, curve = threshold_curve(val_labels, val_calibrated)
            val_metrics = safe_auc_metrics(val_labels, val_calibrated, threshold)
            curve.to_csv(os.path.join(curves_dir, f"fold{fold}_{method}.csv"), index=False)
            calibrator_objects[(fold, method)] = calibrator
            calibration_records.append({
                "fold": fold,
                "calibration_method": method,
                "status": "ok",
                "reason": "",
                "val_metrics": val_metrics,
                "threshold": threshold,
                "val_raw": val_raw,
                "val_calibrated": val_calibrated,
                "calibrator": calibrator,
            })

    method_means = {}
    for method in CALIBRATION_TIE_ORDER:
        valid = [row["val_metrics"]["F1"] for row in calibration_records if row["calibration_method"] == method and row["status"] == "ok"]
        method_means[method] = float(np.mean(valid)) if len(valid) == N_FOLDS else float("nan")
    eligible = [method for method in CALIBRATION_TIE_ORDER if np.isfinite(method_means[method])]
    if not eligible:
        raise RuntimeError("all calibration methods failed on validation")
    selected_method = eligible[0]
    for method in eligible[1:]:
        if method_means[method] > method_means[selected_method] + CALIBRATION_TIE_TOLERANCE:
            selected_method = method

    with open(os.path.join(histories_dir, "calibration_selection.json"), "w") as handle:
        json.dump(
            {
                "criterion": "mean_validation_F1",
                "tie_tolerance": CALIBRATION_TIE_TOLERANCE,
                "tie_order": list(CALIBRATION_TIE_ORDER),
                "mean_validation_F1": method_means,
                "selected_calibration_method": selected_method,
                "failed_methods": [
                    {"fold": row["fold"], "method": row["calibration_method"], "reason": row["reason"]}
                    for row in calibration_records if row["status"] != "ok"
                ],
            },
            handle,
            indent=2,
        )

    # Test inference starts only after checkpoint/calibration/threshold selection.
    result_rows = []
    for context in fold_contexts:
        fold = context["fold"]
        selected = checkpoint_strategy_predictions[(fold, checkpoint_strategy)]
        model = selected["model"]
        test_raw = model.predict([test_enc, context["test_bio"]], verbose=0).ravel().astype(np.float64)
        for row in [item for item in calibration_records if item["fold"] == fold]:
            method = row["calibration_method"]
            base = {
                "fold": fold,
                "checkpoint_strategy": checkpoint_strategy,
                "calibration_method": method,
                "status": row["status"],
                "reason": row["reason"],
                "threshold": row.get("threshold", np.nan),
                "selected_calibration_method": selected_method,
            }
            if row["status"] != "ok":
                result_rows.append(base)
                continue
            test_calibrated = apply_calibrator(method, row["calibrator"], test_raw)
            val_calibrated = row["val_calibrated"]
            test_metrics = safe_auc_metrics(test_labels, test_calibrated, row["threshold"])
            for split_name, metrics in (("val", row["val_metrics"]), ("test", test_metrics)):
                for metric, value in metrics.items():
                    base[f"{split_name}_{metric}"] = value
            base.update({
                "AUROC": test_metrics["AUROC"],
                "PRAUC": test_metrics["PRAUC"],
                "F1": test_metrics["F1"],
                "Precision": test_metrics["Precision"],
                "Recall": test_metrics["Recall"],
                "MCC": test_metrics["MCC"],
                "Brier": test_metrics["Brier"],
                "TP": test_metrics["TP"],
                "FP": test_metrics["FP"],
                "FN": test_metrics["FN"],
                "TN": test_metrics["TN"],
            })
            val_output = pd.DataFrame({
                "source_id": context["val_df"]["source_id"].values,
                "true_label": context["val_labels"],
                "raw_proba": row["val_raw"],
                "calibrated_proba": val_calibrated,
                "pred_binary": (val_calibrated >= row["threshold"]).astype(int),
            })
            test_output = pd.DataFrame({
                "source_id": test_df["source_id"].values,
                "true_label": test_labels,
                "raw_proba": test_raw,
                "calibrated_proba": test_calibrated,
                "pred_binary": (test_calibrated >= row["threshold"]).astype(int),
            })
            val_output.to_csv(os.path.join(predictions_dir, f"fold{fold}_{method}_val_predictions.csv"), index=False)
            test_output.to_csv(os.path.join(predictions_dir, f"fold{fold}_{method}_test_predictions.csv"), index=False)
            result_rows.append(base)

    per_fold = pd.DataFrame(result_rows)
    per_fold.to_csv(os.path.join(output_dir, "E3V2_calibration_per_fold.csv"), index=False)

    paired_rows = []
    for fold in range(1, N_FOLDS + 1):
        raw = per_fold[(per_fold.fold == fold) & (per_fold.calibration_method == "raw")].iloc[0]
        for method in ("platt", "isotonic"):
            calibrated = per_fold[(per_fold.fold == fold) & (per_fold.calibration_method == method)].iloc[0]
            paired = {"fold": fold, "calibration_method": method, "status": calibrated["status"], "reason": calibrated["reason"]}
            if calibrated["status"] == "ok":
                for metric in ("threshold", "AUROC", "PRAUC", "F1", "Precision", "Recall", "MCC", "Brier", "TP", "FP", "FN", "TN"):
                    paired[f"diff_{metric}"] = float(calibrated[metric]) - float(raw[metric])
            paired_rows.append(paired)
    pd.DataFrame(paired_rows).to_csv(os.path.join(output_dir, "E3V2_calibration_paired_differences.csv"), index=False)

    summary_rows = []
    paper = PAPER_REFERENCE.get(dataset, {})
    for method in CALIBRATION_METHODS:
        method_rows = [row for row in result_rows if row["calibration_method"] == method]
        successful_rows = [row for row in method_rows if row["status"] == "ok"]
        row = {
            "calibration_method": method,
            "status": "ok" if all(item["status"] == "ok" for item in method_rows) else "failed",
            "selected": method == selected_method,
            "checkpoint_strategy": checkpoint_strategy,
            "paper_reference": f"Zhang et al. IJMS 2024 S1 - {dataset}",
            "paper_auroc": paper.get("auroc", np.nan),
            "paper_prauc": paper.get("prauc", np.nan),
        }
        for prefix in ("val", "test"):
            summary = metric_summary(method_rows, prefix)
            row.update(summary)
        thresholds = [float(item["threshold"]) for item in successful_rows if np.isfinite(item["threshold"])]
        row["threshold_mean"] = float(np.mean(thresholds)) if thresholds else float("nan")
        row["threshold_std"] = float(np.std(thresholds, ddof=0)) if thresholds else float("nan")
        for metric in ("AUROC", "PRAUC", "F1", "Precision", "Recall", "MCC", "Brier", "TP", "FP", "FN", "TN"):
            row[f"{metric}_mean"] = row[f"test_{metric}_mean"]
            row[f"{metric}_std"] = row[f"test_{metric}_std"]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "E3V2_calibration_summary.csv"), index=False)

    evaluated_split_hashes = {"test": split_hash(test_data)}
    for context in fold_contexts:
        evaluated_split_hashes[f"fold_{context['fold']}_train"] = split_hash(
            list(context["train_df"][["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None))
        )
        evaluated_split_hashes[f"fold_{context['fold']}_val"] = split_hash(
            list(context["val_df"][["sgRNA", "off", "label", "source_id"]].itertuples(index=False, name=None))
        )

    config = {
        "experiment": "E3v2-D Calibration and Threshold Stability",
        "dataset": dataset,
        "feature_set": feature_set,
        "checkpoint_strategy": checkpoint_strategy,
        "source_checkpoint_dir": checkpoint_dir,
        "source_checkpoint_config": os.path.join(checkpoint_dir, "config.json"),
        "dataset_file": data_path,
        "dataset_file_hash": file_hash(data_path),
        "loss_mode": loss_mode,
        "sampling_mode": sampling_mode,
        "calibration_selection": {
            "criterion": "mean_validation_F1",
            "tie_tolerance": CALIBRATION_TIE_TOLERANCE,
            "tie_order": list(CALIBRATION_TIE_ORDER),
            "selected": selected_method,
            "means": method_means,
            "minimum_class_count": min_class_count,
        },
        "threshold_selection": {
            "method": "cumulative_probability_breakpoints",
            "tie_tolerance": THRESHOLD_TOLERANCE,
            "tie_break": "highest_threshold",
        },
        "calibration_methods": list(CALIBRATION_METHODS),
        "platt_parameters": {"solver": "lbfgs", "C": 1e6, "max_iter": 1000},
        "isotonic_parameters": {"out_of_bounds": "clip", "y_min": 0.0, "y_max": 1.0},
        "fit_split": "validation_only",
        "test_selection_allowed": False,
        "metric_methods": {
            "AUROC": "sklearn.metrics.roc_auc_score",
            "PRAUC": "sklearn.metrics.average_precision_score",
            "Brier": "sklearn.metrics.brier_score_loss",
        },
        "n_folds": N_FOLDS,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "paper_reference": paper,
        "split_hashes": evaluated_split_hashes,
        "checkpoint_hashes": {
            f"fold_{fold}_{checkpoint_strategy}": checkpoint_hash(
                os.path.join(checkpoint_dir, "checkpoints", checkpoint_strategy, f"fold{fold}_best.weights.h5")
            )
            for fold in range(1, N_FOLDS + 1)
        },
        "timestamp": timestamp,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"E3v2-D complete: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3v2-D calibration evaluation")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FEATURE_MAP))
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--checkpoint-dir", default=None, help="Completed E3v2-C directory; latest is used if omitted")
    parser.add_argument("--checkpoint-strategy", required=True, choices=["val_loss", "val_prauc"], help="Fixed checkpoint strategy from E3v2-C winner")
    parser.add_argument("--min-class-count", type=int, default=2)
    args = parser.parse_args()
    evaluate_e3v2_calibration(args.dataset, args.feature_set, args.checkpoint_dir, args.checkpoint_strategy, args.min_class_count)
