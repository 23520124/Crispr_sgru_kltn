"""Validate and lightly normalize Crispr-SGRU CSV datasets.

The output remains compatible with the existing loaders: three headerless
columns containing sgRNA, off-target, and a binary label. Original files are
never modified.

Preserve duplicate observations. For a normalized (sgRNA, off-target) pair
with both labels, keep every positive row and remove only its negative rows.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Datasets" / "preprocessed"
DATASET_PATHS = {
    "K562": ROOT / "Datasets" / "K562.csv",
    "HEK293t": ROOT / "Datasets" / "HEK293t.csv",
    "II5": ROOT / "Datasets" / "II5.csv",
    "II6": ROOT / "Datasets" / "II6.csv",
    "I1": ROOT / "Datasets" / "I1.csv",
    "I2": ROOT / "Datasets" / "I2.csv",
    "BE3": ROOT / "Datasets" / "BE3.csv",
    "CHANGE-seq": ROOT / "Datasets" / "CHANGE-seq" / "CHANGE-seq.csv",
}

# Structural validation of the current repository files shows that these two
# files store the guide in raw column 1 and the off-target in raw column 0.
SWAP_DATASETS = {"I2", "CHANGE-seq"}
ALLOWED_GUIDE_SYMBOLS = set("ACGTN_-")
ALLOWED_OFFTARGET_SYMBOLS = set("ACGT_-")


def normalize_sequence(value):
    return str(value).strip().upper()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def orientation_for(dataset, requested):
    if requested == "raw":
        return "raw_column_0_sgrna_column_1_offtarget"
    if requested == "swap":
        return "raw_column_1_sgrna_column_0_offtarget"
    if requested == "manifest":
        return (
            "raw_column_1_sgrna_column_0_offtarget"
            if dataset in SWAP_DATASETS
            else "raw_column_0_sgrna_column_1_offtarget"
        )
    raise ValueError(f"Unsupported orientation: {requested}")


def read_rows(path, dataset, requested_orientation):
    orientation = orientation_for(dataset, requested_orientation)
    swap = orientation.startswith("raw_column_1")
    rows = []
    skipped = Counter()
    length_counts = Counter()

    with path.open("r", newline="", encoding="utf-8") as handle:
        for source_row_id, raw in enumerate(csv.reader(handle)):
            if not raw or all(not str(item).strip() for item in raw):
                skipped["blank"] += 1
                continue
            if len(raw) < 3:
                skipped["too_few_columns"] += 1
                continue
            try:
                raw_label = float(raw[2].strip())
                if not math.isfinite(raw_label):
                    raise ValueError("non-finite label")
                label = 0 if raw_label == 0 else 1
            except (TypeError, ValueError):
                skipped["invalid_label_or_header"] += 1
                continue

            if swap:
                sgrna = normalize_sequence(raw[1])
                offtarget = normalize_sequence(raw[0])
            else:
                sgrna = normalize_sequence(raw[0])
                offtarget = normalize_sequence(raw[1])

            if not sgrna or not offtarget:
                skipped["empty_sequence"] += 1
                continue
            if len(sgrna) not in (23, 24) or len(offtarget) not in (23, 24):
                skipped["unsupported_sequence_length"] += 1
                continue
            if (
                set(sgrna) - ALLOWED_GUIDE_SYMBOLS
                or set(offtarget) - ALLOWED_OFFTARGET_SYMBOLS
            ):
                skipped["invalid_sequence_symbol"] += 1
                continue

            length_counts[f"{len(sgrna)}/{len(offtarget)}"] += 1
            rows.append(
                {
                    "sgrna": sgrna,
                    "offtarget": offtarget,
                    "label": label,
                    "source_row_id": source_row_id,
                }
            )

    return rows, orientation, skipped, length_counts


def inspect_duplicates(rows):
    grouped = {}
    for row in rows:
        pair = (row["sgrna"], row["offtarget"])
        state = grouped.get(pair)
        if state is None:
            grouped[pair] = [1, row["label"], row["source_row_id"], False]
            continue
        state[0] += 1
        if row["label"] != state[1]:
            state[3] = True

    conflicts = []
    conflict_pairs = {pair for pair, state in grouped.items() if state[3]}
    conflict_ids = defaultdict(list)
    for row in rows:
        pair = (row["sgrna"], row["offtarget"])
        if pair in conflict_pairs:
            conflict_ids[pair].append(row["source_row_id"])

    same_label_duplicate_pairs = 0
    duplicate_rows = 0
    for pair, (count, first_label, first_row_id, has_conflict) in grouped.items():
        if has_conflict:
            conflicts.append(
                {
                    "pair": list(pair),
                    "source_row_ids": conflict_ids[pair],
                    "labels": [0, 1],
                }
            )
            duplicate_rows += count - 1
        elif count > 1:
            same_label_duplicate_pairs += 1
            duplicate_rows += count - 1

    report = {
        "input_rows": len(rows),
        "output_rows": len(rows),
        "same_label_duplicate_pairs": same_label_duplicate_pairs,
        "duplicate_rows": duplicate_rows,
        "conflict_pair_count": len(conflicts),
        "conflict_rows": sum(
            len(item["source_row_ids"]) for item in conflicts
        ),
        "conflicts": conflicts,
    }
    return report


def schema_stats(rows, guide_key="sgrna", offtarget_key="offtarget"):
    positive_guides = {row[guide_key] for row in rows if row["label"] == 1}
    negative_guides = {row[guide_key] for row in rows if row["label"] == 0}
    guide_suffix_eligible = 0
    guide_suffix_gg = 0
    offtarget_suffix_eligible = 0
    offtarget_suffix_gg = 0
    guide_gap_rows = 0
    offtarget_gap_rows = 0
    guide_gap_positions = Counter()
    offtarget_gap_positions = Counter()
    for row in rows:
        guide = row[guide_key]
        offtarget = row[offtarget_key]
        if len(guide) >= 2:
            guide_suffix_eligible += 1
            guide_suffix_gg += guide[-2:] == "GG"
        if len(offtarget) >= 2:
            offtarget_suffix_eligible += 1
            offtarget_suffix_gg += offtarget[-2:] == "GG"
        guide_gaps = [i for i, char in enumerate(guide) if char in "_-"]
        offtarget_gaps = [i for i, char in enumerate(offtarget) if char in "_-"]
        if guide_gaps:
            guide_gap_rows += 1
            for position in guide_gaps:
                guide_gap_positions[str(position)] += 1
        if offtarget_gaps:
            offtarget_gap_rows += 1
            for position in offtarget_gaps:
                offtarget_gap_positions[str(position)] += 1
    return {
        "positive_unique_guides": len(positive_guides),
        "negative_unique_guides": len(negative_guides),
        "positive_guides_with_negative": len(positive_guides & negative_guides),
        "positive_guides_without_negative": len(positive_guides - negative_guides),
        "guide_suffix_GG_fraction": (
            guide_suffix_gg / guide_suffix_eligible
            if guide_suffix_eligible
            else 0.0
        ),
        "offtarget_suffix_GG_fraction": (
            offtarget_suffix_gg / offtarget_suffix_eligible
            if offtarget_suffix_eligible
            else 0.0
        ),
        "guide_gap_rows": guide_gap_rows,
        "offtarget_gap_rows": offtarget_gap_rows,
        "guide_gap_position_counts_0_based": dict(sorted(guide_gap_positions.items())),
        "offtarget_gap_position_counts_0_based": dict(
            sorted(offtarget_gap_positions.items())
        ),
    }


def orientation_evidence(rows, selected_orientation):
    selected_is_swapped = selected_orientation.startswith("raw_column_1")
    if selected_is_swapped:
        raw_stats = schema_stats(rows, guide_key="offtarget", offtarget_key="sgrna")
        swapped_stats = schema_stats(rows)
    else:
        raw_stats = schema_stats(rows)
        swapped_stats = schema_stats(
            rows, guide_key="offtarget", offtarget_key="sgrna"
        )
    return {
        "raw_column_0_as_sgrna": raw_stats,
        "raw_column_1_as_sgrna": swapped_stats,
        "selected_orientation": selected_orientation,
        "basis": "positive-guide overlap with negative guides and guide suffix-GG fraction; structural evidence, not source documentation",
    }


def resolve_label_conflicts(rows, duplicate_report):
    """Drop negative rows of mixed-label pairs; preserve duplicates and order."""
    conflict_pairs = {
        tuple(item["pair"]) for item in duplicate_report["conflicts"]
    }
    output = [
        row for row in rows
        if row["label"] == 1
        or (row["sgrna"], row["offtarget"]) not in conflict_pairs
    ]
    report = dict(duplicate_report)
    report["output_rows"] = len(output)
    report["duplicate_rows_removed"] = 0
    report["conflict_rows_removed"] = len(rows) - len(output)
    report["conflict_policy"] = "keep_positive_rows"
    report["duplicate_policy"] = "preserve_all"
    report["deduplication_applied"] = False
    report["conflict_resolution_applied"] = True
    return output, report


def write_dataset(
    dataset, rows, report, orientation, skipped, lengths, output_root, source_path=None
):
    output_root.mkdir(parents=True, exist_ok=True)
    output_csv = output_root / f"{dataset}.csv"
    output_json = output_root / f"{dataset}.json"
    source_path = source_path or DATASET_PATHS[dataset]

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in rows:
            writer.writerow([row["sgrna"], row["offtarget"], row["label"]])

    metadata = {
        "dataset": dataset,
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_file": str(output_csv),
        "output_sha256": sha256_file(output_csv),
        "sequence_column_order": orientation,
        "normalization": {
            "strip_and_uppercase": True,
            "allowed_symbols": {
                "sgRNA": "ACGTN_-",
                "off_target": "ACGT_-",
            },
            "underscore_and_hyphen_kept_distinct": True,
            "binary_label_rule": "1 if raw label != 0 else 0 (matches Train.py class split)",
            "padding": "deferred to Encoder_sgRNA_off (23 nt -> left-pad to 24)",
        },
        "rows_skipped": dict(skipped),
        "source_row_id_policy": "output rows are reindexed by training loaders; source IDs are recorded only for label conflicts",
        "input_sequence_lengths": dict(lengths),
        "schema_stats": schema_stats(rows),
        "orientation_evidence": orientation_evidence(rows, orientation),
        "deduplication": report,
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_csv, output_json


def process_dataset(dataset, output_root, orientation):
    source = DATASET_PATHS[dataset]
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    rows, resolved_orientation, skipped, lengths = read_rows(
        source, dataset, orientation
    )
    duplicate_report = inspect_duplicates(rows)
    cleaned, report = resolve_label_conflicts(rows, duplicate_report)

    output_csv, output_json = write_dataset(
        dataset,
        cleaned,
        report,
        resolved_orientation,
        skipped,
        lengths,
        output_root,
        source_path=source,
    )
    print(
        f"[{dataset}] {len(rows)} valid rows -> {len(cleaned)} output rows; "
        f"negative conflict rows removed={report['conflict_rows_removed']}; "
        f"orientation={resolved_orientation}; output={output_csv}"
    )
    return output_csv, output_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["all", *sorted(DATASET_PATHS)], default="all"
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT,
        help="Directory for normalized CSV files and JSON audits.",
    )
    parser.add_argument(
        "--orientation", choices=["manifest", "raw", "swap"], default="manifest",
        help="Use repository schema mapping, or force raw/swap column order.",
    )
    args = parser.parse_args()

    datasets = sorted(DATASET_PATHS) if args.dataset == "all" else [args.dataset]
    failures = []
    for dataset in datasets:
        try:
            process_dataset(
                dataset,
                args.output_root,
                args.orientation,
            )
        except Exception as exc:
            failures.append(f"{dataset}: {exc}")
            print(f"[FAIL] {failures[-1]}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
