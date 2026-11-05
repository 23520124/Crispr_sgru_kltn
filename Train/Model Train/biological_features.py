"""Unified biological features for the E3 BioFusion experiment.

The extractor keeps the aligned sequence representation intact and exposes a
single fixed feature set. It does not perform alignment or silently rewrite
gap symbols; preprocessing and Encoder_sgRNA_off remain responsible for that.
"""

from typing import List

import numpy as np


FEATURE_SET_NAME = "enhanced"


class BiologicalFeatureExtractor:
    """Extract a fixed vector of mismatch, PAM, GC and indel features."""

    FEATURE_SETS = {"e3a", "e3b", "e3c", "enhanced", "unified"}

    def __init__(self, feature_set: str = FEATURE_SET_NAME):
        if feature_set not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature_set={feature_set!r}; use 'enhanced'")
        # Legacy names remain accepted for existing callers, but all resolve
        # to this single unified feature definition.
        self.feature_set = FEATURE_SET_NAME

    @staticmethod
    def _clean(seq) -> str:
        return str(seq).strip().upper()

    @staticmethod
    def _transition(a: str, b: str) -> bool:
        return {a, b} in ({"A", "G"}, {"C", "T"})

    @staticmethod
    def _transversion(a: str, b: str) -> bool:
        return a in "ACGT" and b in "ACGT" and a != b and not BiologicalFeatureExtractor._transition(a, b)

    def _aligned_positions(self, sgrna: str, offtarget: str):
        # PAM is terminal in the repository's aligned 23/24 nt sequences.
        # Anchor the protospacer alignment to it so an inserted alignment
        # column does not shift PAM-proximal mismatches out of the feature set.
        sg_proto = sgrna[:-3] if len(sgrna) >= 3 else ""
        off_proto = offtarget[:-3] if len(offtarget) >= 3 else ""
        width = max(len(sg_proto), len(off_proto))
        if width > 21:
            raise ValueError("Expected at most 21 aligned protospacer columns before NGG PAM")
        sg_proto = "-" * (width - len(sg_proto)) + sg_proto
        off_proto = "-" * (width - len(off_proto)) + off_proto
        pairs = list(zip(sg_proto, off_proto))
        return [("-", "-")] * (21 - width) + pairs

    def _mismatch_features(self, sgrna: str, offtarget: str) -> List[float]:
        pairs = self._aligned_positions(sgrna, offtarget)
        mask = [float(a != b) for a, b in pairs]
        transitions = sum(
            self._transition(a, b) for (a, b), mismatch in zip(pairs, mask) if mismatch
        )
        transversions = sum(
            self._transversion(a, b) for (a, b), mismatch in zip(pairs, mask) if mismatch
        )
        max_run = 0
        current_run = 0
        for mismatch in mask:
            if mismatch:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        regional = [
            sum(mask[1:7]), sum(mask[7:12]), sum(mask[12:17]), sum(mask[17:21])
        ]
        return [
            sum(mask), *regional, sum(mask[1:4]), sum(mask[17:20]),
            float(transitions), float(transversions), float(max_run), *mask,
        ]

    def _pam_features(self, sgrna: str, offtarget: str) -> List[float]:
        # Section 4.1 specifies NGG PAMs for all datasets, so only the variable
        # N base is represented; the terminal GG bases carry no variation.
        source = offtarget if len(offtarget) >= 3 else sgrna
        nt = source[-3] if len(source) >= 3 else "N"
        return [float(nt == base) for base in "ACGT"]

    @staticmethod
    def _gc(seq: str, terminal_pam: bool = False) -> float:
        if terminal_pam:
            protospacer = seq[:-3] if len(seq) >= 3 else seq
        else:
            # Preserve the original e3a/e3b/e3c feature definitions.
            protospacer = seq[:20]
        valid = [base for base in protospacer if base in "ACGT"]
        return (sum(base in "GC" for base in valid) / len(valid)) if valid else 0.0

    def _gc_features(
        self, sgrna: str, offtarget: str, terminal_pam: bool = False
    ) -> List[float]:
        gc_sg = self._gc(sgrna, terminal_pam=terminal_pam)
        gc_off = self._gc(offtarget, terminal_pam=terminal_pam)
        return [gc_sg, gc_off, abs(gc_sg - gc_off)]

    @staticmethod
    def _gap_positions(seq: str):
        return [i for i, char in enumerate(seq) if char in "_-"]

    def _indel_features(self, sgrna: str, offtarget: str) -> List[float]:
        sg_gaps = self._gap_positions(sgrna)
        off_gaps = self._gap_positions(offtarget)
        all_gaps = sg_gaps + off_gaps
        gap_count = len(all_gaps)
        gap_distances = [
            len(seq) - 4 - pos
            for seq in (sgrna, offtarget)
            for pos in self._gap_positions(seq)
            if pos < len(seq) - 3
        ]
        gap_near_pam = any(distance < 6 for distance in gap_distances)
        nearest_distance = min(
            gap_distances,
            default=max(len(sgrna), len(offtarget)) - 3,
        )
        return [
            float(bool(all_gaps)), float(gap_count), float(len(sg_gaps)),
            float(len(off_gaps)), float(gap_near_pam),
            float(nearest_distance) / max(float(max(len(sgrna), len(offtarget)) - 3), 1.0),
        ]

    def extract(self, sgrna: str, offtarget: str) -> np.ndarray:
        sgrna = self._clean(sgrna)
        offtarget = self._clean(offtarget)
        features = (
            self._mismatch_features(sgrna, offtarget)
            + self._pam_features(sgrna, offtarget)
            + self._gc_features(sgrna, offtarget, terminal_pam=True)
            + self._indel_features(sgrna, offtarget)
        )
        return np.asarray(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        names = [
            "total_mismatch", "mm_distal", "mm_middle", "mm_proximal",
            "mm_seed", "mm_first3", "mm_cut17_19", "transition_count",
            "transversion_count", "max_consecutive_mismatch",
        ]
        names.extend(f"mm_mask_{i}" for i in range(21))
        names.extend(f"pam_variable_N_{base}" for base in "ACGT")
        names.extend(["gc_sgrna", "gc_offtarget", "gc_diff"])
        names.extend([
            "has_indel", "indel_count", "sgrna_indel_count",
            "offtarget_indel_count", "indel_near_pam",
            "nearest_indel_distance_norm",
        ])
        return names

    def get_feature_dim(self) -> int:
        return len(self.get_feature_names())


def extract_features_batch(data_tuples: list, feature_set: str = FEATURE_SET_NAME) -> np.ndarray:
    extractor = BiologicalFeatureExtractor(feature_set=feature_set)
    features = np.asarray(
        [extractor.extract(row[0], row[1]) for row in data_tuples], dtype=np.float32
    )
    if features.size == 0:
        return np.empty((0, extractor.get_feature_dim()), dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != extractor.get_feature_dim():
        raise ValueError(
            f"Feature dimension mismatch: got {features.shape}, expected {extractor.get_feature_dim()}"
        )
    if not np.isfinite(features).all():
        raise ValueError("NaN or Inf detected in biological features")
    return features


def extract_position_features(sgrna: str, offtarget: str) -> np.ndarray:
    """Return (24, 21): 16 ordered base-pair channels plus 5 biology channels."""
    sgrna = BiologicalFeatureExtractor._clean(sgrna)
    offtarget = BiologicalFeatureExtractor._clean(offtarget)
    if len(sgrna) > 24 or len(offtarget) > 24:
        raise ValueError("Position-wise features require sequences of at most 24 nt")
    sgrna = "-" * (24 - len(sgrna)) + sgrna
    offtarget = "-" * (24 - len(offtarget)) + offtarget
    result = np.zeros((24, 21), dtype=np.float32)
    base_index = {base: i for i, base in enumerate("ATGC")}
    for pos, (a, b) in enumerate(zip(sgrna, offtarget)):
        if a in base_index and b in base_index:
            result[pos, base_index[a] * 4 + base_index[b]] = 1.0
        # Positions 21-23 are the terminal PAM, not protospacer mismatch
        # positions. Keep the interaction channels restricted to the aligned
        # protospacer, with the Cas9 cut-site marker anchored immediately
        # before the three-base PAM.
        in_protospacer = pos < 21
        mismatch = in_protospacer and a != b
        result[pos, 16:] = [
            float(mismatch),
            float(mismatch and BiologicalFeatureExtractor._transition(a, b)),
            float(mismatch and BiologicalFeatureExtractor._transversion(a, b)),
            float(in_protospacer and ("_" in (a, b) or ((a == "-") != (b == "-")))),
            float(17 <= pos <= 19),
        ]
    return result


def extract_position_features_batch(data_tuples: list) -> np.ndarray:
    result = np.asarray(
        [extract_position_features(row[0], row[1]) for row in data_tuples],
        dtype=np.float32,
    )
    if result.size == 0:
        return np.empty((0, 24, 21), dtype=np.float32)
    if result.shape[1:] != (24, 21) or not np.isfinite(result).all():
        raise ValueError(f"Invalid position-wise feature matrix: {result.shape}")
    return result


if __name__ == "__main__":
    extractor = BiologicalFeatureExtractor("enhanced")
    values = extractor.extract(
        "GCTGAAGCACTGCACGCCGTAGG", "GCTGAAGCACTGCACGCCATAGG"
    )
    assert values.shape == (extractor.get_feature_dim(),)
    assert np.isfinite(values).all()
    print(f"[OK] Unified feature extractor: {values.shape[0]} features")
