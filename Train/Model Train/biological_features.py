"""
Biological Feature Extraction for E3

Extract biological features from sgRNA/off-target pairs for fusion with P12.
Features không phụ thuộc label và có fixed dimension.

Feature groups:
- Mismatch: count, binary mask, regional counts
- PAM: 3-nt one-hot encoding
- Indel: length difference, has_indel flag
- Sequence: GC content, GC difference

Alignment convention:
- Mismatch datasets (same length): compare first 20 nucleotides
- Indel datasets (I1/I2): length difference, heuristic gap detection
- PAM: positions 20-22 if available (23+ chars)
"""

import numpy as np
from typing import Tuple, Dict, List


class BiologicalFeatureExtractor:
    """
    Extract biological features from sgRNA and off-target sequences.

    Feature dimensions:
    - E3a (mismatch only): 25 features
    - E3b (mismatch + PAM + GC): 25 + 12 + 3 = 40 features
    - E3c (full with indel): 25 + 12 + 3 + 4 = 44 features
    """

    def __init__(self, feature_set: str = 'e3c'):
        """
        Args:
            feature_set: 'e3a' (mismatch), 'e3b' (mismatch+PAM+GC), 'e3c' (full)
        """
        assert feature_set in ['e3a', 'e3b', 'e3c'], f"Unknown feature_set: {feature_set}"
        self.feature_set = feature_set

    def extract(self, sgrna: str, offtarget: str) -> np.ndarray:
        """
        Extract features from sgRNA and off-target pair.

        Args:
            sgrna: sgRNA sequence (23+ nt)
            offtarget: off-target sequence (23+ nt)

        Returns:
            feature vector (float32)
        """
        features = []

        # Always include mismatch features (E3a)
        features.extend(self._extract_mismatch_features(sgrna, offtarget))

        if self.feature_set in ['e3b', 'e3c']:
            # Add PAM + GC features
            features.extend(self._extract_pam_features(sgrna, offtarget))
            features.extend(self._extract_gc_features(sgrna, offtarget))

        if self.feature_set == 'e3c':
            # Add indel features
            features.extend(self._extract_indel_features(sgrna, offtarget))

        return np.array(features, dtype=np.float32)

    def _extract_mismatch_features(self, sgrna: str, offtarget: str) -> List[float]:
        """
        Extract mismatch features (25 features).

        Convention (verified from Encoder_sgRNA_off.py line 348: isPAM[-3:, :] = 1):
        - Sequences are 5' → 3', 23 nt total
        - Positions 0-19: protospacer (20 nt)
        - Positions 20-22: PAM (3 nt, right end)
        - Position 0: distal region start (5' end, left)
        - Position 19: seed/PAM-proximal (3' end before PAM, right)

        Regional nomenclature:
        - Positions 0-5: distal region
        - Positions 6-10: middle region
        - Positions 11-15: proximal region
        - Positions 16-19: seed/PAM-proximal region

        Returns:
            [mm_distal, mm_middle, mm_proximal, mm_seed, mask_20]
        """
        # Compare first 20 positions (0-based indexing)
        limit = min(20, len(sgrna), len(offtarget))

        # Binary mismatch mask (20 features)
        mask = [1.0 if sgrna[i] != offtarget[i] else 0.0 for i in range(limit)]
        # Pad if sequences < 20
        while len(mask) < 20:
            mask.append(0.0)

        # Regional counts
        mm_distal = sum(mask[0:6])      # Distal region (5' end, positions 0-5)
        mm_middle = sum(mask[6:11])     # Middle region (positions 6-10)
        mm_proximal = sum(mask[11:16])  # Proximal region (positions 11-15)
        mm_seed = sum(mask[16:20])      # Seed/PAM-proximal region (3' end, positions 16-19)

        # Total mismatch count (sum of all regions)
        total_mm = mm_distal + mm_middle + mm_proximal + mm_seed

        # 5 summary features + 20 position mask = 25 features
        features = [total_mm, mm_distal, mm_middle, mm_proximal, mm_seed] + mask
        return features

    def _extract_pam_features(self, sgrna: str, offtarget: str) -> List[float]:
        """
        Extract PAM features (12 features).

        PAM positions 20-22 (0-based), one-hot encode 4 nucleotides x 3 positions.

        Returns:
            [pam_pos20_A, pam_pos20_C, pam_pos20_G, pam_pos20_T,
             pam_pos21_A, pam_pos21_C, pam_pos21_G, pam_pos21_T,
             pam_pos22_A, pam_pos22_C, pam_pos22_G, pam_pos22_T]
        """
        nucleotides = ['A', 'C', 'G', 'T']
        features = []

        # Use off-target PAM (positions 20-22)
        for pos in range(20, 23):
            if pos < len(offtarget):
                nt = offtarget[pos].upper()
                one_hot = [1.0 if nt == n else 0.0 for n in nucleotides]
            else:
                # Missing PAM -> all zeros
                one_hot = [0.0, 0.0, 0.0, 0.0]
            features.extend(one_hot)

        return features

    def _extract_gc_features(self, sgrna: str, offtarget: str) -> List[float]:
        """
        Extract GC content features (3 features).

        Returns:
            [gc_sgrna, gc_offtarget, gc_diff]
        """
        def gc_content(seq: str) -> float:
            # Use first 20 nt for consistency
            limit = min(20, len(seq))
            seq_20 = seq[:limit].upper()
            if limit == 0:
                return 0.0
            gc_count = seq_20.count('G') + seq_20.count('C')
            return gc_count / limit

        gc_sg = gc_content(sgrna)
        gc_off = gc_content(offtarget)
        gc_diff = abs(gc_sg - gc_off)

        return [gc_sg, gc_off, gc_diff]

    def _extract_indel_features(self, sgrna: str, offtarget: str) -> List[float]:
        """
        Extract indel features (4 features).

        Heuristic for I1/I2 datasets. This is NOT a full alignment.

        Returns:
            [has_indel, len_diff_norm, sgrna_len_norm, offtarget_len_norm]
        """
        len_sg = len(sgrna)
        len_off = len(offtarget)

        # Binary flag: sequences have different lengths
        has_indel = 1.0 if len_sg != len_off else 0.0

        # Normalized length difference
        len_diff_norm = abs(len_sg - len_off) / 23.0

        # Normalized lengths (assuming typical 20-23 nt range)
        sgrna_len_norm = len_sg / 23.0
        offtarget_len_norm = len_off / 23.0

        return [has_indel, len_diff_norm, sgrna_len_norm, offtarget_len_norm]

    def get_feature_names(self) -> List[str]:
        """Return feature names for this configuration."""
        names = []

        # Mismatch features (25)
        names.append('total_mismatch')
        names.append('mm_distal')      # Positions 0-5
        names.append('mm_middle')      # Positions 6-10
        names.append('mm_proximal')    # Positions 11-15
        names.append('mm_seed')        # Positions 16-19 (seed/PAM-proximal)
        for i in range(20):
            names.append(f'mm_mask_{i}')

        if self.feature_set in ['e3b', 'e3c']:
            # PAM features (12)
            nucleotides = ['A', 'C', 'G', 'T']
            for pos in range(20, 23):
                for nt in nucleotides:
                    names.append(f'pam_pos{pos}_{nt}')

            # GC features (3)
            names.extend(['gc_sgrna', 'gc_offtarget', 'gc_diff'])

        if self.feature_set == 'e3c':
            # Indel features (4)
            names.extend(['has_indel', 'len_diff_norm', 'sgrna_len_norm', 'offtarget_len_norm'])

        return names

    def get_feature_dim(self) -> int:
        """Return feature dimension for this configuration."""
        if self.feature_set == 'e3a':
            return 25
        elif self.feature_set == 'e3b':
            return 40
        elif self.feature_set == 'e3c':
            return 44
        else:
            raise ValueError(f"Unknown feature_set: {self.feature_set}")


def extract_features_batch(data_tuples: list, feature_set: str = 'e3c') -> np.ndarray:
    """
    Extract features for a batch of (sgrna, offtarget, label) tuples.

    Args:
        data_tuples: list of (sgrna, offtarget, label) or (sgrna, offtarget, label, source_id)
        feature_set: 'e3a', 'e3b', or 'e3c'

    Returns:
        feature matrix (n_samples, feature_dim)
    """
    extractor = BiologicalFeatureExtractor(feature_set=feature_set)
    expected_dim = extractor.get_feature_dim()
    features = []

    for row in data_tuples:
        sgrna = row[0]
        offtarget = row[1]
        feat = extractor.extract(sgrna, offtarget)
        features.append(feat)

    feature_matrix = np.array(features, dtype=np.float32)

    # Validate shape
    assert feature_matrix.shape[1] == expected_dim, \
        f"Feature dimension mismatch: got {feature_matrix.shape[1]}, expected {expected_dim}"

    # Check for NaN/Inf
    if np.isnan(feature_matrix).any():
        raise ValueError(f"NaN detected in features")
    if np.isinf(feature_matrix).any():
        raise ValueError(f"Inf detected in features")

    return feature_matrix


if __name__ == "__main__":
    # Quick test
    print("Testing BiologicalFeatureExtractor...")

    # Example: mismatch-only data
    sgrna = "GCTGAAGCACTGCACGCCGTAGG"
    offtarget = "GCTGAAGCACTGCACGCCATAGG"  # 1 mismatch at pos 18

    for feature_set in ['e3a', 'e3b', 'e3c']:
        print(f"\n{feature_set.upper()}:")
        extractor = BiologicalFeatureExtractor(feature_set=feature_set)
        features = extractor.extract(sgrna, offtarget)
        names = extractor.get_feature_names()
        dim = extractor.get_feature_dim()

        print(f"  Feature dim: {dim}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Regional counts: {features[0:4]}")
        print(f"  First 5 features: {features[:5]}")

        assert len(features) == dim, f"Dimension mismatch: {len(features)} != {dim}"
        assert len(names) == dim, f"Name count mismatch: {len(names)} != {dim}"
        assert not np.isnan(features).any(), "NaN detected"
        assert not np.isinf(features).any(), "Inf detected"

    print("\n✓ BiologicalFeatureExtractor test passed")
