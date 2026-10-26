"""
MODEL_E5C_Encoding — Encoding Augmentation Study

E5-C experiment: isolate effect of augmented encoding with mismatch/indel/PAM masks.

Two modes:
1. control: Baseline 7-channel encoding (5 nucleotide + 2 direction)
   Input shape: (batch, 1, 24, 7)

2. mask_augmented: 10-channel encoding (7 base + 3 mask channels)
   Input shape: (batch, 1, 24, 10)
   Additional channels:
   - Channel 7: mismatch mask (1 if sgRNA[i] != off[i] and no gap, else 0)
   - Channel 8: indel mask (1 if '_' in sgRNA[i] or off[i], else 0)
   - Channel 9: PAM mask (1 for positions 20-22, else 0)

Single factor change: encoding representation ONLY
All other factors FIXED to E3v2 winner per dataset
"""

import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam


class BCEDiceLoss(tf.keras.losses.Loss):
    """BCE-Dice Loss (0.5 BCE + 0.5 Soft Dice)."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1e-7, from_logits=False, name="bce_dice_loss"):
        super(BCEDiceLoss, self).__init__(name=name)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)
        self.from_logits = from_logits
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        p = tf.math.sigmoid(y_pred) if self.from_logits else y_pred
        loss_bce = self.bce(y_true, y_pred if self.from_logits else p)
        y_true_f = tf.reshape(y_true, [-1])
        p_f = tf.reshape(p, [-1])
        intersection = tf.reduce_sum(y_true_f * p_f)
        denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(p_f) + self.eps
        loss_dice = 1.0 - (2.0 * intersection + self.eps) / denominator
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice

    def get_config(self):
        return {'bce_weight': self.bce_weight, 'dice_weight': self.dice_weight,
                'eps': self.eps, 'from_logits': self.from_logits}


class FocalBCELoss(tf.keras.losses.Loss):
    """Binary Focal Loss."""
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, name="focal_bce_loss"):
        super(FocalBCELoss, self).__init__(name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        p = tf.math.sigmoid(y_pred) if self.from_logits else y_pred
        p = tf.clip_by_value(p, 1e-7, 1.0 - 1e-7)
        p_t = tf.where(tf.equal(y_true, 1.0), p, 1.0 - p)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        bce = -(y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p))
        focal_loss = focal_weight * bce
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * focal_loss
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        return {'gamma': self.gamma, 'alpha': self.alpha, 'from_logits': self.from_logits}


def encode_with_masks(sgrna: str, offtarget: str, mode: str) -> np.ndarray:
    """
    Encode sgRNA/off-target pair with optional mask channels.

    Returns: (1, 24, 7) for control, (1, 24, 10) for mask_augmented
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from Encoder_sgRNA_off import Encoder

    encoder = Encoder(sgrna, offtarget)
    base_encoding = encoder.on_off_code  # (24, 7)

    if mode == 'control':
        return base_encoding.reshape(1, 24, 7)

    elif mode == 'mask_augmented':
        tlen = 24
        padded_sgrna = "-" * (tlen - len(sgrna)) + sgrna
        padded_off = "-" * (tlen - len(offtarget)) + offtarget

        mismatch_mask = np.zeros(24, dtype=np.float32)
        indel_mask = np.zeros(24, dtype=np.float32)
        pam_mask = np.zeros(24, dtype=np.float32)

        for i in range(24):
            sg_char = padded_sgrna[i]
            off_char = padded_off[i]

            # Mismatch: different nucleotides (not gaps/padding)
            if sg_char != '-' and off_char != '-' and sg_char != '_' and off_char != '_':
                if sg_char != off_char:
                    mismatch_mask[i] = 1.0

            # Indel: gap character present
            if sg_char == '_' or off_char == '_':
                indel_mask[i] = 1.0

            # PAM: positions 20-22 (0-indexed)
            if 20 <= i <= 22:
                pam_mask[i] = 1.0

        augmented = np.concatenate([
            base_encoding,
            mismatch_mask.reshape(24, 1),
            indel_mask.reshape(24, 1),
            pam_mask.reshape(24, 1)
        ], axis=1)

        return augmented.reshape(1, 24, 10)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_e5c_model(mode: str, stage: str, loss_mode: str, bio_feature_dim=44,
                    focal_gamma=2.0, focal_alpha=0.25, learning_rate=0.0001):
    """
    Build E5-C model with E3v2 winner architecture.

    Args:
        mode: 'control' (7ch) or 'mask_augmented' (10ch)
        stage: 'p12' or 'biofusion'
        loss_mode: 'control' (BCE-Dice) or 'focal_bce'
        bio_feature_dim: 25 (e3a), 40 (e3b), or 44 (e3c)
        focal_gamma: gamma for focal loss (2.0)
        focal_alpha: alpha for focal loss (0.25)
    """
    assert mode in ['control', 'mask_augmented']
    assert stage in ['p12', 'biofusion']
    assert loss_mode in ['control', 'focal_bce']

    input_channels = 7 if mode == 'control' else 10

    # Sequence branch
    sequence_input = Input(shape=(1, 24, input_channels), name='sequence_input')

    # Inception CNN (10 filters each)
    conv_1 = layers.Conv2D(10, (1, 1), padding='same', activation='relu', name='conv_1x1')(sequence_input)
    conv_2 = layers.Conv2D(10, (1, 2), padding='same', activation='relu', name='conv_1x2')(sequence_input)
    conv_3 = layers.Conv2D(10, (1, 3), padding='same', activation='relu', name='conv_1x3')(sequence_input)
    conv_4 = layers.Conv2D(10, (1, 5), padding='same', activation='relu', name='conv_1x5')(sequence_input)
    conv_output = layers.Concatenate(axis=-1, name='conv_concat')([conv_1, conv_2, conv_3, conv_4])
    conv_output = layers.Reshape((24, 40), name='conv_reshape')(conv_output)

    # BiGRU stack with skip connections
    x0 = layers.Bidirectional(layers.GRU(30, return_sequences=True), name='bigru_30')(conv_output)

    # Skip connection from input (use first 7 channels for both modes)
    if input_channels == 7:
        inputs_skip = layers.Reshape((24, 7), name='input_reshape')(sequence_input)
    else:
        # Extract first 7 channels for skip
        inputs_skip = layers.Lambda(lambda x: x[:, :, :, :7], name='input_slice')(sequence_input)
        inputs_skip = layers.Reshape((24, 7), name='input_reshape')(inputs_skip)

    x = layers.Concatenate(axis=2, name='skip_1')([inputs_skip, x0])

    x1 = layers.Bidirectional(layers.GRU(20, return_sequences=True), name='bigru_20')(x)
    x = layers.Concatenate(axis=2, name='skip_2')([x0, x1])

    x2 = layers.Bidirectional(layers.GRU(10, return_sequences=True), name='bigru_10')(x)
    x = layers.Concatenate(axis=-1, name='skip_3')([x0, x1, x2])

    # Dense head
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(128, activation="relu", name='dense_128')(x)
    seq_repr = layers.Dense(64, activation="relu", name='seq_repr')(x)
    seq_repr = layers.Dropout(rate=0.35, name='seq_dropout')(seq_repr)

    if stage == 'p12':
        probs = layers.Dense(1, activation='sigmoid', name='prob_output')(seq_repr)
        model = Model(inputs=sequence_input, outputs=probs, name=f'E5C_{mode}_p12')

    elif stage == 'biofusion':
        bio_input = Input(shape=(bio_feature_dim,), name='bio_input')
        bio_dense = layers.Dense(64, activation='relu', name='bio_dense')(bio_input)
        bio_norm = layers.LayerNormalization(name='bio_norm')(bio_dense)
        bio_repr = layers.Dropout(0.2, name='bio_dropout')(bio_norm)

        fusion = layers.Concatenate(name='fusion')([seq_repr, bio_repr])
        fusion_dense = layers.Dense(64, activation='relu', name='fusion_dense')(fusion)
        fusion_dense = layers.Dropout(0.3, name='fusion_dropout')(fusion_dense)

        probs = layers.Dense(1, activation='sigmoid', name='prob_output')(fusion_dense)
        model = Model(inputs=[sequence_input, bio_input], outputs=probs, name=f'E5C_{mode}_biofusion')

    # Loss function
    if loss_mode == 'control':
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=False)
    elif loss_mode == 'focal_bce':
        loss_fn = FocalBCELoss(gamma=focal_gamma, alpha=focal_alpha, from_logits=False)

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_fn, optimizer=opt,
                  metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                           tf.keras.metrics.AUC(curve='ROC', name='auroc')])
    return model


def get_shared_layer_hashes(model):
    """
    Compute hash for each shared layer (GRU, Dense, bio branch, output).

    Returns dict: {layer_name: hash_string}
    """
    shareable = ['bigru_30', 'bigru_20', 'bigru_10', 'dense_128', 'seq_repr',
                 'bio_dense', 'bio_norm', 'fusion_dense', 'prob_output']

    hashes = {}
    for name in shareable:
        try:
            layer = model.get_layer(name)
            weights = layer.get_weights()
            if weights:
                weight_bytes = b''.join([w.tobytes() for w in weights])
                hashes[name] = hashlib.md5(weight_bytes).hexdigest()
        except:
            pass

    return hashes


def copy_cnn_weights_fair(model_src_7ch, model_dst_10ch):
    """
    Fair CNN weight initialization: copy 7 base channels, zero 3 mask channels.

    Copies from 7-channel source to 10-channel destination.
    All other compatible layers (GRU, Dense, bio branch) are also copied.
    """
    conv_layers = ['conv_1x1', 'conv_1x2', 'conv_1x3', 'conv_1x5']

    for layer_name in conv_layers:
        src_layer = model_src_7ch.get_layer(layer_name)
        dst_layer = model_dst_10ch.get_layer(layer_name)

        src_kernel, src_bias = src_layer.get_weights()
        # src_kernel: (1, K, 7, 10), dst needs (1, K, 10, 10)

        dst_kernel_shape = dst_layer.kernel.shape
        dst_kernel = np.zeros(dst_kernel_shape, dtype=np.float32)

        # Copy first 7 input channels
        dst_kernel[:, :, :7, :] = src_kernel
        # Last 3 input channels remain zero

        dst_layer.set_weights([dst_kernel, src_bias.copy()])

    # Copy all other compatible layers
    shareable_layers = ['bigru_30', 'bigru_20', 'bigru_10', 'dense_128', 'seq_repr', 'prob_output']

    # Bio branch layers (if biofusion)
    try:
        bio_layers = ['bio_dense', 'bio_norm', 'fusion_dense']
        shareable_layers.extend(bio_layers)
    except:
        pass

    for layer_name in shareable_layers:
        try:
            src_layer = model_src_7ch.get_layer(layer_name)
            dst_layer = model_dst_10ch.get_layer(layer_name)

            src_weights = src_layer.get_weights()
            dst_weights = dst_layer.get_weights()

            # Only copy if shapes match
            if all(s.shape == d.shape for s, d in zip(src_weights, dst_weights)):
                dst_layer.set_weights([w.copy() for w in src_weights])
        except:
            pass


if __name__ == "__main__":
    print("E5-C Encoding Architecture Test")
    print("="*70)

    # Test encoding
    sgrna = "GCTGAAGCACTGCACGCCGTAGG"
    off = "GCTGAAGCACTGCACGCCATAGG"

    enc_ctrl = encode_with_masks(sgrna, off, 'control')
    enc_aug = encode_with_masks(sgrna, off, 'mask_augmented')

    print(f"Control encoding: {enc_ctrl.shape}")
    print(f"Mask-augmented encoding: {enc_aug.shape}")

    # Test models
    for stage in ['p12', 'biofusion']:
        print(f"\n{stage.upper()}:")
        m_ctrl = build_e5c_model('control', stage, 'control', 44)
        m_aug = build_e5c_model('mask_augmented', stage, 'control', 44)

        print(f"  Control: {m_ctrl.count_params():,} params")
        print(f"  Augmented: {m_aug.count_params():,} params")

        copy_cnn_weights_fair(m_ctrl, m_aug)
        print(f"  ✓ Fair init applied")

        # Test shared layer hashes
        ctrl_hashes = get_shared_layer_hashes(m_ctrl)
        aug_hashes = get_shared_layer_hashes(m_aug)
        print(f"  Shared layer hashes computed: {len(ctrl_hashes)} layers")

    print("\n✓ E5-C architecture test passed")
