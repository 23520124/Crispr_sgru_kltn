"""
MODEL_E5D_Loss — Loss Function Study

E5-D experiment: isolate effect of loss function.

Three loss modes:
1. bce: Pure BinaryCrossentropy
2. bce_dice: Control (0.5 BCE + 0.5 Soft Dice)
3. bce_ranking: BCE + lambda_rank * pairwise ranking loss

Single factor change: loss function ONLY
All other factors FIXED to E3v2 winner per dataset:
- Control encoding (7 channels)
- E3v2 architecture
- Dataset-specific feature set, sampling, checkpoint monitor
"""

import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam


class BCELoss(tf.keras.losses.Loss):
    """Pure Binary Cross-Entropy Loss."""
    def __init__(self, from_logits=False, name="bce_loss"):
        super(BCELoss, self).__init__(name=name)
        self.from_logits = from_logits
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, y_true, y_pred):
        return self.bce(y_true, y_pred)

    def get_config(self):
        return {'from_logits': self.from_logits}


class BCEDiceLoss(tf.keras.losses.Loss):
    """BCE-Dice Loss (0.5 BCE + 0.5 Soft Dice) — E5-D control."""
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


class BCERankingLoss(tf.keras.losses.Loss):
    """
    BCE + Pairwise Ranking Loss.

    Uses positive-negative pairs from same batch.
    Ranking term: lambda_rank * mean(log(1 + exp(-(p_pos - p_neg))))
    """
    def __init__(self, lambda_rank=0.01, from_logits=False, name="bce_ranking_loss"):
        super(BCERankingLoss, self).__init__(name=name)
        self.lambda_rank = float(lambda_rank)
        self.from_logits = from_logits
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        p = tf.math.sigmoid(y_pred) if self.from_logits else y_pred

        # BCE term
        loss_bce = self.bce(y_true, y_pred if self.from_logits else p)

        # Ranking term: pairwise logistic loss
        # Find positive and negative indices
        y_true_flat = tf.reshape(y_true, [-1])
        p_flat = tf.reshape(p, [-1])

        pos_mask = tf.equal(y_true_flat, 1.0)
        neg_mask = tf.equal(y_true_flat, 0.0)

        # Cast indices to int32 for consistent gather operations
        pos_indices = tf.cast(tf.reshape(tf.where(pos_mask), [-1]), tf.int32)
        neg_indices = tf.cast(tf.reshape(tf.where(neg_mask), [-1]), tf.int32)

        n_pos = tf.shape(pos_indices)[0]
        n_neg = tf.shape(neg_indices)[0]

        # If no pairs, ranking loss = 0
        has_pairs = tf.logical_and(n_pos > 0, n_neg > 0)

        def compute_ranking_loss():
            p_pos = tf.gather(p_flat, pos_indices)  # (n_pos,)
            p_neg = tf.gather(p_flat, neg_indices)  # (n_neg,)

            # Pairwise differences: p_pos[:, None] - p_neg[None, :]
            # Shape: (n_pos, n_neg)
            p_pos_exp = tf.expand_dims(p_pos, axis=1)  # (n_pos, 1)
            p_neg_exp = tf.expand_dims(p_neg, axis=0)  # (1, n_neg)
            diff = p_pos_exp - p_neg_exp  # (n_pos, n_neg)

            # Logistic loss: log(1 + exp(-diff))
            # Use softplus(-diff) for numerical stability
            ranking_term = tf.nn.softplus(-diff)
            return tf.reduce_mean(ranking_term)

        def no_ranking_loss():
            # Return same dtype/shape as loss_bce for consistent tf.cond
            return tf.zeros_like(loss_bce)

        ranking_loss = tf.cond(has_pairs, compute_ranking_loss, no_ranking_loss)

        return loss_bce + self.lambda_rank * ranking_loss

    def get_config(self):
        return {'lambda_rank': self.lambda_rank, 'from_logits': self.from_logits}


def encode_control(sgrna: str, offtarget: str) -> np.ndarray:
    """
    Encode sgRNA/off-target pair with control 7-channel encoding.

    Returns: (1, 24, 7)
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from Encoder_sgRNA_off import Encoder

    encoder = Encoder(sgrna, offtarget)
    base_encoding = encoder.on_off_code  # (24, 7)
    return base_encoding.reshape(1, 24, 7)


def build_e5d_model(stage: str, loss_mode: str, bio_feature_dim=44,
                    lambda_rank=0.01, learning_rate=0.0001):
    """
    Build E5-D model with E3v2 winner architecture and control 7-channel encoding.

    Args:
        stage: 'p12' or 'biofusion'
        loss_mode: 'bce', 'bce_dice', or 'bce_ranking'
        bio_feature_dim: 25 (e3a), 40 (e3b), or 44 (e3c)
        lambda_rank: ranking loss weight for bce_ranking mode
        learning_rate: Adam learning rate
    """
    assert stage in ['p12', 'biofusion']
    assert loss_mode in ['bce', 'bce_dice', 'bce_ranking']

    # Sequence branch (control 7-channel encoding)
    sequence_input = Input(shape=(1, 24, 7), name='sequence_input')

    # Inception CNN (10 filters each)
    conv_1 = layers.Conv2D(10, (1, 1), padding='same', activation='relu', name='conv_1x1')(sequence_input)
    conv_2 = layers.Conv2D(10, (1, 2), padding='same', activation='relu', name='conv_1x2')(sequence_input)
    conv_3 = layers.Conv2D(10, (1, 3), padding='same', activation='relu', name='conv_1x3')(sequence_input)
    conv_4 = layers.Conv2D(10, (1, 5), padding='same', activation='relu', name='conv_1x5')(sequence_input)
    conv_output = layers.Concatenate(axis=-1, name='conv_concat')([conv_1, conv_2, conv_3, conv_4])
    conv_output = layers.Reshape((24, 40), name='conv_reshape')(conv_output)

    # BiGRU stack with skip connections
    x0 = layers.Bidirectional(layers.GRU(30, return_sequences=True), name='bigru_30')(conv_output)

    # Skip connection from input
    inputs_skip = layers.Reshape((24, 7), name='input_reshape')(sequence_input)
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
        model = Model(inputs=sequence_input, outputs=probs, name=f'E5D_{loss_mode}_p12')

    elif stage == 'biofusion':
        bio_input = Input(shape=(bio_feature_dim,), name='bio_input')
        bio_dense = layers.Dense(64, activation='relu', name='bio_dense')(bio_input)
        bio_norm = layers.LayerNormalization(name='bio_norm')(bio_dense)
        bio_repr = layers.Dropout(0.2, name='bio_dropout')(bio_norm)

        fusion = layers.Concatenate(name='fusion')([seq_repr, bio_repr])
        fusion_dense = layers.Dense(64, activation='relu', name='fusion_dense')(fusion)
        fusion_dense = layers.Dropout(0.3, name='fusion_dropout')(fusion_dense)
        probs = layers.Dense(1, activation='sigmoid', name='prob_output')(fusion_dense)

        model = Model(inputs=[sequence_input, bio_input], outputs=probs, name=f'E5D_{loss_mode}_biofusion')

    # Select loss
    if loss_mode == 'bce':
        loss_fn = BCELoss(from_logits=False)
    elif loss_mode == 'bce_dice':
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=False)
    elif loss_mode == 'bce_ranking':
        loss_fn = BCERankingLoss(lambda_rank=lambda_rank, from_logits=False)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[]
    )

    return model


def get_shared_layer_hashes(model):
    """
    Compute SHA256 hash for each layer's weights.
    Returns dict: {layer_name: hash_hex}
    """
    hashes = {}
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights_concat = np.concatenate([w.flatten() for w in layer.get_weights()])
            weights_bytes = weights_concat.tobytes()
            layer_hash = hashlib.sha256(weights_bytes).hexdigest()[:16]
            hashes[layer.name] = layer_hash
    return hashes


def copy_all_weights_fair(model_source, model_target):
    """
    Copy all weights from source to target for fair initialization.
    Both models must have identical architecture.
    """
    model_target.set_weights(model_source.get_weights())
