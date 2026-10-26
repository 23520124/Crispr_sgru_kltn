"""
MODEL_E3V2_Loss — Loss Function Comparison for E3 BioFusion

Three loss variants for imbalanced data:
1. BCE-Dice (control): 0.5 BCE + 0.5 Soft Dice
2. Weighted BCE: class-weighted BCE with pos_weight
3. Focal BCE: focal loss with gamma=2.0

Architecture identical to MODEL_E3_BioFusion, only loss changes.
"""

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


class BCEDiceLoss(tf.keras.losses.Loss):
    """
    BCEDiceLoss (serializable class-based).
    0.5 * BCE + 0.5 * (1 - Soft Dice coefficient)
    """
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

        # BCE component
        loss_bce = self.bce(y_true, y_pred if self.from_logits else p)

        # Soft Dice component
        y_true_f = tf.reshape(y_true, [-1])
        p_f = tf.reshape(p, [-1])
        intersection = tf.reduce_sum(y_true_f * p_f)
        denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(p_f) + self.eps
        loss_dice = 1.0 - (2.0 * intersection + self.eps) / denominator

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice

    def get_config(self):
        return {
            'bce_weight': self.bce_weight,
            'dice_weight': self.dice_weight,
            'eps': self.eps,
            'from_logits': self.from_logits
        }


class WeightedBCELoss(tf.keras.losses.Loss):
    """
    Weighted Binary Cross-Entropy Loss.

    Uses pos_weight to handle class imbalance:
    loss = -[y * log(p) * pos_weight + (1-y) * log(1-p)]

    Args:
        pos_weight: weight for positive class (n_neg / n_pos)
    """
    def __init__(self, pos_weight=1.0, from_logits=False, name="weighted_bce_loss"):
        super(WeightedBCELoss, self).__init__(name=name)
        self.pos_weight = float(pos_weight)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        if self.from_logits:
            # Use tf.nn.weighted_cross_entropy_with_logits for numerical stability
            loss = tf.nn.weighted_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred,
                pos_weight=self.pos_weight
            )
        else:
            # Manual calculation for sigmoid output
            p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            loss = -(y_true * tf.math.log(p) * self.pos_weight +
                    (1.0 - y_true) * tf.math.log(1.0 - p))

        return tf.reduce_mean(loss)

    def get_config(self):
        return {
            'pos_weight': self.pos_weight,
            'from_logits': self.from_logits
        }


class FocalBCELoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss.

    Reduces loss contribution from easy examples (high confidence correct predictions).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else 1-p

    This is PURE focal loss, no Dice component.

    Args:
        gamma: focusing parameter (default 2.0)
        alpha: balancing parameter for positive class (default 0.25)
        from_logits: whether y_pred is logits or probabilities
    """
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, name="focal_bce_loss"):
        super(FocalBCELoss, self).__init__(name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        if self.from_logits:
            p = tf.math.sigmoid(y_pred)
        else:
            p = y_pred

        p = tf.clip_by_value(p, 1e-7, 1.0 - 1e-7)

        # Compute p_t
        p_t = tf.where(tf.equal(y_true, 1.0), p, 1.0 - p)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Binary cross-entropy
        bce = -(y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p))

        # Focal loss with modulation
        focal_loss = focal_weight * bce

        # Apply alpha balancing
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * focal_loss

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        return {
            'gamma': self.gamma,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        }


def Crispr_SGRU_BioFusion_Loss(bio_feature_dim=44, loss_mode='control',
                                pos_weight=1.0, focal_gamma=2.0, focal_alpha=0.25,
                                learning_rate=0.0001):
    """
    MODEL12 + Biological Feature Fusion with configurable loss.

    Sequence branch: MODEL12.Crispr_SGRU backbone
    Biological branch: Dense 64 + LayerNorm + Dropout 0.2
    Fusion: concat → Dense 64 → Dropout 0.3 → Sigmoid

    Args:
        bio_feature_dim: biological feature vector dimension (25/40/44)
        loss_mode: 'control' (BCE-Dice), 'weighted_bce', or 'focal_bce'
        pos_weight: weight for positive class (only for weighted_bce)
        focal_gamma: gamma parameter for focal loss (default 2.0)
        focal_alpha: alpha parameter for focal loss (default 0.25)
        learning_rate: Adam learning rate

    Returns:
        compiled Keras model with two inputs: [sequence, bio_features]
    """
    # =========================================================================
    # SEQUENCE BRANCH (MODEL12 backbone)
    # =========================================================================
    sequence_input = Input(shape=(1, 24, 7), name='sequence_input')

    # Inception module (10 filters each)
    conv_1 = layers.Conv2D(10, (1, 1), padding='same', activation='relu')(sequence_input)
    conv_2 = layers.Conv2D(10, (1, 2), padding='same', activation='relu')(sequence_input)
    conv_3 = layers.Conv2D(10, (1, 3), padding='same', activation='relu')(sequence_input)
    conv_4 = layers.Conv2D(10, (1, 5), padding='same', activation='relu')(sequence_input)
    conv_output = layers.Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    conv_output = layers.Reshape((24, 40))(conv_output)

    # BiGRU stack with skip connections
    x0 = layers.Bidirectional(layers.GRU(30, return_sequences=True))(conv_output)
    inputs_2 = layers.Reshape((24, 7))(sequence_input)
    x = layers.Concatenate(axis=2)([inputs_2, x0])

    x1 = layers.Bidirectional(layers.GRU(20, return_sequences=True))(x)
    x = layers.Concatenate(axis=2)([x0, x1])

    x2 = layers.Bidirectional(layers.GRU(10, return_sequences=True))(x)
    x = layers.Concatenate(axis=-1)([x0, x1, x2])

    # Flatten
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    seq_repr = layers.Dense(64, activation="relu", name='seq_repr')(x)
    seq_repr = layers.Dropout(rate=0.35)(seq_repr)

    # =========================================================================
    # BIOLOGICAL FEATURE BRANCH
    # =========================================================================
    bio_input = Input(shape=(bio_feature_dim,), name='bio_input')

    bio_dense = layers.Dense(64, activation='relu', name='bio_dense')(bio_input)
    bio_norm = layers.LayerNormalization(name='bio_norm')(bio_dense)
    bio_repr = layers.Dropout(0.2, name='bio_dropout')(bio_norm)

    # =========================================================================
    # FUSION
    # =========================================================================
    fusion = layers.Concatenate(name='fusion')([seq_repr, bio_repr])
    fusion_dense = layers.Dense(64, activation='relu', name='fusion_dense')(fusion)
    fusion_dense = layers.Dropout(0.3)(fusion_dense)

    # Output
    probs = layers.Dense(1, activation='sigmoid', name='prob_output')(fusion_dense)

    model = Model(inputs=[sequence_input, bio_input], outputs=probs,
                  name=f'E3V2_{loss_mode}')

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================
    if loss_mode == 'control':
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=False)
    elif loss_mode == 'weighted_bce':
        loss_fn = WeightedBCELoss(pos_weight=pos_weight, from_logits=False)
    elif loss_mode == 'focal_bce':
        loss_fn = FocalBCELoss(gamma=focal_gamma, alpha=focal_alpha, from_logits=False)
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}. Must be 'control', 'weighted_bce', or 'focal_bce'")

    opt = Adam(learning_rate=learning_rate)

    model.compile(
        loss=loss_fn,
        optimizer=opt,
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.AUC(curve='ROC', name='auroc')
        ]
    )
    return model


if __name__ == "__main__":
    print("Testing E3V2 Loss architectures...")
    print("\nNOTE: This is a demonstration only. Actual pos_weight values")
    print("are calculated per-fold during training from train_pos/train_neg counts.\n")

    bio_dim = 44  # E3c

    for loss_mode in ['control', 'weighted_bce', 'focal_bce']:
        print("\n" + "="*70)
        print(f"Loss mode: {loss_mode}")
        print("="*70)

        if loss_mode == 'weighted_bce':
            # Demo value only - actual value computed per fold
            model = Crispr_SGRU_BioFusion_Loss(
                bio_feature_dim=bio_dim,
                loss_mode=loss_mode,
                pos_weight=10.0,  # placeholder for demo
                learning_rate=0.0001
            )
        elif loss_mode == 'focal_bce':
            model = Crispr_SGRU_BioFusion_Loss(
                bio_feature_dim=bio_dim,
                loss_mode=loss_mode,
                focal_gamma=2.0,
                focal_alpha=0.25,
                learning_rate=0.0001
            )
        else:
            model = Crispr_SGRU_BioFusion_Loss(
                bio_feature_dim=bio_dim,
                loss_mode=loss_mode,
                learning_rate=0.0001
            )

        model.summary()
        params = model.count_params()
        print(f"\nTotal parameters: {params:,}")

    print("\n✓ Architecture test passed")

