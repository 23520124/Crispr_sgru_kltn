"""
MODEL_E3_BioFusion — MODEL12 + Biological Feature Fusion

Kiến trúc:
- Sequence branch: MODEL12.Crispr_SGRU backbone
  - Inception: Conv2D 10 filters × 4 kernels (1,1), (1,2), (1,3), (1,5)
  - BiGRU stack: 30/20/10 với skip connections (concat input gốc và tầng trước)
  - GRU thứ ba: return_sequences=True
  - Flatten → Dense 128 → Dense 64 → Dropout 0.35
- Biological branch: Dense 64 + LayerNorm + Dropout 0.2
- Fusion: concatenate + Dense head + sigmoid output
- Loss: BCEDiceLoss (0.5 BCE + 0.5 Soft Dice)
- Optimizer: Adam 1e-4

So với MODEL12:
- Control: sequence input only
- BioFusion: sequence + biological features, cùng backbone

Kiến trúc backbone giống MODEL12.Crispr_SGRU.
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


def Crispr_SGRU_MODEL12_Control(learning_rate=0.0001):
    """
    MODEL12.Crispr_SGRU baseline for E3 control.

    Architecture (exact copy from MODEL12.PY):
    - Inception: Conv2D 10 filters × 4 kernels
    - BiGRU 30/20/10 với skip connections (concat input gốc và tầng trước)
    - GRU thứ ba return_sequences=True
    - Flatten → Dense 128 → Dense 64 → Dropout 0.35 → Sigmoid

    Returns:
        compiled Keras model
    """
    inputs_1 = Input(shape=(1, 24, 7), name='main_input')

    # Inception module (10 filters each)
    conv_1 = layers.Conv2D(10, (1, 1), padding='same', activation='relu')(inputs_1)
    conv_2 = layers.Conv2D(10, (1, 2), padding='same', activation='relu')(inputs_1)
    conv_3 = layers.Conv2D(10, (1, 3), padding='same', activation='relu')(inputs_1)
    conv_4 = layers.Conv2D(10, (1, 5), padding='same', activation='relu')(inputs_1)
    conv_output = layers.Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    conv_output = layers.Reshape((24, 40))(conv_output)

    # BiGRU stack with skip connections
    x0 = layers.Bidirectional(layers.GRU(30, return_sequences=True))(conv_output)
    inputs_2 = layers.Reshape((24, 7))(inputs_1)
    x = layers.Concatenate(axis=2)([inputs_2, x0])

    x1 = layers.Bidirectional(layers.GRU(20, return_sequences=True))(x)
    x = layers.Concatenate(axis=2)([x0, x1])

    x2 = layers.Bidirectional(layers.GRU(10, return_sequences=True))(x)  # return_sequences=True
    x = layers.Concatenate(axis=-1)([x0, x1, x2])

    # Flatten and classifier
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(rate=0.35)(x)

    # Output
    probs = layers.Dense(1, activation='sigmoid', name='prob_output')(x)

    model = Model(inputs=inputs_1, outputs=probs, name='MODEL12_Control')

    opt = Adam(learning_rate=learning_rate)
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=False)

    model.compile(
        loss=loss_fn,
        optimizer=opt,
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.AUC(curve='ROC', name='auroc')
        ]
    )
    return model


def Crispr_SGRU_BioFusion(bio_feature_dim=44, learning_rate=0.0001):
    """
    MODEL12 + Biological Feature Fusion.

    Sequence branch: MODEL12.Crispr_SGRU backbone (giống control)
    Biological branch: Dense 64 + LayerNorm + Dropout 0.2
    Fusion: concat → Dense 64 → Dropout 0.3 → Sigmoid

    Args:
        bio_feature_dim: biological feature vector dimension (25/40/44)
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

    x2 = layers.Bidirectional(layers.GRU(10, return_sequences=True))(x)  # return_sequences=True
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

    model = Model(inputs=[sequence_input, bio_input], outputs=probs, name='MODEL12_BioFusion')

    opt = Adam(learning_rate=learning_rate)
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=False)

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
    print("Testing E3 BioFusion architecture...")

    # Test MODEL12 control
    print("\n" + "="*70)
    print("MODEL12 CONTROL")
    print("="*70)
    model_control = Crispr_SGRU_MODEL12_Control(learning_rate=0.0001)
    model_control.summary()
    control_params = model_control.count_params()
    print(f"\nTotal parameters: {control_params:,}")

    # Test E3 variants
    for feature_set, dim in [('E3a', 25), ('E3b', 40), ('E3c', 44)]:
        print("\n" + "="*70)
        print(f"{feature_set.upper()} (bio_dim={dim})")
        print("="*70)
        model_e3 = Crispr_SGRU_BioFusion(
            bio_feature_dim=dim,
            learning_rate=0.0001
        )
        model_e3.summary()
        e3_params = model_e3.count_params()
        print(f"\nTotal parameters: {e3_params:,}")
        print(f"Parameter increase: {e3_params - control_params:,} (+{(e3_params/control_params - 1)*100:.1f}%)")

    print("\n✓ Architecture test passed")
