"""
MODEL_E5B_GRU — GRU Ablation Study

Kiến trúc E5-B: kiểm tra riêng tác dụng của tầng GRU stack.

Two modes:
1. control: Full architecture WITH BiGRU stack
   CNN encoder → BiGRU stack (30/20/10) → Flatten → Dense head → output

2. no_gru: Direct connection WITHOUT GRU stack
   CNN encoder → Flatten → Dense head → output

Single factor change:
- Presence/absence of GRU stack ONLY
- Same CNN encoder, loss, optimizer, sampling, checkpoint, threshold
- Same splits, feature_set, batch_size, epochs, steps_per_epoch
- Same seed for CNN initialization

Shared weights between control and no_gru:
- CNN encoder: all Conv2D layers (p12_control, residual_cnn, or convnext_1d)
- Biofusion branch (if stage=='biofusion'): bio_dense, bio_norm, bio_dropout
- Head and GRU NOT shared: no_gru doesn't have GRU, head input shapes differ

Stage support:
- p12: sequence-only
- biofusion: sequence + biological features

This is E5-B: isolate GRU effect only.
Other E5 sub-experiments (bottleneck, encoding, loss) are separate.
"""

import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam


class BCEDiceLoss(tf.keras.losses.Loss):
    """
    BCEDiceLoss (0.5 BCE + 0.5 Soft Dice).
    Serializable class-based implementation.
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

        # Soft Dice component (positive class only)
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


class FocalBCELoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
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


def build_p12_control_encoder(sequence_input):
    """
    MODEL12 Inception CNN encoder (p12_control baseline).

    Architecture:
    - 4 parallel Conv2D: kernels (1,1), (1,2), (1,3), (1,5), each 10 filters
    - Concatenate → 40 channels
    - Reshape to (24, 40)

    Args:
        sequence_input: Input tensor (batch, 1, 24, 7)

    Returns:
        encoder_output: (batch, 24, 40)
    """
    conv_1 = layers.Conv2D(10, (1, 1), padding='same', activation='relu', name='p12_conv_1x1')(sequence_input)
    conv_2 = layers.Conv2D(10, (1, 2), padding='same', activation='relu', name='p12_conv_1x2')(sequence_input)
    conv_3 = layers.Conv2D(10, (1, 3), padding='same', activation='relu', name='p12_conv_1x3')(sequence_input)
    conv_4 = layers.Conv2D(10, (1, 5), padding='same', activation='relu', name='p12_conv_1x5')(sequence_input)

    conv_output = layers.Concatenate(axis=-1, name='p12_concat')([conv_1, conv_2, conv_3, conv_4])
    encoder_output = layers.Reshape((24, 40), name='p12_reshape')(conv_output)

    return encoder_output


def build_residual_cnn_encoder(sequence_input):
    """
    Residual CNN encoder (E4 variant).

    Architecture:
    - Conv2D 10 filters (1, 1) → projection to 10 channels
    - Conv2D 10 filters (1, 3) with residual connection
    - Conv2D 40 filters (1, 1) → projection to 40 channels
    - Reshape to (24, 40)

    Args:
        sequence_input: Input tensor (batch, 1, 24, 7)

    Returns:
        encoder_output: (batch, 24, 40)
    """
    # Projection 7→10
    x = layers.Conv2D(10, (1, 1), padding='same', activation='relu', name='res_proj_in')(sequence_input)

    # Residual block
    residual = x
    x = layers.Conv2D(10, (1, 3), padding='same', activation='relu', name='res_conv_1x3')(x)
    x = layers.Add(name='res_add')([residual, x])

    # Projection 10→40
    x = layers.Conv2D(40, (1, 1), padding='same', activation='relu', name='res_proj_out')(x)

    encoder_output = layers.Reshape((24, 40), name='res_reshape')(x)
    return encoder_output


def build_convnext_1d_encoder(sequence_input, kernel_size=7):
    """
    ConvNeXt-style 1D block encoder on sequence dimension (E4 variant).

    Architecture (true 1D ConvNeXt block):
    - Reshape input (1, 24, 7) → (24, 7) for 1D operations
    - Projection: Dense 7→40 channels
    - Depthwise Conv1D: kernel_size on sequence dimension (24)
    - LayerNormalization
    - Pointwise expansion: Dense 40→160
    - GELU activation
    - Pointwise projection: Dense 160→40
    - Residual connection
    - Output: (24, 40)

    Args:
        sequence_input: Input tensor (batch, 1, 24, 7)
        kernel_size: depthwise conv kernel size on sequence dimension (5 or 7)

    Returns:
        encoder_output: (batch, 24, 40)
    """
    # Reshape (batch, 1, 24, 7) -> (batch, 24, 7)
    x = layers.Reshape((24, 7), name='cnext_reshape_in')(sequence_input)

    # Projection 7→40 (channel-wise, applied at each position)
    x = layers.Dense(40, name='cnext_proj')(x)
    residual = x

    # Depthwise convolution on sequence dimension using Conv1D with groups
    x = layers.Conv1D(
        filters=40,
        kernel_size=kernel_size,
        padding='same',
        groups=40,  # Depthwise: each of 40 channels has its own filter
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name=f'cnext_dw_1d_k{kernel_size}'
    )(x)

    # LayerNormalization (over channels at each position)
    x = layers.LayerNormalization(epsilon=1e-6, name='cnext_norm')(x)

    # Pointwise expansion: 40→160 (4x expansion)
    x = layers.Dense(160, name='cnext_pw_expand')(x)

    # GELU activation
    x = layers.Activation('gelu', name='cnext_gelu')(x)

    # Pointwise projection: 160→40
    x = layers.Dense(40, name='cnext_pw_project')(x)

    # Residual connection
    encoder_output = layers.Add(name='cnext_add')([residual, x])

    return encoder_output


def build_gru_backbone(encoder_output, sequence_input):
    """
    Shared BiGRU stack (MODEL12 architecture).

    Architecture:
    - BiGRU 30 (return_sequences=True)
    - Concat with original input (24, 7)
    - BiGRU 20 (return_sequences=True)
    - Concat with first GRU output
    - BiGRU 10 (return_sequences=True)
    - Concat all GRU outputs → (24, 120)

    Args:
        encoder_output: CNN output (batch, 24, 40)
        sequence_input: Original sequence input (batch, 1, 24, 7) for skip connection

    Returns:
        gru_output: (batch, 24, 120) concatenated GRU features
    """
    # First BiGRU
    x0 = layers.Bidirectional(layers.GRU(30, return_sequences=True), name='bigru_30')(encoder_output)

    # Skip connection with original input
    inputs_reshaped = layers.Reshape((24, 7), name='input_reshape')(sequence_input)
    x = layers.Concatenate(axis=2, name='concat_input_gru1')([inputs_reshaped, x0])

    # Second BiGRU
    x1 = layers.Bidirectional(layers.GRU(20, return_sequences=True), name='bigru_20')(x)
    x = layers.Concatenate(axis=2, name='concat_gru1_gru2')([x0, x1])

    # Third BiGRU
    x2 = layers.Bidirectional(layers.GRU(10, return_sequences=True), name='bigru_10')(x)

    # Concatenate all GRU outputs
    gru_output = layers.Concatenate(axis=-1, name='concat_all_gru')([x0, x1, x2])

    return gru_output


def build_e5b_model(stage='p12', mode='control', cnn_variant='p12_control',
                    bio_feature_dim=44, loss_mode='control', learning_rate=0.0001):
    """
    Build E5-B model with or without GRU stack.

    Control mode (WITH GRU):
        CNN encoder → BiGRU stack (30/20/10) → Flatten → Dense(128) → Dense(64) → output

    No_gru mode (WITHOUT GRU):
        CNN encoder → Flatten → Dense(128) → Dense(64) → output

    Args:
        stage: 'p12' (sequence-only) or 'biofusion' (sequence + bio features)
        mode: 'control' (with GRU) or 'no_gru' (without GRU)
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d' (from E4 winners)
        bio_feature_dim: biological feature dimension (for biofusion stage)
        loss_mode: 'control' (BCEDiceLoss) or 'focal_bce' (FocalBCELoss)
        learning_rate: Adam learning rate

    Returns:
        compiled Keras model
    """
    # =============================================================================
    # SEQUENCE BRANCH
    # =============================================================================
    sequence_input = Input(shape=(1, 24, 7), name='sequence_input')

    # CNN encoder (selected based on E4 winner for dataset/stage)
    if cnn_variant == 'p12_control':
        encoder_output = build_p12_control_encoder(sequence_input)
    elif cnn_variant == 'residual_cnn':
        encoder_output = build_residual_cnn_encoder(sequence_input)
    elif cnn_variant == 'convnext_1d':
        encoder_output = build_convnext_1d_encoder(sequence_input, kernel_size=7)
    else:
        raise ValueError(f"Unknown cnn_variant: {cnn_variant}")

    # encoder_output shape: (batch, 24, 40)

    # =============================================================================
    # GRU STACK (E5-B INTERVENTION) - only in control mode
    # =============================================================================
    if mode == 'control':
        # WITH GRU stack
        gru_output = build_gru_backbone(encoder_output, sequence_input)
        # gru_output shape: (batch, 24, 120)
        x = layers.Flatten(name='flatten')(gru_output)
        # x shape: (batch, 2880)
    elif mode == 'no_gru':
        # WITHOUT GRU stack - direct connection from CNN to Flatten
        x = layers.Flatten(name='flatten')(encoder_output)
        # x shape: (batch, 24 * 40) = (batch, 960)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'control' or 'no_gru'")

    # =============================================================================
    # HEAD
    # =============================================================================
    x = layers.Dense(128, activation="relu", name='dense_128')(x)
    x = layers.Dense(64, activation="relu", name='dense_64')(x)
    x = layers.Dropout(rate=0.35, name='dropout_head')(x)

    # =============================================================================
    # BIOFUSION (if stage=='biofusion')
    # =============================================================================
    if stage == 'biofusion':
        bio_input = Input(shape=(bio_feature_dim,), name='bio_input')

        # Biological branch
        bio_branch = layers.Dense(64, activation='relu', name='bio_dense')(bio_input)
        bio_branch = layers.LayerNormalization(name='bio_norm')(bio_branch)
        bio_branch = layers.Dropout(0.2, name='bio_dropout')(bio_branch)

        # Fusion
        x = layers.Concatenate(name='fusion_concat')([x, bio_branch])
        x = layers.Dense(64, activation='relu', name='fusion_dense')(x)
        x = layers.Dropout(0.3, name='fusion_dropout')(x)

        inputs = [sequence_input, bio_input]
    else:
        inputs = sequence_input

    # =============================================================================
    # OUTPUT
    # =============================================================================
    probs = layers.Dense(1, activation='sigmoid', name='prob_output')(x)

    model = Model(inputs=inputs, outputs=probs, name=f'E5B_{stage}_{mode}')

    # =============================================================================
    # COMPILE
    # =============================================================================
    opt = Adam(learning_rate=learning_rate)

    if loss_mode == 'focal_bce':
        loss_fn = FocalBCELoss(gamma=2.0, alpha=0.25, from_logits=False)
    else:  # 'control' mode
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


def get_shared_layer_names(stage='p12', cnn_variant='p12_control'):
    """
    Get list of shared layer names (CNN only, excluding GRU and head).

    For E5-B, only CNN encoder can be shared between control and no_gru modes.
    GRU layers exist only in control mode.
    Head layers have different input shapes (2880 for control, 960 for no_gru).

    Args:
        stage: 'p12' or 'biofusion'
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'

    Returns:
        list of layer names
    """
    # CNN encoder layers (variant-specific)
    if cnn_variant == 'p12_control':
        cnn_layers = ['p12_conv_1x1', 'p12_conv_1x2', 'p12_conv_1x3', 'p12_conv_1x5',
                      'p12_concat', 'p12_reshape']
    elif cnn_variant == 'residual_cnn':
        cnn_layers = ['res_proj_in', 'res_conv_1x3', 'res_add', 'res_proj_out', 'res_reshape']
    elif cnn_variant == 'convnext_1d':
        cnn_layers = ['cnext_reshape_in', 'cnext_proj', 'cnext_dw_1d_k7',
                      'cnext_norm', 'cnext_pw_expand', 'cnext_gelu',
                      'cnext_pw_project', 'cnext_add']
    else:
        raise ValueError(f"Unknown cnn_variant: {cnn_variant}")

    shared = cnn_layers.copy()

    # Biofusion branch can be shared (same shape regardless of GRU presence)
    if stage == 'biofusion':
        shared.extend([
            'bio_dense', 'bio_norm', 'bio_dropout'
        ])

    # Note: GRU layers (bigru_30, bigru_20, bigru_10, etc.) are NOT shared
    # because they don't exist in no_gru mode
    # Head layers (flatten, dense_128, dense_64, dropout_head) are NOT shared
    # because input shapes differ between control and no_gru

    return shared


def capture_shared_weights(model, stage='p12', cnn_variant='p12_control'):
    """
    Capture immutable numpy copies of shared weights (CNN encoder only).

    Args:
        model: Keras model
        stage: 'p12' or 'biofusion'
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'

    Returns:
        dict {layer_name: list of numpy arrays}
    """
    shared_names = get_shared_layer_names(stage, cnn_variant)
    shared_weights = {}

    for layer in model.layers:
        if layer.name in shared_names:
            weights = layer.get_weights()
            if len(weights) > 0:
                # Create immutable copies
                shared_weights[layer.name] = [w.copy() for w in weights]

    return shared_weights


def apply_shared_weights(model, shared_weights, stage='p12', cnn_variant='p12_control'):
    """
    Apply captured shared weights to model.

    Args:
        model: Keras model
        shared_weights: dict from capture_shared_weights
        stage: 'p12' or 'biofusion'
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'
    """
    for layer in model.layers:
        if layer.name in shared_weights:
            layer.set_weights(shared_weights[layer.name])


def verify_shared_weights_match(model, shared_weights, stage='p12', cnn_variant='p12_control'):
    """
    Verify that model's shared weights match captured weights exactly.

    Args:
        model: Keras model
        shared_weights: dict from capture_shared_weights
        stage: 'p12' or 'biofusion'
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'

    Raises:
        AssertionError if weights don't match
    """
    for layer in model.layers:
        if layer.name in shared_weights:
            current = layer.get_weights()
            reference = shared_weights[layer.name]
            assert len(current) == len(reference), \
                f"Layer {layer.name}: weight count mismatch"
            for i, (c, r) in enumerate(zip(current, reference)):
                assert np.array_equal(c, r), \
                    f"Layer {layer.name}, weight {i}: values don't match"


def hash_shared_weights(shared_weights):
    """
    Hash shared weights in canonical order.

    Args:
        shared_weights: dict from capture_shared_weights

    Returns:
        hash string (16 chars)
    """
    # Sort by layer name for deterministic ordering
    sorted_items = sorted(shared_weights.items())

    weights_bytes = b''
    for layer_name, weight_list in sorted_items:
        for w in weight_list:
            weights_bytes += w.tobytes()

    return hashlib.md5(weights_bytes).hexdigest()[:16]


def get_shared_weights_hash_from_model(model, stage='p12', cnn_variant='p12_control'):
    """
    Unified function to compute shared weights hash from model.

    Args:
        model: Keras model
        stage: 'p12' or 'biofusion'
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'

    Returns:
        hash string (16 chars)
    """
    shared_weights = capture_shared_weights(model, stage, cnn_variant)
    return hash_shared_weights(shared_weights)


def count_model_parameters(stage='p12', mode='control', cnn_variant='p12_control',
                           bio_feature_dim=44):
    """
    Count parameters for E5-B model configuration.

    Args:
        stage: 'p12' or 'biofusion'
        mode: 'control' or 'no_gru'
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'
        bio_feature_dim: biological feature dimension

    Returns:
        dict with parameter counts
    """
    model = build_e5b_model(
        stage=stage,
        mode=mode,
        cnn_variant=cnn_variant,
        bio_feature_dim=bio_feature_dim,
        loss_mode='control',
        learning_rate=0.0001
    )

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'mode': mode,
        'cnn_variant': cnn_variant,
        'has_gru': mode == 'control'
    }
