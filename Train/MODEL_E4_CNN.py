"""
MODEL_E4_CNN — CNN Encoder Comparison for E4 Experiment

Three CNN encoder variants on MODEL12 backbone:
1. p12_control: Original MODEL12 Inception CNN (baseline)
2. residual_cnn: Residual CNN with comparable parameter budget
3. convnext_1d: ConvNeXt-style 1D block (depthwise conv + LayerNorm + GELU)

All variants:
- Input: (1, 24, 7) sequence encoding
- Output: (24, 40) for BiGRU input
- Identical GRU stack (30/20/10), head, optimizer
- Shared initial weights enforced via explicit copy

Two stages:
- p12: sequence-only (single input)
- biofusion: sequence + biological features (two inputs)

Loss: Dataset-dependent
- K562: FocalBCELoss (gamma=2.0, alpha=0.25) from E3v2-A
- HEK293t, II5, II6: BCEDiceLoss (0.5 BCE + 0.5 Soft Dice)

Supported datasets: K562, HEK293t, II5, II6
BE3 not supported (E3v2-B baseline not available)

Optimizer: Adam 1e-4
"""

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam


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


class FocalBCELoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss.

    Reduces loss contribution from easy examples (high confidence correct predictions).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else 1-p

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


# =============================================================================
# CNN ENCODER VARIANTS
# =============================================================================

def build_p12_control_encoder(sequence_input):
    """
    P12 Inception CNN encoder (baseline from MODEL12).

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
    Residual CNN encoder with comparable parameter budget to P12 control.

    Architecture:
    - Projection: Conv2D (1,1) 7→10 channels + ReLU
    - Residual block: Conv2D (1,3) 10→10 + ReLU + residual connection
    - Projection: Conv2D (1,1) 10→40 channels + ReLU
    - Reshape to (24, 40)

    Parameter count:
    - P12 control: ~810 params (4 branches × 10 filters × 7 channels × kernel)
    - This residual: ~800 params (7*10 + 10 + 10*3*10 + 10 + 10*40 + 40)

    Args:
        sequence_input: Input tensor (batch, 1, 24, 7)

    Returns:
        encoder_output: (batch, 24, 40)
    """
    # Projection 7→10
    x = layers.Conv2D(10, (1, 1), padding='same', activation='relu', name='res_proj_in')(sequence_input)

    # Residual block: conv (1,3) 10→10 with residual connection
    residual = x
    x = layers.Conv2D(10, (1, 3), padding='same', activation='relu', name='res_conv_1x3')(x)
    x = layers.Add(name='res_add')([residual, x])

    # Projection 10→40
    x = layers.Conv2D(40, (1, 1), padding='same', activation='relu', name='res_proj_out')(x)

    encoder_output = layers.Reshape((24, 40), name='res_reshape')(x)
    return encoder_output


def build_convnext_1d_encoder(sequence_input, kernel_size=7):
    """
    ConvNeXt-style 1D block encoder on sequence dimension.

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

    Key ConvNeXt design choices:
    - Depthwise convolution on sequence axis
    - LayerNorm instead of BatchNorm
    - GELU activation
    - Inverted bottleneck (expand then compress)
    - Residual connection

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
    # TensorFlow Conv1D supports groups parameter for depthwise convolution
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


# =============================================================================
# SHARED GRU BACKBONE
# =============================================================================

def build_gru_backbone(encoder_output, sequence_input):
    """
    Shared BiGRU stack for all CNN variants (MODEL12 architecture).

    Architecture:
    - BiGRU 30 (return_sequences=True)
    - Concat with original input (24, 7)
    - BiGRU 20 (return_sequences=True)
    - Concat with first GRU output
    - BiGRU 10 (return_sequences=True)
    - Concat all GRU outputs
    - Flatten

    Args:
        encoder_output: CNN output (batch, 24, 40)
        sequence_input: Original sequence input (batch, 1, 24, 7) for skip connection

    Returns:
        flattened_output: (batch, 24*120) = (batch, 2880)
    """
    x0 = layers.Bidirectional(layers.GRU(30, return_sequences=True), name='bigru_30')(encoder_output)
    inputs_2 = layers.Reshape((24, 7), name='skip_reshape')(sequence_input)
    x = layers.Concatenate(axis=2, name='skip_concat_1')([inputs_2, x0])

    x1 = layers.Bidirectional(layers.GRU(20, return_sequences=True), name='bigru_20')(x)
    x = layers.Concatenate(axis=2, name='skip_concat_2')([x0, x1])

    x2 = layers.Bidirectional(layers.GRU(10, return_sequences=True), name='bigru_10')(x)
    x = layers.Concatenate(axis=-1, name='gru_concat_final')([x0, x1, x2])

    flattened = layers.Flatten(name='flatten')(x)
    return flattened


def build_classifier_head_p12(flattened_output, stage='p12'):
    """
    Classifier head for P12-only stage (sequence-only).

    Architecture (MODEL12):
    - Dense 128 + ReLU
    - Dense 64 + ReLU
    - Dropout 0.35
    - Dense 1 + Sigmoid

    Args:
        flattened_output: (batch, 2880)
        stage: 'p12' for this head

    Returns:
        probs: (batch, 1) sigmoid probabilities
    """
    x = layers.Dense(128, activation="relu", name='head_dense_128')(flattened_output)
    x = layers.Dense(64, activation="relu", name='head_dense_64')(x)
    x = layers.Dropout(rate=0.35, name='head_dropout')(x)
    probs = layers.Dense(1, activation='sigmoid', name='prob_output')(x)
    return probs


def build_biofusion_head(flattened_output, bio_input, bio_feature_dim):
    """
    Classifier head for biofusion stage (sequence + biological features).

    Architecture (E3 BioFusion):
    - Sequence representation: Dense 128 → Dense 64 → Dropout 0.35
    - Biological branch: Dense 64 → LayerNorm → Dropout 0.2
    - Fusion: Concat → Dense 64 → Dropout 0.3 → Dense 1 + Sigmoid

    Args:
        flattened_output: (batch, 2880) from GRU backbone
        bio_input: (batch, bio_feature_dim) biological features
        bio_feature_dim: dimension of biological feature vector

    Returns:
        probs: (batch, 1) sigmoid probabilities
    """
    # Sequence representation
    seq_repr = layers.Dense(128, activation="relu", name='seq_dense_128')(flattened_output)
    seq_repr = layers.Dense(64, activation="relu", name='seq_dense_64')(seq_repr)
    seq_repr = layers.Dropout(rate=0.35, name='seq_dropout')(seq_repr)

    # Biological branch
    bio_dense = layers.Dense(64, activation='relu', name='bio_dense')(bio_input)
    bio_norm = layers.LayerNormalization(name='bio_norm')(bio_dense)
    bio_repr = layers.Dropout(0.2, name='bio_dropout')(bio_norm)

    # Fusion
    fusion = layers.Concatenate(name='fusion')([seq_repr, bio_repr])
    fusion_dense = layers.Dense(64, activation='relu', name='fusion_dense')(fusion)
    fusion_dense = layers.Dropout(0.3, name='fusion_dropout')(fusion_dense)
    probs = layers.Dense(1, activation='sigmoid', name='prob_output')(fusion_dense)

    return probs


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_e4_cnn_model(cnn_variant='p12_control', stage='p12', bio_feature_dim=44,
                       loss_mode='control', focal_gamma=2.0, focal_alpha=0.25,
                       convnext_kernel_size=7, learning_rate=0.0001):
    """
    Build E4 CNN model with specified encoder variant and stage.

    Args:
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'
        stage: 'p12' (sequence-only) or 'biofusion' (sequence + bio features)
        bio_feature_dim: dimension of biological feature vector (only for biofusion)
        loss_mode: 'control' (BCE-Dice) or 'focal_bce'
        focal_gamma: gamma parameter for focal loss (default 2.0)
        focal_alpha: alpha parameter for focal loss (default 0.25)
        convnext_kernel_size: kernel size for ConvNeXt depthwise conv (5 or 7)
        learning_rate: Adam learning rate

    Returns:
        compiled Keras model
    """
    # Input
    sequence_input = Input(shape=(1, 24, 7), name='sequence_input')

    # CNN Encoder
    if cnn_variant == 'p12_control':
        encoder_output = build_p12_control_encoder(sequence_input)
    elif cnn_variant == 'residual_cnn':
        encoder_output = build_residual_cnn_encoder(sequence_input)
    elif cnn_variant == 'convnext_1d':
        encoder_output = build_convnext_1d_encoder(sequence_input, kernel_size=convnext_kernel_size)
    else:
        raise ValueError(f"Unknown cnn_variant: {cnn_variant}. Must be 'p12_control', 'residual_cnn', or 'convnext_1d'")

    # GRU Backbone
    flattened_output = build_gru_backbone(encoder_output, sequence_input)

    # Classifier Head
    if stage == 'p12':
        probs = build_classifier_head_p12(flattened_output, stage=stage)
        inputs = sequence_input
        model_name = f'E4_{cnn_variant}_{stage}_{loss_mode}'
    elif stage == 'biofusion':
        bio_input = Input(shape=(bio_feature_dim,), name='bio_input')
        probs = build_biofusion_head(flattened_output, bio_input, bio_feature_dim)
        inputs = [sequence_input, bio_input]
        model_name = f'E4_{cnn_variant}_{stage}_{loss_mode}'
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'p12' or 'biofusion'")

    # Build model
    model = Model(inputs=inputs, outputs=probs, name=model_name)

    # Loss function
    if loss_mode == 'control':
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=False)
    elif loss_mode == 'focal_bce':
        loss_fn = FocalBCELoss(gamma=focal_gamma, alpha=focal_alpha, from_logits=False)
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}. Must be 'control' or 'focal_bce'")

    # Compile
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


def count_cnn_parameters(cnn_variant='p12_control', convnext_kernel_size=7):
    """
    Count parameters in CNN encoder only (for reporting).

    Args:
        cnn_variant: 'p12_control', 'residual_cnn', or 'convnext_1d'
        convnext_kernel_size: kernel size for ConvNeXt (5 or 7)

    Returns:
        dict with parameter counts
    """
    sequence_input = Input(shape=(1, 24, 7))

    if cnn_variant == 'p12_control':
        encoder_output = build_p12_control_encoder(sequence_input)
    elif cnn_variant == 'residual_cnn':
        encoder_output = build_residual_cnn_encoder(sequence_input)
    elif cnn_variant == 'convnext_1d':
        encoder_output = build_convnext_1d_encoder(sequence_input, kernel_size=convnext_kernel_size)
    else:
        raise ValueError(f"Unknown cnn_variant: {cnn_variant}")

    temp_model = Model(inputs=sequence_input, outputs=encoder_output)

    total_params = temp_model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in temp_model.trainable_weights])

    return {
        'cnn_variant': cnn_variant,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'output_shape': tuple(encoder_output.shape[1:])
    }


def get_shared_layer_names(stage):
    """
    Get list of shared layer names (GRU + head, excluding encoder).

    Args:
        stage: 'p12' or 'biofusion'

    Returns:
        list of layer names
    """
    if stage == 'p12':
        # P12 stage: GRU + classifier head
        return [
            'bigru_30', 'bigru_20', 'bigru_10',
            'head_dense_128', 'head_dense_64', 'head_dropout',
            'prob_output'
        ]
    else:  # biofusion
        # Biofusion stage: GRU + sequence head + bio branch + fusion head
        return [
            'bigru_30', 'bigru_20', 'bigru_10',
            'seq_dense_128', 'seq_dense_64', 'seq_dropout',
            'bio_dense', 'bio_norm', 'bio_dropout',
            'fusion', 'fusion_dense', 'fusion_dropout',
            'prob_output'
        ]


def capture_shared_weights(model, stage):
    """
    Capture immutable copies of shared weights from model.

    Args:
        model: model to capture weights from
        stage: 'p12' or 'biofusion'

    Returns:
        dict mapping layer_name -> list of numpy arrays (immutable copies)

    Raises:
        RuntimeError if shared layers not found
    """
    import numpy as np

    shared_names = get_shared_layer_names(stage)
    captured = {}

    for layer_name in shared_names:
        layer = None
        for l in model.layers:
            if l.name == layer_name:
                layer = l
                break

        if layer is None:
            raise RuntimeError(f"Shared layer '{layer_name}' not found in model")

        # Get weights and make immutable copies
        weights = layer.get_weights()
        captured[layer_name] = [np.array(w, copy=True) for w in weights]

    return captured


def apply_shared_weights(model, shared_weights, stage):
    """
    Apply captured shared weights to model.

    Args:
        model: model to receive weights
        shared_weights: dict from capture_shared_weights()
        stage: 'p12' or 'biofusion'

    Raises:
        RuntimeError if shape mismatch or layers not found
    """
    import numpy as np

    shared_names = get_shared_layer_names(stage)

    for layer_name in shared_names:
        if layer_name not in shared_weights:
            raise RuntimeError(f"Layer '{layer_name}' not in captured shared_weights")

        layer = None
        for l in model.layers:
            if l.name == layer_name:
                layer = l
                break

        if layer is None:
            raise RuntimeError(f"Shared layer '{layer_name}' not found in target model")

        captured_weights = shared_weights[layer_name]
        current_weights = layer.get_weights()

        # Verify shape compatibility
        if len(captured_weights) != len(current_weights):
            raise RuntimeError(f"Layer '{layer_name}': weight count mismatch "
                               f"(captured={len(captured_weights)}, current={len(current_weights)})")

        for i, (cw, tw) in enumerate(zip(captured_weights, current_weights)):
            if cw.shape != tw.shape:
                raise RuntimeError(f"Layer '{layer_name}' weight {i}: shape mismatch "
                                   f"(captured={cw.shape}, current={tw.shape})")

        # Apply weights
        layer.set_weights(captured_weights)


def hash_shared_weights(shared_weights):
    """
    Compute hash of captured shared weights.

    Args:
        shared_weights: dict from capture_shared_weights()

    Returns:
        hash string (16 chars)
    """
    import hashlib

    # Sort by layer name for consistency (canonical order)
    sorted_names = sorted(shared_weights.keys())

    weights_bytes = b''
    for layer_name in sorted_names:
        for w in shared_weights[layer_name]:
            weights_bytes += w.tobytes()

    return hashlib.md5(weights_bytes).hexdigest()[:16]


def get_shared_weights_hash_from_model(model, stage):
    """
    Get hash of shared weights from model.

    Uses capture_shared_weights + hash_shared_weights for consistency.

    Args:
        model: Keras model
        stage: 'p12' or 'biofusion'

    Returns:
        hash string (16 chars)
    """
    captured = capture_shared_weights(model, stage)
    return hash_shared_weights(captured)


def verify_shared_weights_match(model, shared_weights, stage):
    """
    Verify that model's shared weights match captured weights.

    Args:
        model: model to verify
        shared_weights: dict from capture_shared_weights()
        stage: 'p12' or 'biofusion'

    Returns:
        True if all match

    Raises:
        RuntimeError if weights do not match
    """
    import numpy as np

    shared_names = get_shared_layer_names(stage)

    for layer_name in shared_names:
        if layer_name not in shared_weights:
            raise RuntimeError(f"Layer '{layer_name}' not in captured shared_weights")

        layer = None
        for l in model.layers:
            if l.name == layer_name:
                layer = l
                break

        if layer is None:
            raise RuntimeError(f"Shared layer '{layer_name}' not found in model")

        captured_weights = shared_weights[layer_name]
        current_weights = layer.get_weights()

        if len(captured_weights) != len(current_weights):
            raise RuntimeError(f"Layer '{layer_name}': weight count mismatch")

        for i, (cw, tw) in enumerate(zip(captured_weights, current_weights)):
            if not np.array_equal(cw, tw):
                raise RuntimeError(f"Layer '{layer_name}' weight {i}: weights do not match")

    return True
