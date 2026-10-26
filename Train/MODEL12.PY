import os
import sys
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Bidirectional, Concatenate, GRU, Dense, Flatten, 
    Conv2D, Reshape, Dropout
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BCEDiceLoss(tf.keras.losses.Loss):
    """
    Kết hợp Binary Cross-Entropy và Soft-Dice Loss:
    - Soft-Dice Loss giải quyết mất cân bằng mà không bị lệch thiên kiến cực đoan.
    - BCE giữ gradient ổn định, ngăn chặn hiện tượng đẩy toàn bộ xác suất về 1.
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

        # 1. Binary Cross-Entropy Loss
        loss_bce = self.bce(y_true, y_pred if self.from_logits else p)

        # 2. Soft-Dice Loss đối xứng (Symmetric)
        y_true_f = tf.reshape(y_true, [-1])
        p_f = tf.reshape(p, [-1])

        intersection = tf.reduce_sum(y_true_f * p_f)
        denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(p_f) + self.eps
        loss_dice = 1.0 - (2.0 * intersection + self.eps) / denominator

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def Crispr_SGRU(learning_rate=0.0001):
    """
    Backbone Crispr-SGRU chuẩn: Inception + Stacked BiGRU + Flatten.
    Đầu ra Sigmoid 1 node ổn định.
    """
    inputs_1 = Input(shape=(1, 24, 7), name='main_input')
    
    # Inception module
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs_1)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs_1)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs_1)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs_1)
    conv_output = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((24, 40))(conv_output)
    
    # 3 tầng BiGRU nối tầng (Skip connections)
    x0 = Bidirectional(GRU(30, return_sequences=True))(conv_output)
    inputs_2 = Reshape((24, 7))(inputs_1)
    x = Concatenate(axis=2)([inputs_2, x0])

    x1 = Bidirectional(GRU(20, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])
    
    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=-1)([x0, x1, x2])
    
    # Tầng Flatten bảo toàn vị trí 24 base
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.35)(x)
    
    # Đầu ra Sigmoid xác suất trực tiếp
    probs = Dense(1, activation='sigmoid', name='prob_output')(x)

    model = Model(inputs=inputs_1, outputs=probs)
    
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


if __name__ == '__main__':
    dummy_input = tf.ones((2, 1, 24, 7))
    model = Crispr_SGRU()
    out = model(dummy_input)
    print("Mô hình khởi tạo thành công! Shape đầu ra:", out.shape)
    model.summary()