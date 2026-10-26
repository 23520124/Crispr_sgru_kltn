import os
import sys
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Bidirectional, Concatenate, GRU, Dense, Flatten, 
    Conv2D, Reshape, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AsymmetricFocalLoss(tf.keras.losses.Loss):
    """
    Asymmetric Focal Loss (ASL) chuyên trị mất cân bằng cực đoan.
    - gamma_pos: Giảm phạt các mẫu dương khó để gradient hội tụ tốt.
    - gamma_neg: Tăng phạt lũy thừa đối với các mẫu âm tính dễ.
    - clip: Ngưỡng biên triệt tiêu gradient của các mẫu âm có p < clip.
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.03, eps=1e-7, from_logits=True, name="asymmetric_focal_loss"):
        super(AsymmetricFocalLoss, self).__init__(name=name)
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        
        if self.from_logits:
            p = tf.math.sigmoid(y_pred)
        else:
            p = y_pred

        # 1. Thành phần Loss cho mẫu dương (y = 1)
        p_pos = tf.clip_by_value(p, self.eps, 1.0 - self.eps)
        loss_pos = - y_true * tf.math.pow(1.0 - p_pos, self.gamma_pos) * tf.math.log(p_pos)

        # 2. Thành phần Loss cho mẫu âm (y = 0) có cơ chế dịch chuyển xác suất bất đối xứng
        p_neg = tf.maximum(p - self.clip, 0.0)
        p_neg = tf.clip_by_value(p_neg, 0.0, 1.0 - self.eps)
        loss_neg = - (1.0 - y_true) * tf.math.pow(p_neg, self.gamma_neg) * tf.math.log(1.0 - p_neg + self.eps)

        return tf.reduce_mean(loss_pos + loss_neg)


def Crispr_SGRU(learning_rate=0.0001, gamma_neg=4.0, gamma_pos=1.0, clip=0.03):
    """
    Kiến trúc Crispr_SGRU nâng cấp cho Giai đoạn 1:
    - Đầu ra Dense(1) dạng tuyến tính (Logit thuần)
    - Tối ưu hóa bằng Asymmetric Focal Loss
    - Theo dõi PR-AUC trực tiếp trong quá trình huấn luyện
    """
    inputs_1 = Input(shape=(1, 24, 7), name='main_input')
    
    # Khối tích chập Inception đa trường tiếp nhận
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs_1)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs_1)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs_1)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs_1)
    conv_output = tf.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((24, 40))(conv_output)
    
    # 3 tầng BiGRU xếp chồng có kết nối skip-connections
    x0 = Bidirectional(GRU(30, return_sequences=True))(conv_output)
    inputs_2 = Reshape((24, 7))(inputs_1)
    x = Concatenate(axis=2)([inputs_2, x0])

    x1 = Bidirectional(GRU(20, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])
    
    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=-1)([x0, x1, x2])
    
    # Phân loại
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.35)(x)
    
    # Giai đoạn 1: Xuất Logit 1 chiều duy nhất, không dùng Sigmoid tại đây
    logits = Dense(1, activation=None, name='logit_output')(x) 

    model = Model(inputs=inputs_1, outputs=logits)
    
    opt = Adam(learning_rate=learning_rate)
    loss_fn = AsymmetricFocalLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, from_logits=True)
    
    model.compile(
        loss=loss_fn,
        optimizer=opt,
        metrics=[
            tf.keras.metrics.AUC(curve='PR', name='pr_auc', from_logits=True),
            tf.keras.metrics.AUC(curve='ROC', name='auroc', from_logits=True)
        ]
    )
    return model


if __name__ == '__main__':
    dummy_input = tf.ones((2, 1, 24, 7))
    model = Crispr_SGRU()
    out = model(dummy_input)
    print("Mô hình khởi tạo thành công!")
    print(f"Shape đầu ra Logit: {out.shape}")  # Phải là (2, 1)
    model.summary()