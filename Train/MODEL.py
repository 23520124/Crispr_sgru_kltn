#
import os
import sys
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Multiply, BatchNormalization, Concatenate, GRU, Dense, Flatten, Conv2D, Reshape, Dropout, Add, Activation, MaxPooling2D, Conv1D, Layer, Embedding, Lambda, MaxPool1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPool1D, Attention, MultiHeadAttention, LayerNormalization, MaxPooling1D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model

# Add path to loss_generator module
sys.path.append(os.path.join(os.path.dirname(__file__), "Model Train"))
import loss_generator as lg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#

# import os
# import tensorflow as tf
# from tensorflow.keras import Input, Model
# from tensorflow.keras.layers import LSTM, Bidirectional, Multiply, BatchNormalization, Concatenate, GRU, Dense, Flatten,Conv2D,Reshape
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Nadam
# from tensorflow.keras import backend as K
# import loss_generator as lg

def Crispr_SGRU():
    #inputs = Input(shape=(24, 7))
    inputs_1 = Input(shape=(1, 24, 7), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs_1)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs_1)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs_1)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs_1)
    conv_output = tf.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((24, 40))(conv_output)
    x0 = Bidirectional(GRU(30, return_sequences=True))(conv_output)
    inputs_2=Reshape((24,7))(inputs_1)
    x = Concatenate(axis=2)([inputs_2, x0])

    x1 = Bidirectional(GRU(20, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])
    
    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=2)([x1, x2])

    x = Concatenate(axis=-1)([x0, x1, x2])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.35)(x)
    x = Dense(2, activation="sigmoid")(x) 

    model = Model(inputs=inputs_1, outputs=x)
    print(model.summary())
    opt = Adam(learning_rate=0.0001)
    # opt = SGD(learning_rate=0.0001)
    # model.compile(loss=lg.dice_loss(), optimizer=opt, metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    # model.compile(loss=lg.asymmetric_focal_loss(), optimizer=opt, metrics=['accuracy'])
    return model

def Student_model():
    Input1 = Input(shape=(1, 24, 16), name='student_input')
    Inputs= Reshape((24, 16))(Input1)
    x0 = Bidirectional(GRU(10, return_sequences=True))(Inputs)
    x = Concatenate(axis=2)([Inputs, x0])

    x1 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])
    
    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=2)([x1, x2])

    x = Concatenate(axis=-1)([x0, x1, x2])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate=0.35)(x)
    output = Dense(2, activation="sigmoid")(x)
    student_model = Model(inputs=Input1, outputs=output)
    opt = Adam(learning_rate=0.0001)
    print(student_model.summary())
    student_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    return student_model

class MulitHeadAttention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(MulitHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(MulitHeadAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):

        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, Q_seq.shape[1], self.nb_head, self.size_per_head))  # B,N,heads,heads_dim
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))  # B,N,heads,heads_dim -> B,heads,N,heads_dim

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K_seq.shape[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, V_seq.shape[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        # 计算内积，然后mask，然后softmax
        # B,heads,N,heads_dim @ B,heads,heads_dim,N ->  B,heads,N,N
        A = (Q_seq @ K.permute_dimensions(K_seq, (0, 1, 3, 2))) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))  # B,heads,N,N -> B,N,N,heads
        A = self.Mask(A, V_len, 'add')

        A = K.permute_dimensions(A, (0, 3, 2, 1))  # B,N,N,heads -> B,heads,N,N
        A = K.softmax(A)
        # 输出并mask
        # B,heads,N,N @ B,heads,N,heads_dim -> N,heads,N,heads_dim
        O_seq = A @ V_seq
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, O_seq.shape[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim, 'nb_head': self.nb_head, 'size_per_head': self.size_per_head}
        base_config = super(MulitHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def transformerBlock(inputs, head, size_per_head):
    x = Dense(8, activation='relu')(inputs)
    x1 = MulitHeadAttention(head, size_per_head)([x, x, x])
    x2 = Dropout(0.5)(x1)
    x3 = Add()([x2, x])

    x4 = Dense(32, activation='relu')(x3)
    x5 = Dropout(0.5)(x4)
    x6 = Dense(8)(x5)
    x7 = Dropout(0.5)(x6)
    x8 = Add()([x3, x7])

    y4 = MulitHeadAttention(head, size_per_head)([x3, x8, x8])
    y5 = Dropout(0.5)(y4)
    y6 = Add()([x3, y5])

    y7 = Dense(32, activation='relu')(y6)
    y8 = Dropout(0.5)(y7)
    y9 = Dense(8)(y8)
    y10 = Dropout(0.5)(y9)
    y10 = Add()([y6, y10])
    return y10


def AttnToMismatch():
    sg_input = Input(name='sg_input', shape=(23,))
    sg_emd = Embedding(16, 8, input_length=23)(sg_input)
    pos_input = Input(name='sg_pos_input', shape=(23,))
    pos_emd = Embedding(23, 8, input_length=23)(pos_input)

    emd = Add()([sg_emd, pos_emd])
    emd = Dropout(0.5)(emd)

    trans = transformerBlock(emd, 4, 2)

    d_output = Lambda(lambda x: K.expand_dims(x, axis=3))(trans)

    cnn_1 = Conv2D(32, kernel_size=(3, 1), kernel_initializer='glorot_uniform', padding='same')
    C1 = cnn_1(d_output)
    maxpool_1 = MaxPooling2D(pool_size=(2, 1), padding='same')
    P1 = maxpool_1(C1)
    A1 = Activation('relu')(P1)
    print(A1.shape)

    cnn_2 = Conv2D(64, kernel_size=(3, 1), kernel_initializer='glorot_uniform', padding='same')
    C2 = cnn_2(A1)
    maxpool_2 = MaxPooling2D(pool_size=(2, 1))
    P2 = maxpool_2(C2)
    A2 = Activation('relu')(P2)
    print(A2.shape)

    flat = Flatten()(A2)
    drop1 = Dropout(0.5)(flat)
    dense1 = Dense(100, activation='elu')(drop1)
    drop2 = Dropout(0.5)(dense1)
    out = Dense(2, activation='sigmoid', name='output')(drop2)

    model = Model(inputs=[sg_input, pos_input], outputs=[out])
    # model.summary()
    optimizer = Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.98, epsilon=1e-9, decay=0.00001)
    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['acc'])
    return model

def CnnCrispr(model_ini):
    # The benchmark model CnnCrispr for Classification schema
    print("model1 loaded with 1 biLSTM, 5 conv and 2 dense")
    model_message = "Dropout 0.3,biLSTM.40, Conv1D.[10,20,40,80,100],  Dense[20,2], BatchNormalization,Activition='relu'"
    model = model_ini
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))
    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    # print(model.summary())
    return model, model_message

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True,
              name=None, trainable=True):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name, trainable=trainable)(x)

    # x = BatchNormalization(axis=-1,scale=True)(x)
    if activation is not None:#activation='relu'
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def CRISPR_Net_model():#模型定义
    inputs = Input(shape=(1, 24, 7), name='main_input')
    branch_0 = conv2d_bn(inputs, 10, (1, 1))
    branch_1 = conv2d_bn(inputs, 10, (1, 2))
    branch_2 = conv2d_bn(inputs, 10, (1, 3))
    branch_3 = conv2d_bn(inputs, 10, (1, 5))
    branches = [inputs, branch_0, branch_1, branch_2, branch_3]

    mixed = Concatenate(axis=-1)(branches)
    mixed = Reshape((24, 47))(mixed)
    blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(24, 47)))(mixed)
    x=Flatten()(blstm_out)
    x = Dense(80, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dropout(rate=0.35)(x)
    prediction = Dense(2, activation='sigmoid', name='main_output')(x)
    model = Model(inputs, prediction)
    print(model.summary())
    adam_opt = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    return model

def crispr_ip():
    #参数为（scale=1.0,mode="fan_in",distribution="normal",seed=None，dtype=dtypes.float32)
    #scale：缩放尺度，mode：如果mode = "fan_in"， n为输入单元的结点数；如果mode = "fan_out"，n为输出单元的结点数；如果mode = "fan_avg",n为输入和输出单元结点数的平均值。
    #当 distribution="normal" 的时候，生成truncated normal distribution（截断正态分布） 的随机数，当distribution="uniform”的时候 ，生成均匀分布的随机数
    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    input_shape=(1, 24, 7)
    input_value = Input(shape=input_shape)
    #Conv2D：(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, kernel_initializer='glorot_uniform')
    #filters：卷积核的数目（即输出的维度）,kernel_size：卷积核的宽度和长度 ,data_format:kernel_initializer ：卷积核初始化
    #（样本数，通道数，行或称为高，列或称为宽）通道在前的方式NCHW，称为channels_first；而TensorFlow使用（样本数，行或称为高，列或称为宽，通道数）通道在后的方式NHWC，称为channels_last。
    #N-batch number， H height size, W width size； C channel number  NHWC NCHW
    conv_1_output = Conv2D(60, (1,input_shape[-1]), padding='valid', data_format='channels_last', kernel_initializer=initializer)(input_value)
    conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(conv_1_output)
    conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_last')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPool1D(data_format='channels_last')(conv_1_output_reshape2)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
    attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])#？？？一般会对attention层进行定义
    average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
    max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
    concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
    flatten_output = Flatten()(concat_output)
    linear_1_output = BatchNormalization()(Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
    linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
    linear_2_output_dropout = Dropout(0.9)(linear_2_output)
    linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
    model = Model(input_value, linear_3_output)
    model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model.compile(tf.keras.optimizers.Adam(), loss=dice_loss(), metrics=['accuracy'])
    print(model.summary())
    return model

def  CRISPR_M(VOCABULARY_SIZE=30, MAX_STEPS=24, EMBED_SIZE=7):
    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_SIZE)
    position_encoding = PositionalEncoding(max_steps=MAX_STEPS, max_dims=EMBED_SIZE)

    inputs_1 = Input(shape=(MAX_STEPS,), name="input_1")
    embeddings_1 = embedding(inputs_1)
    positional_encoding_1 = position_encoding(embeddings_1)
    inputs_2 = Input(shape=(MAX_STEPS,), name="input_2")
    embeddings_2 = embedding(inputs_2)
    positional_encoding_2 = position_encoding(embeddings_2)
    inputs_3 = Input(shape=(MAX_STEPS,), name="input_3")
    embeddings_3 = embedding(inputs_3)
    positional_encoding_3 = position_encoding(embeddings_3)

    attention_1 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_1, positional_encoding_1)
    attention_2 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_2, positional_encoding_2)
    attention_3 = MultiHeadAttention(num_heads=8, key_dim=6)(positional_encoding_3, positional_encoding_3)
    # attention_out = MultiHeadAttention(num_heads=8, key_dim=6)(attention_2, attention_3, attention_1)
    # attention_out = Reshape(tuple([1, 24, EMBED_SIZE]))(attention_out)

    branch_1 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_1 = Dropout(0.2)(branch_1)
    conv_1 = Conv2D(32, (1,4), activation='relu', padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(0.2)(conv_1)
    conv_1 = Conv2D(64, (1,4), activation='relu', padding='valid')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Reshape(tuple([x for x in conv_1.shape.as_list() if x != 1 and x is not None]))(conv_1)
    conv_1_average = AveragePooling1D(data_format='channels_first')(conv_1)
    conv_1_max = MaxPooling1D(data_format='channels_first')(conv_1)
    conv_1 = Concatenate(axis=-1)([conv_1_average, conv_1_max])
    bidirectional_1_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_1)
    flatten_1 = Flatten()(bidirectional_1_output)

    branch_2 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_1)
    conv_2 = Dropout(0.2)(branch_2)
    conv_2 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Reshape(tuple([x for x in conv_2.shape.as_list() if x != 1 and x is not None]))(conv_2)
    conv_2_average = AveragePooling1D(data_format='channels_first')(conv_2)
    conv_2_max = MaxPooling1D(data_format='channels_first')(conv_2)
    conv_2 = Concatenate(axis=-1)([conv_2_average, conv_2_max])
    bidirectional_2_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_2)
    flatten_2 = Flatten()(bidirectional_2_output)

    branch_3 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_2)
    conv_3 = Dropout(0.2)(branch_3)
    conv_3 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Reshape(tuple([x for x in conv_3.shape.as_list() if x != 1 and x is not None]))(conv_3)
    bidirectional_3_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_3)
    flatten_3 = Flatten()(bidirectional_3_output)

    branch_4 = Reshape(tuple([24, EMBED_SIZE, 1]))(attention_3)
    conv_4 = Dropout(0.2)(branch_4)
    conv_4 = Conv2D(64, (1,7), activation='relu', padding='valid')(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Reshape(tuple([x for x in conv_4.shape.as_list() if x != 1 and x is not None]))(conv_4)
    bidirectional_4_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(conv_4)
    flatten_4 = Flatten()(bidirectional_4_output)

    # con = Concatenate(axis=-1)([bidirectional_1_output, bidirectional_2_output, bidirectional_3_output, bidirectional_4_output])
    con = Concatenate(axis=-1)([flatten_1, flatten_2, flatten_3, flatten_4])
    main = Flatten()(con)
    main = Dropout(0.2)(main)
    main = Dense(256, activation='relu')(main)
    main = BatchNormalization()(main)
    main = Dropout(0.2)(main)
    main = Dense(64, activation='relu')(main)
    main = BatchNormalization()(main)
    main = Dropout(0.8)(main)
    outputs = Dense(2, activation='sigmoid', name='output')(main)
    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    opt = Adam(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def Crispr_DNT(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain, xtest, ytest, input_shape,
                      num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    input_value = Input(shape=(1, 24, 16))

    conv_1_output = Conv2D(64, (1, 1), activation='relu', padding='valid',
                           data_format='channels_last',
                           kernel_initializer=initializer)(input_value)
    print(conv_1_output.shape)
    conv_1_output = BatchNormalization()(conv_1_output)
    conv_1_output_reshape = Reshape(
        tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
        conv_1_output)
    print(conv_1_output_reshape.shape)
    conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
    print(conv_1_output_reshape2.shape)
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_last')(conv_1_output_reshape2)

    conv_1_output_reshape_average = tf.transpose(conv_1_output_reshape_average, perm=[0, 2, 1])

    print(conv_1_output_reshape_average.shape)
    conv_1_output_reshape_max = MaxPool1D(data_format='channels_last')(conv_1_output_reshape2)
    conv_1_output_reshape_max = tf.transpose(conv_1_output_reshape_max, perm=[0, 2, 1])
    print(conv_1_output_reshape_max.shape)
    input_value1 = Reshape((24, 16))(input_value)
    bidirectional_1_output = Bidirectional(
        LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
        Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

    bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
    pos_embedding = PositionalEncoding(sequence_len=24, embedding_dim=64)(bidirectional_1_output_ln)
    # print(pos_embedding.shape)
    attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
    print(attention_1_output.shape)
    residual1 = attention_1_output + pos_embedding
    print('residual1.shape')
    print(residual1.shape)
    laynorm1 = LayerNormalization()(residual1)
    linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
    linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
    residual2 = laynorm1 + linear2
    laynorm2 = LayerNormalization()(residual2)
    print(laynorm2.shape)
    attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
    residual3 = attention_2_output + laynorm2
    laynorm3 = LayerNormalization()(residual3)
    linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
    linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
    residual4 = laynorm3 + linear4
    laynorm4 = LayerNormalization()(residual4)
    print(laynorm4.shape)

    flatten_output = Flatten()(laynorm4)
    linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))

    linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
    linear_2_output_dropout = Dropout(0.25)(linear_2_output)
    linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
    model = Model(input_value, linear_3_output)
    model.compile(tf.keras.optimizers.Adam(0.001), loss=GCE(), metrics=['accuracy'])  # Adam是0.001，SGD是0.01

    return model
if __name__ == '__main__':
    # (b,24,512)
    train= tf.ones((1,1, 24, 16))
    label = tf.ones((1, 2))
    test=tf.ones((1, 1,24, 16))
    labels=tf.ones((1, 2))
    model = Student_model()
    model=Crispr_SGRU()
    # plot_model(model, to_file='model.png', show_shapes=True)
    # model.save('model.png')
    # from matplotlib import pyplot as plt
    # history=model.fit(x=train, y=label,
    #                   validation_data=(test,labels),
    #                   epochs=10)
    
