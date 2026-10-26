#使用CRISPR-net编码
import os
import random
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow import keras

from keras import Model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Encoder_sgRNA_off
from MODEL import Crispr_SGRU
from sklearn.metrics import mean_squared_error, roc_curve, precision_recall_curve, auc,matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import ReduceLROnPlateau

def load_data(file_path):
    data_list = []
    label = []
    Negative = []
    Positive = []
    sgRNA=[]
    with open(file_path) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]  # strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。
            label_item = np.float64(ll[2])
            data_item = [int(i) for i in ll[3:]]
            if label_item == 0.0:
                Negative.append(ll)
                sgRNA.append(ll[0])
            else:
                Positive.append(ll)
                sgRNA.append(ll[0])
            data_list.append(data_item)
            label.append(label_item)
    return Negative, Positive, label,sgRNA


def train_flow(Train_Negative, Train_Positive, batchsize):
    train_Negative = Train_Negative
    train_Positive = Train_Positive
    Num_Positive = len(train_Positive)
    Num_Negative = len(train_Negative)
    Index_Negative = [i for i in range(Num_Negative)]
    np.random.seed(2020)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')  # batchsize256
    # numpy.random.randint(low, high=None, size=None, dtype='l')返回一个随机整型数，范围从低（包括）到高（不包括）
    # Index_Positive=[0~Num_Positive,共256个]
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    num_counter = 0
    X_input = []
    Y_input = []
    i = 0
    while True:
        i += 1
        np.random.seed(i)
        for i in range(Total_num_batch):
            for j in range(batchsize):
                X_input.append(train_Negative[Index_Negative[j + i * batchsize]])
                Y_input.append(0)
                X_input.append(train_Positive[Index_Positive[j]])
                Y_input.append(1)
                num_counter += 1
                if num_counter == batchsize:
                    Y_input = np_utils.to_categorical(Y_input)  # 把类别标签转换为onehot编码
                    yield (np.array(X_input), np.array(Y_input))
                    X_input = []
                    Y_input = []
                    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                    num_counter = 0

def valid_flow(Test_Negative, Test_Positive, batchsize):
    valid_Negative = Test_Negative
    valid_Positive = Test_Positive
    Num_Positive = len(valid_Positive)
    Num_Negative = len(valid_Negative)
    Index_Negative = [i for i in range(Num_Negative)]

    np.random.seed(2020)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')

    random.shuffle(Index_Negative)
    num_counter = 0
    X_input = []
    Y_input = []
    i = 0
    while True:
        i += 1
        np.random.seed(i)
        for j in range(batchsize):
            X_input.append(valid_Negative[Index_Negative[j]])
            Y_input.append(0)
            X_input.append(valid_Positive[Index_Positive[j]])
            Y_input.append(1)
            num_counter += 1
            if num_counter == batchsize:
                Y_input = np_utils.to_categorical(Y_input)
                yield (np.array(X_input), np.array(Y_input))
                X_input = []
                Y_input = []
                Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                num_counter = 0

def encording(data):
    encode=[]
    for idx, row in data.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        label = row[2]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        encode.append(en.on_off_code)
    return encode

def encordingXtest(Xtest):
    final_code = []
    for idx, row in Xtest.iterrows():
        on_seq = row[0]
        off_seq = row[1]
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        final_code.append(en.on_off_code)
    return final_code



def load_data_delete_sgRNA(file_path,sgrna):
    
    Negative = []
    Positive = []
    count=0
    unique_sgrna = set(sgrna) 
    with open(file_path) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]  # strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。
            label_item = np.float64(ll[2])
            if ll[0] in unique_sgrna:
                if label_item == 0.0:
                    Negative.append(ll)
                else:
                    Positive.append(ll)
                unique_sgrna.remove(ll[0])
                print(len(unique_sgrna))
                count=count+1
            label.append(label_item)
    return Negative, Positive, label,count

def load_data_leave_sgRNA(file_path,sgRNA):
    data_list = []
    label = []
    Negative = []
    Positive = []
    Test=[]
    with open(file_path) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]  # strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。
            label_item = np.float64(ll[2])
            data_item = [int(i) for i in ll[3:]]
            if ll[0]!=sgRNA:
                if label_item == 0.0:
                    Negative.append(ll)
                else:
                    Positive.append(ll)
            else:
                Test.append(ll)
            data_list.append(data_item)
            label.append(label_item)
    return Negative, Positive, label,Test

data_name = "II5"
Negative, Positive, label ,sgRNA = load_data('off-target datasets/leave sgrna out/'+data_name+'.csv')
print(len(sgRNA))
unique_sgRNA = set(sgRNA)
print(len(unique_sgRNA))
test_sgRNA = random.sample(unique_sgRNA, 1)[0]
unique_sgRNA = [x for x in unique_sgRNA if x != test_sgRNA]
print(test_sgRNA)
# test_sgRNA='GCTGCAGAAGGGATTCCATGAGG'
Negative, Positive, label ,Xtest=load_data_leave_sgRNA('off-target datasets/leave sgrna out/'+data_name+'.csv',test_sgRNA)

k=0
while(k!=1):
    label_1=0
    label_0=0             
    if len(Xtest)>30:
        for item in Xtest:
            # print(item[-1])
            if item[-1]==str(1):
                label_1=label_1+1
            if item[-1]==str(0):
                label_0=label_0+1 
        print("1:",label_1)
        print("0:",label_0)
        if label_1==0 or label_0==0:
            unique_sgRNA = [x for x in unique_sgRNA if x != test_sgRNA]
            test_sgRNA = random.sample(unique_sgRNA, 1)[0]
            Negative, Positive, label ,Xtest=load_data_leave_sgRNA('off-target datasets/leave sgrna out/'+data_name+'.csv',test_sgRNA) 
            
        else:
            print('test_sgRNA:',test_sgRNA)
            k=1      
    else:
        unique_sgRNA = [x for x in unique_sgRNA if x != test_sgRNA]
        test_sgRNA = random.sample(unique_sgRNA, 1)[0]
        Negative, Positive, label ,Xtest=load_data_leave_sgRNA('off-target datasets/leave sgrna out/'+data_name+'.csv',test_sgRNA)


Positive, Negative = np.array(Positive), np.array(Negative)
# Train_Vilidation_Negative, Test_Negative = train_test_split(Negative, test_size=0.15, random_state=42)
# Train_Vilidation_Positive, Test_Positive = train_test_split(Positive, test_size=0.15, random_state=42)
# Train_Negative, Vilidation_Negative = train_test_split(Train_Vilidation_Negative, test_size=0.2, random_state=42)
# Train_Positive, Vilidation_Positive = train_test_split(Train_Vilidation_Positive, test_size=0.2, random_state=42)
Train_Negative, Vilidation_Negative = train_test_split(Negative, test_size=0.2, random_state=42)#leave one sgrna out
Train_Positive, Vilidation_Positive = train_test_split(Positive, test_size=0.2, random_state=42)

Train_Positive = pd.DataFrame(Train_Positive)
Train_Negative = pd.DataFrame(Train_Negative)
Vilidation_Positive = pd.DataFrame(Vilidation_Positive)
Vilidation_Negative = pd.DataFrame(Vilidation_Negative)

Train_Positive=pd.DataFrame(Train_Positive)
Train_Negative=pd.DataFrame(Train_Negative)
Vilidation_Positive=pd.DataFrame(Vilidation_Positive)
Vilidation_Negative=pd.DataFrame(Vilidation_Negative)

Train_Positive_oneHot = []
Train_Negative_oneHot = []
Vilidation_Positive_oneHot=[]
Vilidation_Negative_oneHot=[]

Train_Positive_oneHot=encording(Train_Positive)
Train_Negative_oneHot=encording(Train_Negative)
Vilidation_Positive_oneHot=encording(Vilidation_Positive)
Vilidation_Negative_oneHot=encording(Vilidation_Negative)
Train_Positive_oneHot = np.array(Train_Positive_oneHot)
Train_Negative_oneHot = np.array(Train_Negative_oneHot)
Vilidation_Positive_oneHot = np.array(Vilidation_Positive_oneHot)
Vilidation_Negative_oneHot = np.array(Vilidation_Negative_oneHot)

BATCH_SIZE = 256
TEST_SIZE = 100

AUROC = []
PRAUC = []
F1_SCORE = []
PRECISION = []
RECALL = []
MCC = []
NUM_BATCH = int(len(Train_Negative) / BATCH_SIZE)
# Xtest = np.vstack((Test_Negative, Test_Positive))  # 按垂直方向（行顺序）堆叠数组构成一个新的数组
Seq = [i for i in range(len(Xtest))]  # 列表解析
Xtest = np.array(Xtest)
random.shuffle(Seq)  # 将序列中所有元素随机排序
Xtest = Xtest[Seq]  # 将样本顺序打乱
Xtest = np.array(Xtest)

num = 5
epochs = 30
model=gru.My_model()

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
# keras.utils.plot_model(model, to_file='LSTM/code/model.png', show_shapes=True, dpi=600)
for i in range(num):
    print("processing fold #", i + 1)
    Train_Negative_oneHot = Train_Negative_oneHot.astype("float64")
    Train_Positive_oneHot = Train_Positive_oneHot.astype("float64")
    Vilidation_Negative_oneHot = Vilidation_Negative_oneHot.astype("float64")
    Vilidation_Positive_oneHot = Vilidation_Positive_oneHot.astype("float64")
    # print(Train_Positive_oneHot.shape)
    # Train_Positive_oneHot = Train_Positive_oneHot.reshape((len(Train_Positive_oneHot), 24, 7))
    # Train_Negative_oneHot = Train_Negative_oneHot.reshape((len(Train_Negative_oneHot), 24, 7))
    # Vilidation_Positive_oneHot = Vilidation_Positive_oneHot.reshape((len(Vilidation_Positive_oneHot), 24, 7))
    # Vilidation_Negative_oneHot = Vilidation_Negative_oneHot.reshape((len(Vilidation_Negative_oneHot), 24, 7))
    Train_Positive_oneHot = Train_Positive_oneHot.reshape((len(Train_Positive_oneHot), 1, 24, 7))
    Train_Negative_oneHot= Train_Negative_oneHot.reshape((len(Train_Negative_oneHot), 1, 24, 7))
    Vilidation_Positive_oneHot = Vilidation_Positive_oneHot.reshape((len(Vilidation_Positive_oneHot), 1, 24, 7))
    Vilidation_Negative_oneHot = Vilidation_Negative_oneHot.reshape((len(Vilidation_Negative_oneHot), 1, 24, 7))
    # print(Train_Positive_oneHot.shape)
    # print(Train_Positive_oneHot)
    history = model.fit_generator(train_flow(Train_Negative_oneHot, Train_Positive_oneHot, BATCH_SIZE),
                                  validation_data=valid_flow(Vilidation_Negative_oneHot, Vilidation_Positive_oneHot,
                                                             TEST_SIZE),
                                  validation_steps=1,
                                  epochs=epochs,
                                  steps_per_epoch=NUM_BATCH,
                                  shuffle=True,
                                  verbose=2,
                                  )  #callbacks=[reduce_lr] 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。

    weighs_path = "LSTM/code/Loso_result/weights/" + data_name + "/" + data_name + "_" + str(i) + ".h5"
    model.save_weights(weighs_path)
    model.load_weights(weighs_path)


    Xtest=np.array(Xtest)
    y_test = [1 if float(i) > 0.0 else 0 for i in Xtest[:, 2]]
    y_test = np_utils.to_categorical(y_test)
    Xtest=pd.DataFrame(Xtest)

#     # print(type(Xtest_oneHot))
    Xtest_oneHot = encordingXtest(Xtest)
    Xtest_oneHot=np.array(Xtest_oneHot)
    Xtest_oneHot = Xtest_oneHot.astype("float64")
    # print("y_test",y_test)
    # print("Xtest_oneHot",Xtest_oneHot)
    Xtest_oneHot=Xtest_oneHot.reshape((len(Xtest_oneHot), 1,24, 7))
    y_pred = model.predict(Xtest_oneHot)
    # print(y_pred)
    y_prob = y_pred[:, 1]
    y_prob = np.array(y_prob)
    # print("y_prob", y_prob)
    y_pred = [int(i[1] > i[0]) for i in y_pred]
    y_test = [int(i[1] > i[0]) for i in y_test]

    fpr, tpr, au_thres = roc_curve(y_test, y_prob)

    auroc = auc(fpr, tpr)
    precision, recall, pr_thres = precision_recall_curve(y_test, y_prob)

    prauc = auc(recall, precision)

    f1score = f1_score(y_test, y_pred)
    precision_scores = precision_score(y_test, y_pred)
    recall_scores = recall_score(y_test, y_pred)
    mcc=matthews_corrcoef(y_test, y_pred)


    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    y_prob = pd.DataFrame(y_prob)

    result = pd.concat([y_test, y_pred, y_prob], axis=1)

    fpr = pd.DataFrame(fpr)
    tpr = pd.DataFrame(tpr)
    fpr_tpr = pd.concat([fpr, tpr], axis=1)

    AUROC.append(auroc)
    PRAUC.append(prauc)
    F1_SCORE.append(f1score)
    PRECISION.append(precision_scores)
    RECALL.append(recall_scores)
    MCC.append(mcc)



auroc_mean, prauc_mean, f1_mean, precision_mean, recall_mean,mcc_mean = np.mean(AUROC), np.mean(PRAUC), np.mean(F1_SCORE), np.mean(PRECISION), np.mean(RECALL),np.mean(MCC)
auroc_std, prauc_std, f1_std, precision_std, recall_std,mcc_std = np.std(AUROC), np.std(PRAUC), np.std(F1_SCORE), np.std(PRECISION), np.std(RECALL),np.std(mcc)
print("Mean result: AUROC=%.3f, PRAUC=%.3f, F1_score=%.3f, Precision=%.3f, Recall=%.3f,Mcc=%.3f" % (auroc_mean, prauc_mean, f1_mean, precision_mean, recall_mean,mcc_mean))

