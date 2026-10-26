import numpy as np
import os
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
<<<<<<< HEAD

import Shuffle
from MODEL import Crispr_SGRU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Encoder-sgRNA_off import Encoder

def load_data(file_path):
    data_list = []
    label = []
    Negative = []
    Positive = []
    with open(file_path) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]  # strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。
            label_item = np.float64(ll[2])
            data_item = [int(i) for i in ll[3:]]
            if label_item == 0.0:
                Negative.append(ll)
            else:
                Positive.append(ll)
            data_list.append(data_item)
            label.append(label_item)
    return Negative, Positive, label
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
        off_seq = row[1
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        final_code.append(en.on_off_code)
    return final_code

# Negative, Positive, label =loadData('change_CIRCLE_seq_10gRNA_wholeDataset.txt')
Negative, Positive, label ='AACACCAGTGAGTAGAGCGGAGG,AACACCAGTTAGACCAGAGGTGG,0'
Xtest = np.vstack((Negative, Positive))
Xtest= encordingXtest(Xtest)
weighs_path = "weight/K562/K562_0.h5"
model=Crispr_SGRU()
model.load_weights(weighs_path)
y_pred=model.predict(Xtest)
print(y_pred)

