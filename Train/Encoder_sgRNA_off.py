# -*- coding: utf-8 -*-
# @Time     :10/15/18 9:50 PM
# @Auther   :Jason Lin
# @File     :Encoder_sgRNA_off$.py
# @Software :PyCharm

import numpy as np

class Encoder:
    def __init__(self, on_seq, off_seq, with_category = False, label = None, with_reg_val = False, value = None):
        tlen = 24
        self.on_seq = "-" *(tlen-len(on_seq)) +  on_seq #补全-[0, 0, 0, 0, 0]
        self.off_seq = "-" *(tlen-len(off_seq)) + off_seq
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
        if with_category:
            self.label = label
        if with_reg_val:
            self.value = value
        self.encode_on_off_dim7()

    def encode_sgRNA(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        sgRNA_bases = list(self.on_seq)
        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_on_off_dim7(self):#进行异或合并运算
        self.encode_sgRNA()
        self.encode_off()
        on_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            on_b = on_bases[i]
            off_b = off_bases[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(2)#direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code = np.array(on_off_dim7_codes)


class Encoder_16:
    def __init__(self, on_seq, off_seq, with_category=False, label=None, with_reg_val=False, value=None):
        tlen = 24
        self.on_seq = "-" * (tlen - len(on_seq)) + on_seq  # 补全-[0, 0, 0, 0, 0]
        self.off_seq = "-" * (tlen - len(off_seq)) + off_seq
        self.encode_sgRNA_DNA()

    def encode_sgRNA_DNA(self):
        code_list = []
        encoded_dict_basepair = {
            '--': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'AA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'AT': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'AG': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'AC': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'TA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'TT': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'TG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'TC': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'GA': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'GT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'GG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'GC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'CA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'CT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'CG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'CC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        sgRNA_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        for i in range(len(sgRNA_bases)):
                sgRnaDna =sgRNA_bases[i]+  off_bases[i] 
                code_list.append(encoded_dict_basepair[sgRnaDna])
        
        self.sgRNA_DNA_code = np.array(code_list)

def AttnToMismatch(file):
    code_dict= {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                        "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16,
                        "A_": 17,"_A": 18,"C_": 19,"_C": 20,"G_": 21,"_G": 22,"T_": 23,"_T": 24,"--":25}
    # def load_data(file_path):
    #     data_list = []
    #     label = []
    #     sgRNA = []
    #     DNA = []
    #     with open(file_path) as f:
    #         for line in f:
    #             ll = [i for i in line.strip().split(',')]
    #             sgRNA.append(ll[0])
    #             DNA.append(ll[1])
    #             label.append(ll[2])
    #     return sgRNA, DNA, label
    sgRNA=[]
    DNA=[]
    label=[]
    Vector_list = []
    sgRNA,DNA,label=load_data(file)
    data_length=len(sgRNA)
    outputList3 = "file.txt"
    fout_own=open(outputList3,'w')
    for i in range(data_length):#数据集one-hot编码
        Vector = []
        Vector.append(sgRNA[i])
        Vector.append(DNA[i])
        Vector.append(float(label[i]))# for Classification schema

        # Vector.append(hek_labels_list[i])# for Regression schema
        for j in range(len(sgRNA[i])):
            temp=sgRNA[i][j] + DNA[i][j]
            # temp = sgRNA[i][j] + DNA[i][j]
            Vector.append(code_dict[temp]-1)#MATCH_ROW_NUMBER1编码操作
            Vector_list.append(Vector[2:])
        fout_own.writelines(",".join('%s' % item for item in Vector) + '\n')



# CnnCrispr词典构建及数据预处理 分类
# from util import util as use
# from util import timingTool

# import xlrd
# from mittens import GloVe
# import numpy as np
# import gc
# tic = timingTool()
MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                    "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}
# 共现矩阵的计算
def countCOOC(cooccurrence, window, coreIndex):
    for index in range(len(window)):
        if index == coreIndex:
            continue
        else:
            cooccurrence[window[coreIndex]][window[index]] = cooccurrence[window[coreIndex]][window[index]] + 1
    return cooccurrence
Vector_list = []
###修改成可以读入自己的数据并进行编辑 记得修改Vector
def load_data(file_path):
    data_list = []
    label = []
    sgRNA = []
    DNA = []
    with open(file_path) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]
            sgRNA.append(ll[0])
            DNA.append(ll[1])
            label.append(ll[2])
    return sgRNA, DNA, label
# sgRNA=[]
# DNA=[]
# label=[]
# sgRNA,DNA,label=load_data('dataset.txt')
# data_length=len(sgRNA)
# outputList3 = "dataset.txt"
# fout_own=open(outputList3,'w')
# for i in range(data_length):#数据集one-hot编码
#     Vector = []
#     Vector.append(sgRNA[i])
#     Vector.append(DNA[i])
#     Vector.append(float(label[i]))# for Classification schema
#     # Vector.append(hek_labels_list[i])# for Regression schema 本来就注释掉的
#     for j in range(len(sgRNA[i])):
#         temp = sgRNA[i][j] + DNA[i][j]
#         Vector.append(MATCH_ROW_NUMBER1[temp]-1)#MATCH_ROW_NUMBER1编码操作
#     Vector_list.append(Vector[2:])
#     fout_own.writelines(",".join('%s' % item for item in Vector) + '\n')#写入
###


##glove共现矩阵有关 cooccurrence_5.csv，根据获得的编码进行计算
# Create an empty table
# tableSize = 16
# coWindow = 5
# vecLength = 100  # The length of the matrix
# max_iter = 10000  # Maximum number of iterations
# display_progress = 1000
# cooccurrence = np.zeros((tableSize, tableSize), "int64")
# print("An empty table had been created.")
# print(cooccurrence.shape)
# Start statistics
# data = Vector_list
# flag = 0
# for item in data:
#     itemInt = [int(x) for x in item]
#     for core in range(1, len(item)):
#         if core <= coWindow + 1:
#             window = itemInt[1:core + coWindow + 1]
#             coreIndex = core - 1
#             cooccurrence = countCOOC(cooccurrence, window, coreIndex)# 共现矩阵的计算
#
#         elif core >= len(item) - 1 - coWindow:
#             window = itemInt[core - coWindow:(len(item))]
#             coreIndex = coWindow
#             cooccurrence = countCOOC(cooccurrence, window, coreIndex)# 共现矩阵的计算
#
#         else:
#             window = itemInt[core - coWindow:core + coWindow + 1]
#             coreIndex = coWindow
#             cooccurrence = countCOOC(cooccurrence, window, coreIndex)# 共现矩阵的计算
#
#     flag = flag + 1
#     # if flag % 20 == 0:
#         # print("%s pieces of data have been calculated, taking %s" % (flag, tic.timmingGet()))
# print("The calculation of co-occurrence matrix was completed, taking %s" % (tic.timmingGet()))
#
# del data, window
# gc.collect()
# # Display of statistical results
# nowTime = tic.getNow().strftime('%Y%m%d_%H%M%S')
# # coocPath = "CnnCrispr-master/Change_CnnCrispr/Datachange/cooccurrence_ll6_%s.csv" % (coWindow)
# coocPath = "CnnCrispr-master/Change_CnnCrispr//Datachange/cooccurrence_dataset_%s.csv" % (coWindow)
# writer = use.csvWrite(coocPath)
# for item in cooccurrence:
#     writer.writerow(item)
# print("The co-occurrence matrix is derived, taking %s" % (tic.timmingGet()))
#
# # GloVe
# print("Start GloVe calculation")
# coocMatric = np.array(cooccurrence, "float32")
# glove_model = GloVe(n=vecLength, max_iter=max_iter,
#                     display_progress=display_progress)
# embeddings = glove_model.fit(coocMatric)
#
# del cooccurrence, coocMatric
# gc.collect()
# # Output calculation result
# dicIndex = 0
# # result=[]
# nowTime = tic.getNow().strftime('%Y%m%d_%H%M%S')
# GlovePath = "CnnCrispr-master/Change_CnnCrispr/Datachange/keras_GloVeVec_dataset_%s_%s_%s.csv" % (coWindow, vecLength,max_iter)
# writer = use.csvWrite(GlovePath)
# for embeddingsItem in embeddings:
#     item = np.array([dicIndex])
#     item = np.append(item, embeddingsItem)
#     writer.writerow(item)
#     dicIndex = dicIndex + 1
# print("Finished!")


class CRISPR_net_Encoder:
    def __init__(self, on_seq, off_seq, with_category = False, label = None, with_reg_val = False, value = None):
        tlen = 24
        self.on_seq = "-" *(tlen-len(on_seq)) +  on_seq #补全-[0, 0, 0, 0, 0]
        self.off_seq = "-" *(tlen-len(off_seq)) + off_seq
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
        if with_category:
            self.label = label
        if with_reg_val:
            self.value = value
        self.encode_on_off_dim7()

    def encode_sgRNA(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        sgRNA_bases = list(self.on_seq)
        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_on_off_dim7(self):#进行异或合并运算
        self.encode_sgRNA()
        self.encode_off()
        on_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            on_b = on_bases[i]
            off_b = off_bases[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(2)#direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code = np.array(on_off_dim7_codes)


encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0],
                '-': [0, 0, 0, 0]}
pos_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, '_': 5, '-': 5}
def CRISPR_IP_Encoder(on_seq, off_seq):
    tlen = 24
    on_seq = "-" * (tlen - len(on_seq)) + on_seq

    off_seq = "-" * (tlen - len(off_seq)) + off_seq
    on_seq_code = np.array([encoded_dict[base] for base in list(on_seq)])
    off_seq_code = np.array([encoded_dict[base] for base in list(off_seq)])
    on_off_dim6_codes = []
    for i in range(len(on_seq)):
        diff_code = np.bitwise_or(on_seq_code[i], off_seq_code[i])  # 进行or运算
        dir_code = np.zeros(2)
        if pos_dict[on_seq[i]] == pos_dict[off_seq[i]]:
            diff_code = diff_code * -1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[on_seq[i]] < pos_dict[off_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[on_seq[i]] > pos_dict[off_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", on_seq, off_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    isPAM = np.zeros((24, 1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code


def encode_in_6_dimensions(on_target_seq, off_target_seq):
    bases_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0],
                  '-': [0, 0, 0, 0]}
    positions_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, '_': 5, '-': 5}

    tlen = 24
    # 将长度不足的，一般是23，补全到24，前面加个空“-”
    on_target_seq = "-" * (tlen - len(on_target_seq)) + on_target_seq
    off_target_seq = "-" * (tlen - len(off_target_seq)) + off_target_seq
    # 碱基，indel和空的 编码，转换
    on_target_seq_code = np.array([bases_dict[base] for base in list(on_target_seq)])
    off_target_seq_code = np.array([bases_dict[base] for base in list(off_target_seq)])

    pair_dim5_codes = []
    for i in range(len(on_target_seq)):
        bases_code = np.bitwise_or(on_target_seq_code[i], off_target_seq_code[i])  # 前四维
        dir_code = np.zeros(1)  # 表示两个碱基在on off target上的可能位置，1，0，-1

        if positions_dict[on_target_seq[i]] == positions_dict[off_target_seq[i]]:
            bases_code = bases_code * -1
        elif positions_dict[on_target_seq[i]] < positions_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif positions_dict[on_target_seq[i]] > positions_dict[off_target_seq[i]]:
            dir_code[0] = -1
        else:
            raise Exception("Invalid seq!", on_target_seq, off_target_seq)

        pair_dim5_codes.append(np.concatenate((bases_code, dir_code)))
    pair_dim5_codes = np.array(pair_dim5_codes)

    isPAM = np.zeros((24, 1))
    isPAM[-3:, :] = 1
    pair_code = np.concatenate((pair_dim5_codes, isPAM), axis=1)
    return pair_code

#CRISPR_M Encoding
# def encode_by_base_pair_vocabulary(on_target_seq, off_target_seq):
#     BASE_PAIR_VOCABULARY_v1 = {
#         "AA": 0, "TT": 1, "GG": 2, "CC": 3,
#         "AT": 4, "AG": 5, "AC": 6, "TG": 7, "TC": 8, "GC": 9,
#         "TA": 10, "GA": 11, "CA": 12, "GT": 13, "CT": 14, "CG": 15,
#         "A_": 16, "T_": 17, "G_": 18, "C_": 19,
#         "_A": 20, "_T": 21, "_G": 22, "_C": 23,
#         "AAP": 24, "TTP": 25, "GGP": 26, "CCP": 27,
#         "ATP": 28, "AGP": 29, "ACP": 30, "TGP": 31, "TCP": 32, "GCP": 33,
#         "TAP": 34, "GAP": 35, "CAP": 36, "GTP": 37, "CTP": 38, "CGP": 39,
#         "A_P": 40, "T_P": 41, "G_P": 42, "C_P": 43,
#         "_AP": 44, "_TP": 45, "_GP": 46, "_CP": 47,
#         "__": 48, "__P": 49
#     }
#     BASE_PAIR_VOCABULARY_v2 = {
#         "AA": 0, "TT": 1, "GG": 2, "CC": 3,
#         "AAP": 4, "TTP": 5, "GGP": 6, "CCP": 7,
#         "AT": 8, "AG": 9, "AC": 10, "TG": 11, "TC": 12, "GC": 13,
#         "TA": 14, "GA": 15, "CA": 16, "GT": 17, "CT": 18, "CG": 19,
#         "ATP": 20, "AGP": 21, "ACP": 22, "TGP": 23, "TCP": 24, "GCP": 25,
#         "TAP": 26, "GAP": 27, "CAP": 28, "GTP": 29, "CTP": 30, "CGP": 31,
#         "A_": 32, "T_": 33, "G_": 34, "C_": 35,
#         "_A": 36, "_T": 37, "_G": 38, "_C": 39,
#         "A_P": 40, "T_P": 41, "G_P": 42, "C_P": 43,
#         "_AP": 44, "_TP": 45, "_GP": 46, "_CP": 47,
#         "__": 48, "__P": 49
#     }
#     BASE_PAIR_VOCABULARY_v3 = {
#         "AA": 0, "TT": 1, "GG": 2, "CC": 3,
#         "AT": 4, "AG": 5, "AC": 6, "TG": 7, "TC": 8, "GC": 9,
#         "TA": 10, "GA": 11, "CA": 12, "GT": 13, "CT": 14, "CG": 15,
#         "A_": 16, "T_": 17, "G_": 18, "C_": 19,
#         "_A": 20, "_T": 21, "_G": 22, "_C": 23,
#         "__": 24
#     }
#     tlen = 24
#     # 将长度不足的，一般是23，补全到24，前面加个空“-”
#     on_target_seq = "_" * (tlen - len(on_target_seq)) + on_target_seq
#     off_target_seq = "_" * (tlen - len(off_target_seq)) + off_target_seq
#     on_target_seq = on_target_seq.replace("-", "_")
#     off_target_seq = off_target_seq.replace("-", "_")
# 
#     pair_vector = list()
#     for i in range(tlen):
#         base_pair = on_target_seq[i] + off_target_seq[i]
#         # if i > 20:
#         #     base_pair += "P"
#         pair_vector.append(BASE_PAIR_VOCABULARY_v3[base_pair])
#     pair_vector = np.array(pair_vector)
#     return pair_vector
# 
# 
# def encode_by_base_vocabulary(seq):
#     BASE_VOCABULARY_v1 = {
#         "A": 50, "T": 51, "G": 52, "C": 53, "_": 54
#     }
#     BASE_VOCABULARY_v3 = {
#         "A": 25, "T": 26, "G": 27, "C": 28, "_": 29
#     }
#     tlen = 24
#     # 将长度不足的，一般是23，补全到24，前面加个空“-”
#     seq = "_" * (tlen - len(seq)) + seq
#     seq = seq.replace("-", "_")
# 
#     seq_vector = list()
#     for i in range(tlen):
#         base = seq[i]
#         seq_vector.append(BASE_VOCABULARY_v3[base])
#     seq_vector = np.array(seq_vector)
#     return seq_vector
# 
# 
# def encode_by_one_hot(on_target_seq, off_target_seq):
#     bases_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0],
#                   '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 1]}
# 
#     tlen = 24
#     # 将长度不足的，一般是23，补全到24，前面加个空“-”
#     on_target_seq = "-" * (tlen - len(on_target_seq)) + on_target_seq
#     off_target_seq = "-" * (tlen - len(off_target_seq)) + off_target_seq
# 
#     pair_dim11_codes = []
#     for i in range(len(on_target_seq)):
#         base_code = bases_dict[on_target_seq[i]] + bases_dict[off_target_seq[i]]
#         if i in [21, 22, 23]:
#             base_code.append(1)
#         elif 0 <= i <= 20:
#             base_code.append(0)
#         else:
#             raise Exception("base code error")
#         pair_dim11_codes.append(base_code)
#     pair_dim11_codes = np.array(pair_dim11_codes)
#     return pair_dim11_codes
# 
# def encording(data):
#     Data_features = []
#     Data_feature_ont = []
#     Data_feature_offt = []
#     Data_labels = []
#     on_epigenetic_code = []
#     off_epigenetic_code = []
# 
#     for idx, row in data.iterrows():
#         on_target_seq = row[0]
# 
#         off_target_seq = row[1]
# 
#         label = row[2]
#         Data_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
#         Data_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
#         Data_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
#         Data_labels.append(label)
#         # on_epigenetic_code.append(get_epigenetic_code(row[2], row[3], row[4], row[5]))
#         # off_epigenetic_code.append(get_epigenetic_code(row[7], row[8], row[9], row[10]))
# 
#     Data_features = np.array(Data_features)
#     Data_feature_ont = np.array(Data_feature_ont)
#     Data_feature_offt = np.array(Data_feature_offt)
#     # on_epigenetic_code = np.array(on_epigenetic_code)
#     # off_epigenetic_code = np.array(off_epigenetic_code)
#     Data_labels = np.array(Data_labels)
#     print("[INFO] Encoded dataset Data features with size of", Data_features.shape)
#     print("[INFO] Encoded dataset Data feature ont with size of", Data_feature_ont.shape)
#     print("[INFO] Encoded dataset Data feature offt with size of", Data_feature_offt.shape)
#     # print("[INFO] The labels number of active off-target sites in dataset Data is {0}, the active+inactive is {1}.".format(len(Data_labels[Data_labels>0]), len(Data_labels)))
#     # print("[INFO] Encoded dataset Data on_epigenetic_code with size of", on_epigenetic_code.shape)
#     # print("[INFO] Encoded dataset Data off_epigenetic_code with size of", off_epigenetic_code.shape)
#     return Data_features, Data_feature_ont, Data_feature_offt, Data_labels
# 
# 
# def get_epigenetic_code(epigenetic_1, epigenetic_2, epigenetic_3, epigenetic_4):
#     epimap = {'A': 1, 'N': 0}
#     tlen = 24
#     epigenetic_1 = epigenetic_1.upper()
#     epigenetic_1 = "N" * (tlen - len(epigenetic_1)) + epigenetic_1
#     epigenetic_2 = epigenetic_2.upper()
#     epigenetic_2 = "N" * (tlen - len(epigenetic_2)) + epigenetic_2
#     epigenetic_3 = epigenetic_3.upper()
#     epigenetic_3 = "N" * (tlen - len(epigenetic_3)) + epigenetic_3
#     epigenetic_4 = epigenetic_4.upper()
#     epigenetic_4 = "N" * (tlen - len(epigenetic_4)) + epigenetic_4
#     epi_code = list()
#     for i in range(len(epigenetic_1)):
#         t = [epimap[epigenetic_1[i]], epimap[epigenetic_2[i]], epimap[epigenetic_3[i]], epimap[epigenetic_4[i]]]
#         epi_code.append(t)
#     return epi_code
# 
# class PositionalEncoding(keras.layers.Layer):
#     def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
#         super().__init__(dtype=dtype, **kwargs)
#         self.max_steps = max_steps
#         self.max_dims = max_dims
# 
#         if max_dims % 2 == 1: max_dims += 1  # max_dims must be even
# 
#         p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
#         pos_emb = np.empty((1, max_steps, max_dims))
#         pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
#         pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
#         self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
# 
#     def call(self, inputs):
#         shape = tf.shape(inputs)
#         return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]
# 
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'max_steps': self.max_steps,
#             'max_dims': self.max_dims
#         })
#         return config

# Testing
# e = Encoder_16(on_seq="AGCTGA", off_seq="CGGGTT")
# e.encode_sgRNA_DNA()
# print(e.sgRNA_DNA_code)












