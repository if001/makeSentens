
"""
学習データと教師データ作るのがおもすぎたのでcythonにまかせる

file読み込みもcythonに任せてみる

"""


#numpyをcythonから扱えるようにするためにcimportを使用します
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
from tqdm import tqdm #プログレスバー


# def makeTrainDataWord2Vec(input_word_lists,word_dict,window_size):
# #def makeTrainDataWord2Vec(input_len,word_dict,word_lists,window):
#     print("cython!! make one hot vector for word2vec train!! ")

#     train_x =[]
#     train_y =[]
#     cdef int i = 0
#     cdef int j = 0
#     cdef int ls,le
#     cdef int input_word_lists_len = len(input_word_lists)
#     cdef int word_dict_len = len(word_dict)

#     for i in range(word_lists_len):
#         sys.stdout.write("\r%d" % i)
#         sys.stdout.flush()

#         tmp_train_x = np.zeros(input_len)
#         tmp_train_y = np.zeros(0)

#         tmp_train_x = np.zeros(input_len, dtype=np.float64)
#         tmp_train_y = np.zeros(0, dtype=np.float64)

#         # for train x
#         tmp_train_x[np.where(word_dict == word_lists[i])] = 1
#         train_x.append(tmp_train_x)

#         # for train y
#         if window == 0:
#             ls = 0
#             le = 1
#         else:
#             ls = -window
#             le = window + 1

#         for j in range(ls,le):
#             tmp_tmp_train_y = np.zeros(word_dict_len)
#             if (( i + j ) > 0) and ((i + j) < len(word_lists)):
#                 tmp_tmp_train_y[np.where(word_dict == word_lists[i+j])] = 1
#             tmp_train_y.append(tmp_tmp_train_y)

#         train_y.append(tmp_train_y)

#     return train_x,train_y




import re
import sys
def readfile_for_word2vec(fname):
    cdef int i = 0
    try:
        wordlists = []
        with open('./'+fname,'r') as file:
            for line in file :
                sys.stdout.write("\r done readfile to make word list : %d" % i)
                sys.stdout.flush()
                i += 1
                line = re.sub(r'\n', "", line)
                if len(line) != 0:
                    wordlists.append(line.split(" "))
        return wordlists

    except:
        print("not such file")
        sys.exit(0)




def makeTrainData(input_wordlist,word_dict,window_size):
    print("make_train_data for make sentens(cython)")
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    X_train = []
    Y_train = []

    for i in range(len(input_wordlist)):
        sys.stdout.write("%d / %d\r" % (i,len(input_wordlist)))
        sys.stdout.flush()
        for j in range(len(input_wordlist[i])-window_size):
            tmp_train_data_x = np.zeros(0)
            for k in range(window_size):
                tmp_tmp_train_data = np.zeros(len(word_dict))
                tmp_tmp_train_data[np.where(word_dict == input_wordlist[i][j+k])] = 1
                tmp_train_data_x = np.r_[tmp_train_data_x, tmp_tmp_train_data]

            tmp_train_data_y = np.zeros(len(word_dict))
            tmp_train_data_y[np.where(word_dict == input_wordlist[i][j+window_size])] = 1

            X_train.append(tmp_train_data_x)
            Y_train.append(tmp_train_data_y)
            # tmp_train_data1 = np.zeros(len(word_dict))
            # tmp_train_data1[np.where(word_dict == input_wordlist[i][j])] = 1
            # tmp_train_data2 = np.zeros(len(word_dict))
            # tmp_train_data2[np.where(word_dict == input_wordlist[i][j+1])] = 1
            # tmp_train_data3 = np.zeros(len(word_dict))
            # tmp_train_data3[np.where(word_dict == input_wordlist[i][j+2])] = 1
            # tmp_train_data4 = np.zeros(len(word_dict))
            # tmp_train_data4[np.where(word_dict == input_wordlist[i][j+3])] = 1

            # X_train.append(np.r_[tmp_train_data1,tmp_train_data2,tmp_train_data3,tmp_train_data4])
            # Y_train.append(tmp_train_data5)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_train = X_train.reshape(len(X_train), 1, len(X_train[0]))
    return X_train,Y_train

    # sys.exit(0)

