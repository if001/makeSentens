'''
lstmを使って文章生成
word2vecを自作して、vector化
今度こそ

word2vecはSkip-gramではなく
CBoWを採用

batch_input_shape=(None,\
                   LSTMの中間層に入力するデータの数（※文書データなら単語の数）,\
                   LSTM中間層に投入するデータの次元数（※文書データなら１次元配列なので1)
                  )
'''


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras import optimizers
from keras.models import Model
from keras import backend as K

import numpy as np
import random
import sys

from mecab_test import get_words
#import matplotlib.pyplot as plt
import pylab as plt
from itertools import chain #配列ふらっと化



# mylib
<<<<<<< HEAD
from wordvec import MyWord2Vec
=======
>>>>>>> 4ffc0a954cb20e688078c8d8e03b934a051fe64e
from MyWord2Vec import myWord2Vec 
from MyVec2Word import myVec2Word
from Lstm import Lstm
from Const import Const
from progress_time import ProgressTime

# cython
import pyximport; pyximport.install()
import cythonFunc


class Trainer(Const):
    def __init__(self):
        super().__init__()
        self.window_size = 1

    def getWordLists(self):
        print("make wordlists!")
        self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/files_all.txt")
        # self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/re_re_test.txt")
        # self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/re_re_akumano_monsho.txt")
        # self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/re_re_kohantei_jiken.txt")
        print("")

    def getDict(self):
        print("make dict!")
        print("word_lists len : ",len(self.word_lists))
        self.word_dict = list(set(chain.from_iterable(self.word_lists)))
        print("word dict length : ",len(self.word_dict))

    def dictController(self,flag):
        if flag == "save" :
            print("save dict")
            file = open('./dictionaly/dict.txt', 'w')
            for value in self.word_dict:
                file.write(value+",")
            file.close()

        if flag == "load" :
            print("load dict")
            file = open('./dictionaly/dict.txt', 'r')
            string = file.read()
            self.word_dict = string.split(",")
            file.close()

    def make_sentens(self,lstm,word2vec):
        sentens = ""
        wbin = np.zeros(self.word_feat_len)
        r = random.randint(0,self.word_feat_len-1)
        wbin[r] = 1
        while(True):
            # 次の特徴量を予測
            predict_list = lstm.predict(wbin)
            # 特徴量を単語に変換
            predict_list = predict_list.reshape(len(predict_list[0]))
            predict_list = np.array(predict_list,dtype='f')
            predict_word = word2vec.get_word(predict_list)
            # print("predict word",predict_word)
            wbin = word2vec.get_vector(predict_word)

            sentens += predict_word
            if((sentens[-1] == "e") or (sentens[-1] == ".") or (sentens[-1] == "。")) :  break
            if len(sentens) > 30: break
        print(sentens)


def main():
    flag = "learn"
    # flag = "make"

    rnp_model = Trainer()

    # ---- 新しく辞書作る場合 -----    
    rnp_model.getWordLists()

    # rnp_model.getDict()
    # rnp_model.dictController("save")
    # ---- 新しく辞書作る場合 -----
    rnp_model.dictController("load")

    # ---- word2vec -----
    word2vec = MyWord2Vec()
    #word2vec.train("./aozora_text/files_all.txt")    
    word2vec.load_model()
    # ---- word2vec -----

    lstm = Lstm()
    lstm.input_len_set(len(rnp_model.word_dict))
    lstm.make_net()

    if flag == "learn":
        # lstm
        print("learninig lstm start")
        for i in range(rnp_model.learning_num):
            x_train = []
            y_train = []

            while(True):
                # 文章からランダムに1行とってくる
                input_word_lists = random.sample(rnp_model.word_lists,1)

                # 半角スペースがあればその数だけ消す
                cnt = input_word_lists[0].count('')
                for i in range(cnt):
                    if '' in input_word_listns[0]: input_word_lists[0].remove('')

                # train date 作成ループ
                for k in range(len(input_word_lists[0])-1-1):
                    print(input_word_lists[0][k],input_word_lists[0][k+1],":",end="")
                    x_train.append(word2vec.get_vector(input_word_lists[0][k]))
                    y_train.append(word2vec.get_vector(input_word_lists[0][k+1]))

                print("---")
                if(rnp_model.batch_size <= len(x_train)): break
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            print(x_train.shape)
            x_train = x_train.reshape(len(x_train), 1, len(x_train[0]))
            print(x_train.shape)
            lstm.train(x_train,y_train)
            rnp_model.make_sentens(lstm,word2vec)
            lstm.waitController("save")

        # vec = word2vec.get_vector('私')
        # predict_list = lstm.predict(vec)
        # t = range(len(predict_list))
        # plt.plot(t,predict_list)
        # plt.show()

    if flag == "make":
        word2vec.load_model()
        lstm.waitController("load")
        for i in range(10):
            rnp_model.make_sentens(lstm,word2vec)


if __name__ == "__main__" :
    main()

