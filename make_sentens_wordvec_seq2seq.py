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
import random


# mylib
from wordvec import MyWord2Vec
from MyWord2Vec import myWord2Vec 
from MyVec2Word import myVec2Word
from Seq2Seq import Seq2Seq
from Const import Const
from progress_time import ProgressTime

# cython
import pyximport; pyximport.install()
import cythonFunc


class Trainer(Const):
    def __init__(self):
        super().__init__()
        self.window_size = 1


    def init_seq2seq(self):
        self.seq2seq = Seq2Seq()
        self.seq2seq.make_net()


    def init_word2vec(self):
        # ---- word2vec -----
        self.word2vec = MyWord2Vec()
        # word2vec.train("./aozora_text/files_all.txt")
        self.word2vec.load_model()
        # ---- word2vec -----

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

    def make_sentens(self):
        sentens = random.sample(self.word_lists,1)[0]
        print(sentens)
        
        sentens_vec = self.sentens_to_vec(sentens)
        sentens_vec = self.EOFpadding(sentens_vec)
        sentens_vec = np.array(sentens_vec)
        sentens_vec = sentens_vec.reshape(1,self.seq_num,self.word_feat_len)

        # 次の特徴量を予測
        predict_sentens = self.seq2seq.predict(sentens_vec)
        predict_sentens = predict_sentens.reshape(self.seq_num,self.word_feat_len)

        output_sentens = ""
        for value in predict_sentens:
            output_sentens += self.word2vec.get_word(value)
            if( (value == "。") or (value == "．") ): break

        print(output_sentens)
                

    def sentens_to_vec(self,sentens):
        sentens_vec = []
        while '' in sentens: sentens.remove('')
        for value in sentens:
            vec = self.word2vec.get_vector(value)
            sentens_vec.append(vec)
        return sentens_vec


    def EOFpadding(self,sentens):
        if self.seq_num > len(sentens):
            diff_len = self.seq_num - len(sentens)
            for i in range(diff_len):
                sentens.append(self.word2vec.get_vector("。"))
        return sentens

def main():
    # flag = "learn"
    flag = "make"

    rnp_model = Trainer()

    # ---- 新しく辞書作る場合 -----    
    rnp_model.getWordLists()
    # rnp_model.getDict()
    # rnp_model.dictController("save")
    # ---- 新しく辞書作る場合 -----
    rnp_model.dictController("load")

    # インスタンス作成
    rnp_model.init_word2vec()

    # インスタンス作成
    rnp_model.init_seq2seq()


    if flag == "learn":
        # lstm
        print("learninig lstm start")
        for i in range(rnp_model.learning_num):
            train_sentens_vec_batch = []
            teach_sentens_vec_batch = []
            for j in range(rnp_model.batch_size):
                while(True):
                    index = random.randint(0,len(rnp_model.word_lists)-1)
                    train_sentens = rnp_model.word_lists[index]
                    teach_sentens = rnp_model.word_lists[index+1]
                    if( (len(train_sentens)<=100) and (len(teach_sentens)<=100) ): break
    
                train_sentens_vec = rnp_model.sentens_to_vec(train_sentens)
                teach_sentens_vec = rnp_model.sentens_to_vec(teach_sentens)
                    
                train_sentens_vec = rnp_model.EOFpadding(train_sentens_vec)
                teach_sentens_vec = rnp_model.EOFpadding(teach_sentens_vec)
                print("train_vec",len(train_sentens_vec))
                print("teach_vec",len(teach_sentens_vec))
                train_sentens_vec_batch.append(train_sentens_vec)
                teach_sentens_vec_batch.append(teach_sentens_vec)

            train_sentens_vec_batch = np.array(train_sentens_vec_batch)
            teach_sentens_vec_batch = np.array(teach_sentens_vec_batch)
            print(train_sentens_vec_batch.shape)
            print(teach_sentens_vec_batch.shape)
            rnp_model.seq2seq.train(train_sentens_vec_batch,teach_sentens_vec_batch)
            rnp_model.seq2seq.waitController("save")


    if flag == "make":
        rnp_model.word2vec.load_model()
        rnp_model.seq2seq.waitController("load")

        for i in range(10):
            rnp_model.make_sentens()


if __name__ == "__main__" :
    main()

