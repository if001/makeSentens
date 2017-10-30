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


#import matplotlib.pyplot as plt
#import pylab as plt
from itertools import chain #配列ふらっと化
import random


# mylib
# from mecab_test import get_words
# from Const import Const
# from progress_time import ProgressTime

import lib
import nn

# cython
#import pyximport; pyximport.install()
#import cythonFunc
import cython_package.cython_package as cy


class Trainer(lib.Const.Const):
    def __init__(self):
        super().__init__()
        self.window_size = 1
        self.models = []

    def init_seq2seq(self):
        self.seq2seq = nn.Seq2Seq.Seq2Seq()
        self.seq2seq.make_net()


    def fact_seq2seq(self,encord_len,decord_len):
        """
        #buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        """
        # for value in self.buckets:
        #     self.models.append(nn.Seq2Seq.Seq2Seq(value[0],value[1]))
        # models[-1].make_net()
        self.models.append(nn.Seq2Seq.Seq2Seq(encord_len,decord_len))
        self.models[-1].make_net()

    def init_word2vec(self,flag):
        self.word2vec = lib.wordvec.MyWord2Vec()
        if flag == "learn":
            self.word2vec.train(self.dict_train_file)
        elif flag == "load":
            self.word2vec.load_model()
        else:
            print("not word2vec model")
            exit(0)


    def get_word_lists(self):
        print("make wordlists!")
        return cy.readfile_to_sentens(self.dict_train_file)

    def sentens_array_to_str(self,sentens_array):
        __sentens = ""
        for value in sentens_array:
            __sentens += value
        return __sentens

    def sentens_to_vec(self,sentens):
        __sentens_vec = []
        for value in sentens:
            vec = self.word2vec.get_vector(value)
            __sentens_vec.append(vec)
        return __sentens_vec

    def select_bucket(self,sentens_arr):
        index = 0
        for i in range(len(self.buckets)-1):
            if (len(sentens_arr) > self.buckets[i][0]):
                index = self.buckets.index(self.buckets[i+1])
        return index

    def predict_sentens_vec(self,sentens_arr):
        print(">> " + self.sentens_array_to_str(sentens_arr))
        bucket_index = self.select_bucket(sentens_arr)
        print(len(sentens_arr),self.buckets[bucket_index])
        __seq_num = self.buckets[bucket_index][0]

        __sentens_vec = self.sentens_to_vec(sentens_arr[::-1])
        __sentens_vec = self.zero_padding(__sentens_vec,__seq_num)
        __sentens_vec = np.array(__sentens_vec)
        __sentens_vec = __sentens_vec.reshape(1,__seq_num,self.word_feat_len)
        __predict_sentens_vec = self.models[bucket_index].predict(__sentens_vec)
        __predict_sentens_vec = __predict_sentens_vec.reshape(self.buckets[bucket_index][1],self.word_feat_len)
        return __predict_sentens_vec


    def sentens_vec_to_word(self,predict_sentens_vec):
        __output_sentens = ""
        for value in predict_sentens_vec:
            word = self.word2vec.get_word(value)
            __output_sentens += (word+",")
            if (word == "。"): break
        return __output_sentens


    def sentens_vec_to_sentens_arr(self,sentens_vec):
        __arr = []
        for value in sentens_vec:
            __arr.append(self.word2vec.get_word(value)) 
        return __arr

    
    def make_sentens_input(self,sentens):
        print(">> ",sentens)
        __sentens_vec = self.sentens_to_vec(sentens)
        __sentens_vec = self.EOFpadding(__sentens_vec)
        __sentens_vec = np.array(__sentens_vec)
        __sentens_vec = __sentens_vec.reshape(1,self.seq_num,self.word_feat_len)

        # 次の特徴量を予測
        __predict_sentens = self.seq2seq.predict(__sentens_vec)
        __predict_sentens = __predict_sentens.reshape(self.seq_num,self.word_feat_len)

        __output_sentens = ""
        for value in __predict_sentens:
            word = self.word2vec.get_word(value)
            __output_sentens += (word + ",")
            if (word == "。"): break
        print(__output_sentens)
        return __output_sentens


    def glaph_plot(self,vec):
        i = 1
        fig = plt.figure()
        for value in vec:
            plt.subplot(6,5,i)
            t = range(len(value))
            plt.plot(t,value)
            plt.ylim(-2,2)
            i+=1

    def EOF_padding(self,sentens):
        if self.seq_num > len(sentens):
            __diff_len = self.seq_num - len(sentens)
            for i in range(__diff_len):
                sentens.append("。")
        return sentens


    def zero_padding(self,sentens_vec,seq_num):
        if seq_num > len(sentens_vec):
            __diff_len = seq_num - len(sentens_vec)
            for i in range(__diff_len):
                sentens_vec.append([0 for i in range(self.word_feat_len)])
        return sentens_vec

    def reshape_sentens(self,sentens):
        if ('「' in sentens): sentens.remove('「')
        if ('」' in sentens): sentens.remove('」')
        while '' in sentens: sentens.remove('')
        return sentens

    def select_random_sentens(self,word_lists,seq_len):
        while(True):
            index = random.randint(0,len(word_lists)-2)
            __sentens = word_lists[index]
            __sentens = self.reshape_sentens(__sentens)
            if (len(__sentens) <= seq_len): break
        return __sentens


    def get_word_lists(self,fname):
        print("make wordlists!")
        return cy.readfile_to_sentens(fname)


    def make_data(self,word_lists,encord_len,decord_len):
        train_sentens_vec_batch = []
        teach_sentens_vec_batch = []
        for j in range(self.batch_size):
            while(True):
                train_sentens = self.select_random_sentens(word_lists,encord_len)
                teach_sentens = word_lists[word_lists.index(train_sentens)+1]
                teach_sentens = self.reshape_sentens(teach_sentens)
                if(len(teach_sentens) <= decord_len): break

            train_sentens = train_sentens[::-1] # 逆順にする

            train_sentens_vec = self.sentens_to_vec(train_sentens)
            teach_sentens_vec = self.sentens_to_vec(teach_sentens)

            train_sentens_vec = self.zero_padding(train_sentens_vec,encord_len)
            teach_sentens_vec = self.zero_padding(teach_sentens_vec,decord_len)

            train_sentens_vec_batch.append(train_sentens_vec)
            teach_sentens_vec_batch.append(teach_sentens_vec)

        train_sentens_vec_batch = np.array(train_sentens_vec_batch)
        teach_sentens_vec_batch = np.array(teach_sentens_vec_batch)

        # print(len(train_sentens_vec_batch))
        # print(len(train_sentens_vec_batch[0]))
        # print(len(train_sentens_vec_batch[0][0]))
        return train_sentens_vec_batch,teach_sentens_vec_batch




def train(train_model,train,teach):
        print("learninig lstm start")
        for i in range(lib.Const.Const().learning_num):
            print("train step : ",i)
            train_model.train(train,teach)

def save_wait(train_model,fname):
    train_model.waitController("save",fname)

def load_wait(train_model,fname):
    train_model.waitController("load",fname)


def train_main(tr):
    if '--resume' in sys.argv:
        tr.init_word2vec("load")
    else:
        tr.init_word2vec("learn")

    word_lists = tr.get_word_lists(lib.Const.Const().dict_train_file)

    for value in tr.buckets:
        print("start bucket ",value)
        tr.fact_seq2seq(value[0],value[1])

        if '--resume' in sys.argv:
            print("resume "+'param_seq2seq_rnp'+"_"+str(value[0])+"_"+str(value[1])+'.hdf5')
            load_wait(tr.models[-1],'param_seq2seq_rnp'+"_"+str(value[0])+"_"+str(value[1])+'.hdf5')

        train_data,teach_data = tr.make_data(word_lists,value[0],value[1])
        train(tr.models[-1],train_data,teach_data)
        save_wait(tr.models[-1],'param_seq2seq_rnp'+"_"+str(value[0])+"_"+str(value[1])+'.hdf5')


def make_sentens_main(tr):
    tr.init_word2vec("load")
    word_lists = tr.get_word_lists(lib.Const.Const().dict_load_file)

    for value in tr.buckets:
        tr.fact_seq2seq(value[0],value[1])
        load_wait(tr.models[-1],'param_seq2seq_rnp'+"_"+str(value[0])+"_"+str(value[1])+'.hdf5')

    import random
    for i in range(10):
        sentens_arr = tr.select_random_sentens(word_lists,random.choice(tr.buckets)[0])

        sentens_vec = tr.predict_sentens_vec(sentens_arr)
        sentens_arr = tr.sentens_vec_to_sentens_arr(sentens_vec)
        sentens = tr.sentens_array_to_str(sentens_arr)
        print(sentens)
        print("")

    # sentens_arr = tr.select_random_sentens(word_lists,5)
    # for i in range(5):
    #     sentens_vec = tr.predict_sentens_vec(sentens_arr)
    #     sentens_arr = tr.sentens_vec_to_sentens_arr(sentens_vec)        
    #     sentens = tr.sentens_array_to_str(sentens_arr)
    #     print(sentens)
    #     print("")


def main():
    tr = Trainer()
    if '--train' in sys.argv:
        train_main(tr)

    elif '--make' in sys.argv:
        make_sentens_main(tr)

    else:
        print("flag is invalid!")

if __name__ == "__main__" :
    main()
