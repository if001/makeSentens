'''
lstmを使って文章生成
bucketを採用
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

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
import pylab as plt
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


    def init_seq2seq(self):
        """
        #buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        """
        self.seq2seq_models = []
        for value in self.buckets:
            self.seq2seq_models.append(nn.Seq2Seq.Seq2Seq(value[0],value[1])) 

        for value in self.seq2seq_models:
            value.make_net()


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


    def select_random_sentens(self,word_lists):
        while(True):
            index = random.randint(0,len(word_lists)-2)
            __sentens = word_lists[index]
            __sentens = self.reshape_sentens(__sentens)
            if (len(__sentens) <= self.max_seq_len): break
        return __sentens

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

    def make_sentens(self,sentens):
        print(">> " + self.sentens_array_to_str(sentens))
        __sentens_vec = self.sentens_to_vec(sentens[::-1])
        __sentens_vec = self.zero_padding(__sentens_vec)

        __sentens_vec = np.array(__sentens_vec)
        __sentens_vec = __sentens_vec.reshape(1,self.seq_num,self.word_feat_len)

        # 次の特徴量を予測
        __predict_sentens_vec = self.seq2seq.predict(__sentens_vec)
        __predict_sentens_vec = __predict_sentens_vec.reshape(self.seq_num,self.word_feat_len)
        return __predict_sentens_vec

    def sentens_vec_to_word(self,predict_sentens_vec):
        __output_sentens = ""
        for value in predict_sentens_vec:
            word = self.word2vec.get_word(value)
            __output_sentens += (word+",")
            if (word == "。"): break
        return __output_sentens

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

def main():
    rflag = ""
    flag = "learn"
    # rflag = "resume"
    # flag = "make"

    rnp_model = Trainer()

    word_lists = rnp_model.get_word_lists()
    # rnp_model.init_word2vec("learn")
    rnp_model.init_word2vec("load")

    rnp_model.init_seq2seq()

    # if rflag == "resume" :
    #     rnp_model.seq2seq.waitController("load")

    if flag == "learn":
        # lstm
        print("learninig lstm start")
        for i in range(rnp_model.learning_num):
            print("train step : ",i)
            train_sentens_vec_batch = [[] for i in range(len(rnp_model.buckets))]
            teach_sentens_vec_batch = [[] for i in range(len(rnp_model.buckets))]
            for j in range(rnp_model.batch_size):
                while(True):
                    train_sentens = rnp_model.select_random_sentens(word_lists)
                    teach_sentens = word_lists[word_lists.index(train_sentens)+1]
                    teach_sentens = rnp_model.reshape_sentens(teach_sentens)
                    
                    for i in range(len(rnp_model.buckets)-1) :
                        seq_len = rnp_model.buckets[0][1]
                        if(rnp_model.buckets[i][0] < len(train_sentens) < rnp_model.buckets[i+1][0]):
                            seq_len = rnp_model.buckets[i+1][1]
                            select_bucket_index = i + 1
                    print("seq",seq_len)
                    print("tr",len(train_sentens))
                    print("te",len(teach_sentens))
                    print("bu",rnp_model.buckets[select_bucket_index])
                    print("")
                    if(len(teach_sentens) <= seq_len): break
                    
                train_sentens = train_sentens[::-1] # 逆順にする

                train_sentens_vec = rnp_model.sentens_to_vec(train_sentens)
                teach_sentens_vec = rnp_model.sentens_to_vec(teach_sentens)

                train_sentens_vec = rnp_model.zero_padding(train_sentens_vec,
                                                           rnp_model.buckets[select_bucket_index][0])
                teach_sentens_vec = rnp_model.zero_padding(teach_sentens_vec,
                                                           rnp_model.buckets[select_bucket_index][1])
                
                train_sentens_vec_batch[select_bucket_index].append(train_sentens_vec)
                teach_sentens_vec_batch[select_bucket_index].append(teach_sentens_vec)

            train_sentens_vec_batch = np.array(train_sentens_vec_batch)
            teach_sentens_vec_batch = np.array(teach_sentens_vec_batch)

        
            for value in range(len(rnp_model.buckets)):
                print("bucket ",value," len:",len(train_sentens_vec_batch[value]))
                print("len:",len(train_sentens_vec_batch[value][0]))
                print("len:",len(train_sentens_vec_batch[value][0][0]))
                rnp_model.seq2seq_models[value].train(train_sentens_vec_batch[value],
                                                      teach_sentens_vec_batch[value])
                rnp_model.seq2seq_models[value].waitController("save")


    if flag == "make":
        rnp_model.init_word2vec("load")
        rnp_model.seq2seq.waitController("load")

        # # 入力文から文章生成
        # while(True):
        #     input_line = input(">> ")
        #     rnp_model.make_sentens_input(input_line)

        for i in range(5):
            input_sentens = rnp_model.select_random_sentens(word_lists)

            predict_sentens_vec = rnp_model.make_sentens(input_sentens)
            output_sentens = rnp_model.sentens_vec_to_word(predict_sentens_vec)
            print(output_sentens)
            print("")

            input_sentens_vec = rnp_model.sentens_to_vec(input_sentens[::-1])
            input_sentens_vec = rnp_model.zero_padding(input_sentens_vec)

        #     rnp_model.glaph_plot(input_sentens_vec)
        #     rnp_model.glaph_plot(predict_sentens_vec)
        # plt.show()

if __name__ == "__main__" :
    main()

