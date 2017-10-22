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
import pylab as plt
from itertools import chain #配列ふらっと化
import random


# mylib
from mecab_test import get_words
from wordvec import MyWord2Vec
# from MyWord2Vec import myWord2Vec 
# from MyVec2Word import myVec2Word
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
        self.word2vec = MyWord2Vec()
        self.word2vec.train(self.dict_train_file)


    def get_word_lists(self):
        print("make wordlists!")
        return cythonFunc.readfile_to_sentens(self.dict_train_file)

    def get_dict(self,word_list):
        print("make dict!")
        print("word_lists len : ",len(self.word_lists))
        self.word_dict = list(set(chain.from_iterable(self.word_lists)))
        print("word dict length : ",len(self.word_dict))
        return list(set(chain.from_iterable(self.word_lists)))

    def dict_controller(self,flag):
        if flag == "save" :
            print("save dict")
            file = open(self.dict_dir, 'w')
            for value in self.word_dict:
                file.write(value+",")
            file.close()

        if flag == "load" :
            print("load dict")
            file = open(self.dict_dir, 'r')
            string = file.read()
            file.close()
            self.word_dict = string.split(",")
            return string.split(",")

    def make_sentens(self,word_lists):
        while(True):
            sentens = random.sample(word_lists,1)[0]
            if( len(sentens) <= self.seq_num ) : break

        input_sentens = ""
        for value in sentens:
            input_sentens += value
        print(">> " + input_sentens)

        sentens_vec = self.sentens_to_vec(sentens)
        sentens_vec = self.EOFpadding(sentens_vec)
        # self.glaph_plot(sentens_vec)
        sentens_vec = np.array(sentens_vec)
        sentens_vec = sentens_vec.reshape(1,self.seq_num,self.word_feat_len)


        # 次の特徴量を予測
        predict_sentens = self.seq2seq.predict(sentens_vec)
        predict_sentens = predict_sentens.reshape(self.seq_num,self.word_feat_len)
        # self.glaph_plot(predict_sentens)

        output_sentens = ""
        for value in predict_sentens:
            word = self.word2vec.get_word(value)
            output_sentens += (word+",")
            if (word == "。"): break

        print(output_sentens)
        print("")


    def make_sentens_input(self,inp):
        print(">> ",inp)
        sentens = get_words(inp)
        sentens_vec = self.sentens_to_vec(sentens)
        sentens_vec = self.EOFpadding(sentens_vec)
        sentens_vec = np.array(sentens_vec)
        sentens_vec = sentens_vec.reshape(1,self.seq_num,self.word_feat_len)

        # 次の特徴量を予測
        predict_sentens = self.seq2seq.predict(sentens_vec)
        predict_sentens = predict_sentens.reshape(self.seq_num,self.word_feat_len)

        output_sentens = ""
        for value in predict_sentens:
            word = self.word2vec.get_word(value)
            output_sentens += (word + ",")
            if (word == "。"): break
        print(output_sentens[::-1])
        return output_sentens[::-1]


    def glaph_plot(self,vec):
        i = 1
        fig = plt.figure()
        for value in vec:
            plt.subplot(5,5,i)
            t = range(len(value))
            plt.plot(t,value)
            i+=1


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
    rflag = ""
    # flag = "learn"
    # rflag = "resume"
    flag = "make"

    rnp_model = Trainer()

    word_lists = rnp_model.get_word_lists()

    rnp_model.init_word2vec()

    rnp_model.init_seq2seq()


    if rflag == "resume" :
        rnp_model.seq2seq.waitController("load")

    if flag == "learn":
        # lstm
        print("learninig lstm start")
        for i in range(rnp_model.learning_num):
            print("train step : ",i)
            train_sentens_vec_batch = []
            teach_sentens_vec_batch = []
            for j in range(rnp_model.batch_size):
                while(True):
                    index = random.randint(0,len(word_lists)-2)
                    train_sentens = word_lists[index][::-1]
                    teach_sentens = word_lists[index+1]
                    if( (len(train_sentens)<=rnp_model.seq_num) and (len(teach_sentens)<=rnp_model.seq_num) ): break

                train_sentens_vec = rnp_model.sentens_to_vec(train_sentens)
                teach_sentens_vec = rnp_model.sentens_to_vec(teach_sentens)

                train_sentens_vec = rnp_model.EOFpadding(train_sentens_vec)
                teach_sentens_vec = rnp_model.EOFpadding(teach_sentens_vec)

                train_sentens_vec_batch.append(train_sentens_vec)
                teach_sentens_vec_batch.append(teach_sentens_vec)

            train_sentens_vec_batch = np.array(train_sentens_vec_batch)
            teach_sentens_vec_batch = np.array(teach_sentens_vec_batch)

            rnp_model.seq2seq.train(train_sentens_vec_batch,teach_sentens_vec_batch)
            rnp_model.seq2seq.waitController("save")


    if flag == "make":
        rnp_model.word2vec.load_model()
        rnp_model.seq2seq.waitController("load")

        # 入力文から文章生成
        while(True):
            input_line = input(">> ")
            rnp_model.make_sentens_input(input_line)

        # for i in range(50):
        #     rnp_model.make_sentens(word_lists)
        # plt.show()

if __name__ == "__main__" :
    main()

