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


import matplotlib.pyplot as plt
#import pylab as plt
from itertools import chain #配列ふらっと化
import random


# mylib
import lib
import nn

# cython
#import pyximport; pyximport.install()
#import cythonFunc
import cython_package.cython_package as cy

# TMP_BUCKET = (20,25)
# TMP_BUCKET = (10,15)

class Trainer(lib.Const.Const):
    def __init__(self):
        super().__init__()
        self.window_size = 1
        self.models = []
        self.hists = [[],[],[],[]]


    def init_word2vec(self,flag):
        self.word2vec = lib.WordVec.MyWord2Vec()
        if flag == "learn":
            self.word2vec.train(self.dict_train_file)
        elif flag == "load":
            self.word2vec.load_model()
        else:
            print("not word2vec model")
            exit(0)


    def fact_seq2seq(self,encord_len,decord_len):
        """
        # buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        """
        for value in self.buckets :
            self.models.append(nn.Seq2SeqOfficial.Seq2Seq(value[0],value[1]))
        self.models[-1].make_net()
        self.models[-1].model_complie()


    def select_random_bucket(self):
        rnd = random.randint(0, len(self.buckets)-1)
        return self.buckets[rnd]


    def make_sentens_vec(self, decoder_model, states_value, start_token, end_token, end_len):
        sentens_vec = []
        word_vec = start_token

        stop_condition = False
        while not stop_condition:
            word_vec, h, c = decoder_model.predict([word_vec] + states_value)
            sentens_vec.append(word_vec.reshape(self.word_feat_len))
            states_value = [h, c]
            if (np.allclose(word_vec, end_token) or len(sentens_vec) == end_len ):
                stop_condition = True

        return sentens_vec

    def append_hist(self, hist, hists):
        hists[0].append(hist.history['acc'][0])
        hists[1].append(hist.history['val_acc'][0])

        hists[2].append(hist.history['loss'][0])
        hists[3].append(hist.history['val_loss'][0])
        return hists

    def save_data(strs,fname):
        with open("./fig/"+fname, "w") as file:
            file.writelines(str(strs))

    def plot(self, hists, save_name):
        labels = ["acc", "val_acc", "loss", "val_loss"]
        color = ["r", "g", "b", "y"]
        # acc
        plt.figure(1)
        for i in range(2):
            t = range(len(hists[i]))
            plt.plot(t, hists[i], label=labels[i], color=color[i])
            self.save_data(hists[i], labels[i])
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid()
        plt.savefig("./fig/graph_acc_"+save_name+".png")
        plt.clf() #一度消去
        plt.cla() #一度消去


        plt.figure(2)
        for i in range(2,4):
            t = range(len(hists[i]))
            plt.plot(t, hists[i], label=labels[i], color=color[i])
            self.save_data(hists[i], labels[i])
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.savefig("./fig/graph_loss_"+save_name+".png")
        print("save " + save_name + " glaph")
        plt.close()

def get_word_lists(file_path):
    print("make wordlists")
    lines = open(file_path).read().split("。")
    wordlists = []
    for line in lines:
        wordlists.append(line.split(" "))

    print("wordlist num:",len(wordlists))
    return wordlists


def train_main(tr):
    if '--resume' in sys.argv:
        tr.init_word2vec("load")
    else:
        tr.init_word2vec("learn")

    ds = lib.DataShaping.DataShaping()

    word_lists = get_word_lists(lib.Const.Const().dict_train_file)

    for value in tr.buckets:
        print("start bucket ", value)
        tr.fact_seq2seq(value[0], value[1])


    while(True):
        chose_bucket = tr.select_random_bucket()
        print("chose bucket", chose_bucket)
        if '--resume' in sys.argv:
            print("resume "+'param_seq2seq_rnp'+"_"+str(chose_bucket[0])+"_"+str(chose_bucket[1])+'.hdf5')
            tr.models[-1].waitController('load', 'param_seq2seq_rnp'+"_"+str(chose_bucket[0])+"_"+str(chose_bucket[1])+'.hdf5')

        train_data, teach_data, teach_target_data = ds.make_data(word_lists, tr.batch_size, chose_bucket)

        print("learninig start")
        for step in range(lib.Const.Const().learning_num):
            print("train step : ", step)
            hist = tr.models[-1].train(train_data, teach_data, teach_target_data)
            tr.hists = tr.append_hist(hist, tr.hists)

            if (step % tr.check_point == 0) and (step != 0):
                tr.plot(tr.hists, str(chose_bucket[0])+"_"+str(chose_bucket[1]))
                tr.models[-1].waitController('save','param_seq2seq_rnp'+"_"+str(chose_bucket[0])+"_"+str(chose_bucket[1])+'.hdf5')


def make_sentens_main(tr):
    tr.init_word2vec("load")
    ds = lib.DataShaping.DataShaping()
    so = lib.StringOperation.StringOperation()

    word_lists = get_word_lists(lib.Const.Const().dict_load_file)

    for value in tr.buckets:
        tr.fact_seq2seq(value[0],value[1])
        tr.models[-1].waitController('load', 'param_seq2seq_rnp'+"_"+str(value[0])+"_"+str(value[1])+'.hdf5')
        tr.models[-1].make_decode_net()


    for i in range(10):
        chose_bucket = tr.select_random_bucket()
        sentens_arr_vec, _, _ = ds.make_data(word_lists, tr.batch_size, chose_bucket)
        __sentens_arr = so.sentens_vec_to_sentens_arr(sentens_arr_vec[0])
        print(">> ",so.sentens_array_to_str(__sentens_arr[::-1]))

        states_value = tr.models[tr.buckets.index(chose_bucket)].encoder_model.predict(sentens_arr_vec)
        decoder_model = tr.models[tr.buckets.index(chose_bucket)].decoder_model

        start_token = so.sentens_array_to_vec(["BOS"])
        start_token = np.array([start_token])
        end_token = so.sentens_array_to_vec(["。"])
        end_token = np.array([end_token])

        decord_sentens_vec = tr.make_sentens_vec(decoder_model, states_value, start_token, end_token, chose_bucket[1])

        decord_sentens_arr = so.sentens_vec_to_sentens_arr(decord_sentens_vec)
        sentens = so.sentens_array_to_str(decord_sentens_arr)
        print(sentens)
        print("--")


def main():
    tr = Trainer()

    if '--train' in sys.argv:
        train_main(tr)

    elif '--make' in sys.argv:
        make_sentens_main(tr)

    else:
        print("consol execute flag is invalid!")


if __name__ == "__main__" :
    main()
