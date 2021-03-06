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

class Trainer(lib.Const.Const):
    def __init__(self):
        super().__init__()
        self.window_size = 1
        self.hists = [[],[],[],[]]


    def init_word2vec(self,flag):
        self.word2vec = lib.WordVec.MyWord2Vec()

        if flag == "learn":
            self.word2vec.train(self.word2vec_train_file)
        elif flag == "load":
            self.word2vec.load_model()
        else:
            print("not word2vec model")
            exit(0)


    def fact_seq2seq(self):
        self.model = nn.Seq2SeqOfficial.Seq2Seq()
        self.model.make_net()
        self.model.model_complie()


    def fact_decode_net(self):
        self.model = nn.Seq2SeqOfficial.Seq2Seq()
        self.model.make_net()
        self.model.model_complie()
        self.model.make_decode_net()
        self.model.waitController('load', 'param_seq2seq.hdf5')


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


    def save_data(self, strs, fname):
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


    def load_test(self, word_lists, ds):
        for value in self.buckets:
            __train, __teach, __target  = ds.make_data(word_lists, self.batch_size, value)
            so = lib.StringOperation.StringOperation()
            __ev = self.model.sequence_autoencoder.evaluate([__train, __teach], __target, batch_size = self.batch_size, verbose=1)
            print(value, " : ", __ev)


def get_word_lists(file_path):
    print("make wordlists")
    # lines = open(file_path).read().split("。")
    lines = open(file_path).read().split("\n")
    wordlists = []
    for line in lines:
        wordlists.append(line.split(" "))

    print("wordlist num:",len(wordlists))
    return wordlists


def train_main(tr):
    tr.init_word2vec("load")
    # tr.init_word2vec("learn")

    tr.fact_seq2seq()

    ds = lib.DataShaping.DataShaping()

    word_lists = get_word_lists(lib.Const.Const().seq2seq_train_file)

    if '--resume' in sys.argv:
        print('resume param_seq2seq.hdf5')
        tr.model.waitController('load', 'param_seq2seq.hdf5')

    for step in range(lib.Const.Const().learning_num):
        chose_bucket = tr.select_random_bucket()
        print("chose bucket", chose_bucket)
        train_data, teach_data, teach_target_data = ds.make_data(word_lists, tr.batch_size, chose_bucket)

        print("train step : ", step)
        hist = tr.model.train(train_data, teach_data, teach_target_data)
        tr.hists = tr.append_hist(hist, tr.hists)

        if (step % tr.check_point == 0) and (step != 0):
            # tr.plot(tr.hists, str(chose_bucket[0])+"_"+str(chose_bucket[1]))
            tr.model.waitController('save', 'param_seq2seq.hdf5')
            tr.load_test(word_lists, ds)


def make_sentens_main(tr):
    tr.init_word2vec("load")
    word_lists = get_word_lists(lib.Const.Const().seq2seq_train_file)
    ds = lib.DataShaping.DataShaping()
    so = lib.StringOperation.StringOperation()


    tr.fact_seq2seq()

    tr.model.waitController('load', 'param_seq2seq.hdf5')
    tr.load_test(word_lists, ds)
    tr.model.make_decode_net()

    for i in range(10):
        chose_bucket = tr.select_random_bucket()
        sentens_arr_vec, _, _ = ds.make_data(word_lists, 1, chose_bucket)

        __sentens_arr = so.sentens_vec_to_sentens_arr(sentens_arr_vec[0])
        print(">> ",so.sentens_array_to_str(__sentens_arr[::-1]))

        states_value = tr.model.encoder_model.predict(sentens_arr_vec)
        decoder_model = tr.model.decoder_model

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
