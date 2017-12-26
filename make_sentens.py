'''
lstmを使って文章生成
word2vecを自作して、vector化
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

# class Trainer(lib.Const.Const):
#     def __init__(self):
#         super().__init__()
#         self.window_size = 1
#         self.hists = [[],[],[],[]]


#     def make_sentens_vec(self, decoder_model, states_value, start_token, end_token, end_len):
#         sentens_vec = []
#         word_vec = start_token

#         stop_condition = False
#         while not stop_condition:
#             word_vec, h, c = decoder_model.predict([word_vec] + states_value)
#             sentens_vec.append(word_vec.reshape(self.word_feat_len))
#             states_value = [h, c]
#             if (np.allclose(word_vec, end_token) or len(sentens_vec) == end_len ):
#                 stop_condition = True
#         return sentens_vec


    # def append_hist(self, hist, hists):
    #     hists[0].append(hist.history['acc'][0])
    #     hists[1].append(hist.history['val_acc'][0])

    #     hists[2].append(hist.history['loss'][0])
    #     hists[3].append(hist.history['val_loss'][0])
    #     return hists


    # def save_data(self, strs, fname):
    #     with open("./fig/"+fname, "w") as file:
    #         file.writelines(str(strs))


    # def load_test(self, word_lists, ds):
    #     for value in self.buckets:
    #         __train, __teach, __target  = ds.make_data(word_lists, self.batch_size, value)
    #         so = lib.StringOperation.StringOperation()
    #         __ev = self.model.sequence_autoencoder.evaluate([__train, __teach], __target, batch_size = self.batch_size, verbose=1)
    #         print(value, " : ", __ev)

def init_word2vec(const, flag):
    word2vec = lib.WordVec.MyWord2Vec()
    if flag == "learn":
        word2vec.train(const.word2vec_train_file)
    elif flag == "load":
        word2vec.load_model()
    else:
        print("not word2vec model")
        exit(0)


def fact_seq2seq(hred):
    context_h = hred.build_context_net()
    context_c = hred.build_context_net()
    autoencoder = hred.build_net(context_h, context_c)

    context_h =  hred.model_complie(context_h)
    context_c = hred.model_complie(context_c)
    autoencoder = hred.model_complie(autoencoder)
    return context_h, context_c, autoencoder


def select_random_bucket(const):
    rnd = random.randint(0, len(const.buckets)-1)
    return const.buckets[rnd]


def get_word_lists(file_path):
    print("make wordlists")
    # lines = open(file_path).read().split("。")
    lines = open(file_path).read().split("\n")
    wordlists = []
    for line in lines:
        wordlists.append(line.split(" "))

    print("wordlist num:",len(wordlists))
    return wordlists


def pre_train_main():
    const = lib.Const.Const()
    # init_word2vec(const, "load")
    init_word2vec(const, "learn")

    ds = lib.DataShaping.DataShaping()
    hred = nn.HRED.HRED()

    word_lists = get_word_lists(lib.Const.Const().seq2seq_train_file)

    if '--resume' in sys.argv:
        encoder_model = hred.load_models('param_seq2seq_encoder.hdf5')
        decoder_model = hred.load_models('param_seq2seq_decoder.hdf5')
    else :
        encoder_model = hred.build_encoder()
        decoder_model = hred.build_decoder()

    autoencoder = hred.build_autoencoder(encoder_model, decoder_model)
    autoencoder = hred.model_compile(autoencoder)

    for step in range(const.learning_num):
        print("pre train step : ", step)
        chose_bucket = select_random_bucket(const)
        print("chose bucket", chose_bucket)

        train_data, teach_data, teach_target_data = ds.make_data(word_lists, const.batch_size, chose_bucket)
        hred.train_autoencoder(autoencoder, train_data, teach_data, teach_target_data)
        hred.save_models('param_seq2seq_encoder.hdf5', encoder_model)
        hred.save_models('param_seq2seq_decoder.hdf5', decoder_model)


def train_main():
    const = lib.Const.Const()
    # init_word2vec(const, "load")
    init_word2vec(const, "learn")

    ds = lib.DataShaping.DataShaping()
    hred = nn.HRED.HRED()

    word_lists = get_word_lists(lib.Const.Const().seq2seq_train_file)


    if '--resume' in sys.argv:
        encoder_model = hred.load_models('param_seq2seq_encoder.hdf5')
        context_h = hred.load_models('param_seq2seq_h.hdf5')
        context_c = hred.load_models('param_seq2seq_c.hdf5')
    else :
        encoder_model = hred.build_encoder()
        decoder_model = hred.build_decoder()
        context_h = hred.build_context_model()
        context_c = hred.build_context_model()

        encoder_model = hred.model_compile(encoder_model)
        context_h =  hred.model_compile(context_h)
        context_c = hred.model_compile(context_c)

    decoder_model = hred.model_compile(decoder_model)

    for step in range(const.learning_num):
        print("train step : ", step)
        chose_bucket = select_random_bucket(const)
        print("chose bucket", chose_bucket)

        state_h_batch = []
        state_c_batch = []
        train_data, teach_data, teach_target_data = ds.make_data(word_lists, const.context_size, chose_bucket)
        state_h, state_c = encoder_model.predict(train_data)
        state_h_batch.append(state_h)
        state_c_batch.append(state_c)



        # print("train context")
        # train_data, teach_data, teach_target_data = ds.make_data(word_lists, const.context_size, chose_bucket)
        # hred.train(context_h, train_data, teach_data)
        # hred.train(context_c, train_data, teach_data)

        # print("train autoencoder")
        # train_data, teach_data, teach_target_data = ds.make_data(word_lists, const.batch_size, chose_bucket)
        # hred.train(autoencoder, train_data, teach_data, teach_target_data)



        # hred.train(context_h, )
        # hred.train(context_c, )
    
        # tr.hists = tr.append_hist(hist, tr.hists)
        # if (step % tr.check_point == 0) and (step != 0):
        #     # tr.plot(tr.hists, str(chose_bucket[0])+"_"+str(chose_bucket[1]))
        #     tr.model.waitController('save', 'param_seq2seq.hdf5')
        #     tr.load_test(word_lists, ds)


def make_sentens_main(tr):
    tr.init_word2vec("load")
    word_lists = get_word_lists(lib.Const.Const().seq2seq_train_file)
    ds = lib.DataShaping.DataShaping()
    so = lib.StringOperation.StringOperation()

    tr.model.waitController('load', 'param_seq2seq.hdf5')
    # tr.load_test(word_lists, ds)
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
    pre_train_main()
    exit(0)

    if '--train' in sys.argv:
        train_main()

    elif '--make' in sys.argv:
        make_sentens_main()

    else:
        print("consol execute flag is invalid!")


if __name__ == "__main__" :
    main()
