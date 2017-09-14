"""

"""

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


class Lstm():
    def __init__(self):
        self.window_size = 1

    def input_len_set(self,worddict_len,window_size):
        self.input_len = word_f * window_size
        self.output_len = 3000
        self.hidden_neurons = 5000
        # self.input_len = worddict_len * window_size
        # self.output_len = worddict_len

    def make_net(self):
        print("make network")
        self.model = Sequential()

        self.model.add(LSTM(self.hidden_neurons, input_shape=(1, self.input_len)))

        self.model.add(Dense(self.output_len))
        self.model.add(Activation("softmax"))

        loss = 'categorical_crossentropy'
        loss = "mean_squared_error"
        loss = "binary_crossentropy"
        optimizer = "adam"
        optimizer = RMSprop(lr=0.01)

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()

    def train(self,X_train,Y_train):
        self.history = self.model.train_on_batch(X_train,Y_train, class_weight=None, sample_weight=None)
        print("lstm :",self.history)

    def predict(self,inp):
        inp = np.array(inp)
        inp = inp.reshape(1,1,self.input_len)
        predict_list = self.model.predict_on_batch(inp)

        return predict_list


    def netScore(self,X_train,Y_train):
        self.score = self.model.evaluate(X_train, Y_train, verbose=0)
        print("lstm : ",self.score)
        # print('test loss:', self.score[0])
        # print('test acc:', self.score[1])

    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.model.save_weights('./wait/param_make_sentens_wordvec_lstm.hdf5')
            if flag == "load":
                print("load")
                self.model.load_weights('./wait/param_make_sentens_wordvec_lstm.hdf5')
        except :
            print("no such file")
            sys.exit(0)
