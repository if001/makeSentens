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

class myVec2Word():
    def __init__(self):pass

    def neuron_num_set(self,word_dict):
        self.input_len = word_f
        self.hidden_len = 3000
        self.output_len = len(word_dict)

    def make_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_len , input_shape=(self.input_len,)))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_len))
        self.model.add(Activation('softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loss = 'categorical_crossentropy'
        loss = "binary_crossentropy"

        self.model.compile(loss=loss, optimizer=sgd)
        self.model.summary()


    def train_net(self,train_x,train_y):
        # モデルの訓練
        self.history = self.model.train_on_batch(train_x,train_y,
                                                 class_weight=None, sample_weight=None)
        print("vec to word:",self.history,", ",end="")

    def vec_to_word(self,vec):
        # inp = np.array(vec)
        # inp = inp.reshape(1,1,self.input_len)
        word_vec = self.model.predict_on_batch(vec)

        # hidden_layer_output = K.function([self.model.layers[2].input],
        #                                   [self.model.layers[3].output])

        # word_vec = hidden_layer_output([vec])[0]
        return word_vec

    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.model.save_weights('./wait/param_make_sentens_wordvec_vec2word.hdf5')
            if flag == "load":
                print("load")
                self.model.load_weights('./wait/param_make_sentens_wordvec_vec2word.hdf5')
        except :
            print("no such file")
            sys.exit(0)
