"""

"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout, advanced_activations

from keras import optimizers

import pylab as plt
import numpy as np
import random
import sys

from mecab_test import get_words

# mylib
from Const import Const

class myVec2Word(Const):
    def __init__(self):
        super().__init__()
        self.loss = []
        self.val_loss = []

    def neuron_num_set(self,word_dict):
        self.input_len = self.word_feat_len
        self.hidden_len1 = self.word_feat_len
        self.hidden_len2 = self.word_feat_len*2
        self.output_len = len(word_dict)

    def leakRelu(self):
        advanced_activations.LeakyReLU(alpha=0.3)

    def make_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_len1 , input_shape=(self.input_len,)))
        self.model.add(Activation(self.leakRelu()))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(self.hidden_len2))
        self.model.add(Activation(self.leakRelu()))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.output_len))
        self.model.add(Activation('softmax'))

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loss = "binary_crossentropy"
        loss = 'categorical_crossentropy'

        self.model.compile(loss=loss, optimizer=adam,metrics=['accuracy'])
        self.model.summary()


    def train_net(self,train_x,train_y):
        # モデルの訓練
        # self.history = self.model.train_on_batch(train_x,train_y,
        #                                          class_weight=None, sample_weight=None)
        self.hist = self.model.fit(train_x,train_y,
                                      nb_epoch=self.batch_size,
                                      validation_split=0.3,
                                      verbose=1)

        print("vec to word loss:",self.hist.history['loss'])
        print("vec to word val_loss:",self.hist.history['val_loss'])


    def vec_to_word(self,vec):
        vec = vec.reshap(1,len(vec))
        word_vec = self.model.predict(vec, batch_size=self.batch_size, verbose=0)
        # word_vec = self.model.predict_on_batch(vec)
        # hidden_layer_output = K.function([self.model.layers[2].input],
        #                                   [self.model.layers[3].output])
        # word_vec = hidden_layer_output([vec])[0]
        return word_vec

    def set_glaph(self):
        self.loss = np.append(self.loss,np.average(self.hist.history['loss']))
        self.val_loss = np.append(self.val_loss,np.average(self.hist.history['val_loss']))

    def glaph_plot(self):
        t = range(len(self.loss))
        plt.plot(t,self.loss,label = "loss")
        plt.plot(t,self.val_loss,label = "val_loss")
        plt.legend()
        filename = "vec2word.png"
        plt.savefig(filename)
        #plt.show()

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
