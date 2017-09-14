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


class myWord2Vec():
    def __init__(self):
        # 0はだめ
        self.window_size = 3

    def neuron_num_set(self,word_dict):
        self.input_len = len(word_dict)
        self.hidden_len = word_f
        self.output_len = len(word_dict)*(self.window_size*2)

    def get_words_W2V(self,word_lists):
        sentens_num = 3
        # ランダムに文章選択
        r1 = random.randint(0,len(word_lists)-1-sentens_num)
        r2 = r1 + 1
        r3 = r1 + 2
        tmp_word_lists = word_lists[r1] + word_lists[r2] + word_lists[r3]

        r4 = random.randint(0,len(tmp_word_lists)-(self.window_size*2+1)-1)

        return tmp_word_lists[r4:r4+(self.window_size)*2+1 ]

        # # 文章内から1語選択
        # r2 = random.randint(0,len(word_lists[r])-(self.window_size*2+1)-1)
        # # 選択した文章からwindowsize文単語を取ってくる
        # return word_lists[r][r2:r2+(self.window_size)*2+1 ]


    def make_one_hot(self,word_dict,word):
        # print("make one hot vector")
        self.wordvec_one_hot = [0 for i in range(len(word_dict))]
        self.wordvec_one_hot[word_dict.index(word)] = 1
        return self.wordvec_one_hot


    def make_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_len , input_shape=(self.input_len,)))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_len))
        self.model.add(Activation('sigmoid'))
        #self.model.add(Activation('softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loss = 'categorical_crossentropy'
        loss = "binary_crossentropy"

        self.model.compile(loss=loss, optimizer=sgd)
        self.model.summary()


    def hidden_layer(self):
        # 中間層を出力するためにモデルを形成
        # with a Sequential model
        self.get_hidden_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[1].output])

        # layer_name = 'my_layer'
        # self.intermediate_layer_model = Model(inputs = self.model.input,outputs = self.model.get_layer(layer_name).output)


    def word_to_vec(self,input_word):
        input_word = np.array(input_word)
        input_word = input_word.reshape(1,len(input_word))
        layer_output = self.get_hidden_layer_output([input_word])[0]
        return layer_output


    def train_net(self,train_x,train_y):
        # モデルの訓練
        self.history = self.model.train_on_batch(train_x,train_y,
                                                 class_weight=None, sample_weight=None)
        print("word to vec",self.history,", ",end="")



    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.model.save_weights('./wait/param_make_sentens_wordvec.hdf5')
            if flag == "load":
                print("load")
                self.model.load_weights('./wait/param_make_sentens_wordvec.hdf5')
        except :
            print("no such file")
            sys.exit(0)
