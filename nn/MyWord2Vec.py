"""

参考；https://www.slideshare.net/okamoto-laboratory/word2vec-70898693
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout, advanced_activations

from keras import optimizers
from keras import backend as K

import pylab as plt
import numpy as np
import random
import sys

# mylib
from Const import Const

class myWord2Vec(Const):
    def __init__(self):
        super().__init__()
        self.window_size = 3 # 0はだめ
        self.loss = []
        self.val_loss = []

    def neuron_num_set(self,word_dict):
        self.input_len = len(word_dict)
        #self.hidden_len1 = self.word_feat_len*2
        self.hidden_len2 = self.word_feat_len
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

    def leakRelu(self):
        advanced_activations.LeakyReLU(alpha=0.3)

    def make_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_len2 , input_shape=(self.input_len,)))
        self.model.add(Activation('linear'))
        # self.model.add(Activation(self.leakRelu()))
        # 
        # self.model.add(Dropout(0.5))

        # self.model.add(Dense(self.hidden_len2))
        # self.model.add(Activation(self.leakRelu()))
        # self.model.add(Activation('relu'))

        self.model.add(Dense(self.output_len))
        # self.model.add(Activation('sigmoid'))
        self.model.add(Activation('softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss = 'categorical_crossentropy'
        loss = "binary_crossentropy"

        self.model.compile(loss=loss, optimizer=sgd,metrics=['accuracy'])

        self.model.summary()


    def output_hidden_layer(self):
        # 中間層を出力するためのモデル形成
        self.get_hidden_output = K.function([self.model.layers[0].input,K.learning_phase()],
                                          [self.model.layers[1].output])


    def input_hidden_layer(self):
        # 中間層へ入力するためのモデル形成
        self.input_hidden_get_output = K.function([self.model.layers[1].input,K.learning_phase()],
                                          [self.model.layers[3].output])


    def word_to_vec(self,input_word):
        input_word = np.array(input_word)
        input_word = input_word.reshape(1,len(input_word))
        # 0でテストモード、1でtrainモード
        layer_output = self.get_hidden_output([input_word,0])[0]
        return layer_output


    def vec_to_word(self,input_word):
        input_word = np.array(input_word)

        input_word = input_word.reshape(1,len(input_word))
        # 0でテストモード、1でtrainモード
        layer_output = self.input_hidden_get_output([input_word,0])[0]
        return layer_output


    def train_net(self,train_x,train_y):
        # モデルの訓練
        # self.history = self.model.train_on_batch(train_x,train_y,
        #                                          class_weight=None, sample_weight=None)

        self.hist = self.model.fit(train_x,train_y,
                                      nb_epoch=self.batch_size,
                                      validation_split=0.3,
                                      verbose=0)
        # self.loss = np.append(self.loss,self.hist.history['loss'])
        # self.val_loss = np.append(self.val_loss,self.hist.history['val_loss'])
        print("vec to word loss:",self.hist.history['loss'])
        print("vec to word val_loss:",self.hist.history['val_loss'])

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

    def set_glaph(self):
        self.loss = np.append(self.loss,np.average(self.hist.history['loss']))
        self.val_loss = np.append(self.val_loss,np.average(self.hist.history['val_loss']))

    def glaph_plot(self):
        t = range(len(self.loss))
        plt.plot(t,self.loss,label = "loss")
        plt.plot(t,self.val_loss,label = "val_loss")
        plt.legend()
        filename = "word2vec.png"
        plt.savefig(filename)
        #plt.show()


def main(): pass


if __name__ == "__main__":
   main()
