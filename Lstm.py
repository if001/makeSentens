"""

"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras import optimizers
from keras.models import Model
from keras import backend as K

import numpy as np
import random
import sys

# mylib
from Const import Const

class Lstm(Const):
    def __init__(self):
        super().__init__()
        self.word_seq_num = 1

    def input_len_set(self,worddict_len):
        self.input_len = self.word_feat_len * self.word_seq_num

        self.output_len = 3000
        self.hidden_neurons = 8000

        # self.input_len = worddict_len * window_size
        # self.output_len = worddict_len

    def make_net(self):
        print("make network")
        self.model = Sequential()

        # 単一層
        # self.model.add(LSTM(self.output_len, input_shape=(None, self.input_len)))
        # self.model.add(Dense(self.output_len))
        # self.model.add(Activation("relu"))
        # self.model.add(Dropout(0.5))

        # 多層
        self.model.add(LSTM(self.output_len, input_shape=(1, self.input_len),implementation=1,return_sequences=True))
        self.model.add(Dropout(0.5))

        # self.model.add(LSTM(self.hidden_neurons,return_sequences=True,implementation=1))
        # self.model.add(Dropout(0.5))

        self.model.add(LSTM(self.hidden_neurons,return_sequences=False,implementation=1))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_dim=self.output_len))
        self.model.add(Activation('tanh'))

        # self.model.add(Activation("linear"))
        # self.model.add(Activation("sigmoid"))
        # self.model.add(Activation("relu"))
        # self.model.add(Activation("softplus"))

        loss = 'categorical_crossentropy'
        loss = "binary_crossentropy"
        loss = "mean_squared_error"
        optimizer = RMSprop(lr=0.01)
        optimizer = "adam"
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()

    def train(self,X_train,Y_train):
        #self.history = self.model.train_on_batch(X_train,Y_train, class_weight=None, sample_weight=None)
        # print("lstm x shape",X_train.shape)
        # print("lstm y shape",Y_train.shape)
        self.hist = self.model.fit(X_train,Y_train,
                                   nb_epoch=1,
                                   batch_size = self.batch_size,
                                   validation_split=0.3,
                                   verbose=1)
        # print("lstm loss:",self.hist.history['loss'])
        # print("lstm val_loss:",self.hist.history['val_loss'])


    def predict(self,inp):
        inp = np.array(inp)
        inp = inp.reshape(1,1,self.input_len)
        #word_vec = self.model.predict(vec, batch_size=self.batch_size, verbose=0)
        #predict_list = self.model.predict(inp,batch_size=self.batch_size, verbose=0)
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



def main():
    lstm = Lstm()
    lstm.input_len_set(3000,1)
    lstm.make_net()


if __name__ == "__main__":
   main()
