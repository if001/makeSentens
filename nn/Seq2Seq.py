
import sys
import numpy as np
#import matplotlib.pylab as plt
from keras.models import Model

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
# from keras.layers.wrappers import TD

from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector, concatenate, Dropout
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.layers          import merge, multiply
from keras.optimizers import Adam,SGD,RMSprop

from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization as BN

from keras import regularizers


import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# mylib
import lib


class Seq2Seq(lib.Const.Const):
    def __init__(self,encord_len,decord_len):
        super().__init__()
        self.input_word_num = 1
        # self.hidden_dim = 7000
        self.encord_len = encord_len
        self.decord_len = decord_len

    def make_net(self):
        # ---------------
        # テストパラメタ
        # input_dim = 20
        # latent_dim = 30
        # hidden_dim1 = 20
        # hidden_dim2 = 20
        # output_dim = 20

        input_dim = self.word_feat_len
        latent_dim = 512
        hidden_dim1 = 256
        hidden_dim1 = 512
        hidden_dim1 = 750
        hidden_dim2 = 64
        hidden_dim3 = 32
        output_dim = self.word_feat_len

        inputs = Input(shape=(self.encord_len, input_dim))
        encoded = LSTM(latent_dim,activation="tanh",recurrent_activation="sigmoid",return_sequences=False)(inputs)
        # encoded = Dense(hidden_dim1, activation="relu")(encoded)
        # encoded = Dropout(0.4)(encoded)
        encoded = Dense(hidden_dim1, activation="linear",
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01))(encoded)
        # encoded = Dense(hidden_dim1, activation="linear")(encoded)


        decoded = RepeatVector(self.decord_len)(encoded)
        decoded = LSTM(latent_dim, activation="tanh",recurrent_activation="sigmoid",return_sequences=True)(decoded)
        #decoded = Dense(hidden_dim1, activation="relu")(decoded)
        #decoded = Dropout(0.8)(decoded)
        # decoded = Dense(hidden_dim2, activation="relu")(decoded)
        # decoded = Dropout(0.5)(decoded)
        # decoded = Dense(hidden_dim2, activation="relu")(decoded)
        # decoded = Dropout(0.2)(decoded)
        #encoded = Dense(hidden_dim1,activation="linear")
        # encoded = Dense(output_dim, activation="linear",
        #                 kernel_regularizer=regularizers.l2(0.01),
        #                 activity_regularizer=regularizers.l1(0.01))(encoded)
        decoded = Dense(output_dim, activation="linear")(decoded)

        self.sequence_autoencoder = Model(inputs, decoded)


        optimizer = 'rmsprop'
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        #optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
        # optimizer = 'Adam'
        loss = 'mean_squared_error'
        loss = 'mean_squared_error'
        loss = 'mean_squared_error'
        #loss = 'kullback_leibler_divergence'

        self.sequence_autoencoder.compile(optimizer=optimizer,
                                          loss=loss,
                                          metrics=['accuracy'])

        self.sequence_autoencoder.summary()



    def train(self,X_train,Y_train):
        es_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
        self.sequence_autoencoder.fit(X_train, Y_train,
                                      shuffle=True,
                                      #nb_epoch=9,
                                      nb_epoch=200,
                                      batch_size=self.batch_size,
                                      validation_split=0.1,
                                      verbose=1,
                                      callbacks=[es_cb])

        # self.model.fit(X_train, Y_train,
        #                shuffle=True,
        #                nb_epoch=1,
        #                batch_size=self.batch_size,
        #                validation_split=0.1,
        #                verbose=1)


    def predict(self,inp):
        predict_list = self.sequence_autoencoder.predict_on_batch(inp)
        return predict_list

    def waitController(self,flag,fname):
        if flag == "save":
            print("save"+self.seq2seq_wait_save_dir+fname)
            self.sequence_autoencoder.save_weights(self.seq2seq_wait_save_dir+fname)
        if flag == "load":
            print("load"+self.seq2seq_wait_save_dir+fname)
            self.sequence_autoencoder.load_weights(self.seq2seq_wait_save_dir+fname)


def main():
    seq2seq = Seq2Seq(5,5)
    seq2seq.make_net()


    import random

    inp_batch = []
    out_batch = []
    for value in range(seq2seq.batch_size):
        word_len = 20
        sentens_len = 5
        sentens = []
        out_sentens = []
        for j in range(sentens_len):
            one_word = []
            num = random.randint(0,10)/10
            for i in range(word_len):
                one_word.append(num+i/10)
            sentens.append(one_word)
            out_sentens.append(one_word[::-1])
        inp_batch.append(sentens)
        out_batch.append(out_sentens)

    inp_batch = np.array(inp_batch)
    out_batch = np.array(out_batch)
    for i in range(10):
        seq2seq.train(inp_batch,out_batch)



    for value in range(10):
        inp_batch = []
        word_len = 20
        sentens_len = 5
        sentens = []
        for j in range(sentens_len):
            one_word = []
            num = random.randint(0,10)/10
            for i in range(word_len):
                one_word.append(num+i/10)
            sentens.append(one_word)
        inp_batch.append(sentens)

        inp_batch = np.array(inp_batch)

        predict = seq2seq.predict(inp_batch)
        print("inp :",inp_batch)
        print("test:",predict)



if __name__ == "__main__":
   main()
