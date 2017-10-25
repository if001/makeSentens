
import sys
import numpy as np
import matplotlib.pylab as plt
from keras.models import Model

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
# from keras.layers.wrappers import TD
from keras.optimizers import Adam


from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.layers          import merge, multiply


# mylib
import lib

class Seq2Seq(lib.Const.Const):
    def __init__(self):
        super().__init__()
        self.input_word_num = 1
        self.hidden_dim = 7000

    def make_net(self):
        # ---------------
        latent_dim = 1000
        inputs = Input(shape=(self.seq_num, self.word_feat_len))
        encoded = LSTM(latent_dim)(inputs)
        decoded = RepeatVector(self.seq_num)(encoded)
        decoded = LSTM(self.word_feat_len, return_sequences=True)(decoded)

        self.sequence_autoencoder = Model(inputs, decoded)
        # encoder = Model(inputs, encoded)
        # ---------------



        # ---- modelで組む ------
        # model=Sequential()
        # model.add(LSTM(units = self.hidden_dim, input_shape=(self.seq_num, self.word_dim)))

        # seq_out_length = 1 #????
        # #decoder
        # model.add(RepeatVector(seq_out_length))
        # model.add(LSTM(units=self.hidden_dim, return_sequences=True))

        # model.add(TD(Dense(units=self.word_dim)))
        # model.add(Activation('softmax'))
        # ---- modelで組む ------


        # ---------------
        # timesteps   = self.input_length
        # latent_dim  = 2000
        # inputs      = Input(shape=(timesteps, self.word_dim))
        # encoded     = LSTM(latent_dim)(inputs)

        # a_vector    = Dense(latent_dim, activation='tanh')(Flatten()(inputs))
        # #a_vector    = Dense(latent_dim, activation='softmax')(Flatten()(inputs))
        # mul         = multiply([encoded, a_vector])
        # # encoder     = Model(inputs, mul)
        # x           = RepeatVector(timesteps)(mul)
        # x           = Bi(LSTM(latent_dim, return_sequences=True))(x)
        # #decoded     = TD(Dense(self.word_dim, activation='softmax'))(x)
        # decoded     = TD(Dense(self.word_dim, activation='tanh'))(x)
        # ---------------



        # a_vector    = Dense(latent_dim, activation='tanh')(Flatten()(inputs))
        # #a_vector    = Dense(latent_dim, activation='softmax')(Flatten()(inputs))
        # mul         = multiply([encoded, a_vector])
        # # encoder     = Model(inputs, mul)
        # x           = RepeatVector(timesteps)(mul)
        # x           = Bi(LSTM(latent_dim, return_sequences=True))(x)
        # #decoded     = TD(Dense(self.word_dim, activation='softmax'))(x)
        # decoded     = TD(Dense(self.word_dim, activation='tanh'))(x)
        # ---------------


        self.sequence_autoencoder = Model(inputs, decoded)



        loss = 'mean_squared_error'
        optimizer = 'rmsprop'
        optimizer = 'Adam'
        # self.model.compile(loss=loss, optimizer=optimizer)
        # self.model.summary()
        self.sequence_autoencoder.compile(loss=loss, optimizer=optimizer)
        self.sequence_autoencoder.summary()



    def train(self,X_train,Y_train):
        self.sequence_autoencoder.fit(X_train, Y_train,
                       shuffle=True,
                       nb_epoch=5,
                       batch_size=self.batch_size,
                       validation_split=0.1,
                       verbose=1)

        # self.model.fit(X_train, Y_train,
        #                shuffle=True,
        #                nb_epoch=1,
        #                batch_size=self.batch_size,
        #                validation_split=0.1,
        #                verbose=1)


    def predict(self,inp):
        predict_list = self.sequence_autoencoder.predict_on_batch(inp)
        return predict_list

    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.sequence_autoencoder.save_weights(self.seq2seq_wait_save_dir)

            if flag == "load":
                print("load")
                self.sequence_autoencoder.load_weights(self.seq2seq_wait_save_dir)
        except :
            print(self.seq2seq_wait_save_dir+" dose not exist")
            sys.exit(0)



def main():
    seq2seq = Seq2Seq()
    seq2seq.make_net()

    word = np.array([1,2,3])
    word2 = np.array([2,3,4])

    # one of train
    input_vec = []
    input_vec.append(word)
    input_vec.append(word2)
    input_vec = np.array(input_vec)

    # one of train
    input_vec2 = []
    input_vec2.append(word)
    input_vec2.append(word2)
    input_vec2 = np.array(input_vec)

    # batch
    batch_input_vec = []
    batch_input_vec.append(input_vec)
    batch_input_vec.append(input_vec2)
    batch_input_vec = np.array(batch_input_vec)
    print(batch_input_vec.shape)

    # train
    seq2seq.train(batch_input_vec,batch_input_vec)

    # test1
    test_input = np.array([input_vec])
    print(test_input)
    predict = seq2seq.predict(test_input)
    print(predict)


if __name__ == "__main__":
   main()
