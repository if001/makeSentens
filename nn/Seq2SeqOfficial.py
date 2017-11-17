import numpy as np
#import matplotlib.pylab as plt
from keras.models import Model

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
# from keras.layers.wrappers import TD

from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector, concatenate, Dropout, Bidirectional
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
    """This is a test program."""

    def __init__(self, encord_len, decord_len):
        super().__init__()
        self.input_word_num = 1
        # self.hidden_dim = 7000
        self.latent_dim = 30
        self.encord_len = encord_len
        self.decord_len = decord_len


    def make_net(self):
        """ make net by reference to Keras official doc """
        # input_dim = 20
        # output_dim = 20

        input_dim = self.word_feat_len
        output_dim = self.word_feat_len

        encoder_inputs = Input(shape=(None, input_dim))
        encoder_outputs, state_h, state_c = LSTM(self.latent_dim, return_state=True)(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, input_dim))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(output_dim, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.sequence_autoencoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense


    def make_decode_net(self, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense):
        """ for decoding net """

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        return encoder_model, decoder_model


    def model_complie(self):
        """ complie """
        optimizer = 'rmsprop'
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        #optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
        # optimizer = 'Adam'
        loss = 'mean_squared_error'
        self.sequence_autoencoder.compile(optimizer=optimizer,
                                          loss=loss,
                                          metrics=['accuracy'])

        self.sequence_autoencoder.summary()


    def train(self, encoder_input_data, decoder_input_data, decoder_target_data):
        """ Run training """
        self.sequence_autoencoder.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                      batch_size=self.batch_size,
                                      epochs=2,
                                      validation_split=0.2)


    def predict(self,inp):
        predict_list = self.sequence_autoencoder.predict_on_batch(inp)
        return predict_list


    def make_sentens_vec(self, decoder_model, states_value, start_token):
        sentens_vec = []
        end_len = 20
        word_vec = start_token

        stop_condition = False
        while not stop_condition:
            word_vec, h, c = decoder_model.predict([word_vec] + states_value)
            sentens_vec.append(word_vec)
            states_value = [h, c]
            if (sentens_vec == 0 or len(sentens_vec) == end_len ):
                stop_condition = True

        return sentens_vec, states_value


    def waitController(self,flag,fname):
        if flag == "save":
            print("save"+self.seq2seq_wait_save_dir+fname)
            self.sequence_autoencoder.save_weights(self.seq2seq_wait_save_dir+fname)
        if flag == "load":
            print("load"+self.seq2seq_wait_save_dir+fname)
            self.sequence_autoencoder.load_weights(self.seq2seq_wait_save_dir+fname)


def main():
    seq2seq = Seq2Seq(5,5)
    #seq2seq.make_net()
    encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense = seq2seq.make_net()
    seq2seq.model_complie()

    # load_wait(tr.models[-1],'param_seq2seq_rnp'+"_"+str(value[0])+"_"+str(value[1])+'.hdf5')
    # seq2seq.waitController(self,"load","tmp")

    import random

    """ test train """
    inp_batch = []
    out_batch = []
    out_target_batch = []
    word_len = 20
    sentens_len = 5
    for value in range(seq2seq.batch_size):
        sentens = []
        out_sentens = []
        out_target_sentens = []
        for j in range(sentens_len):
            one_word = []
            num = random.randint(0,10)/10
            for i in range(word_len):
                one_word.append(num+i/10)

            sentens.append(one_word)
            out_sentens.append(one_word)
            out_target_sentens.append(one_word[::-1])

        inp_batch.append(sentens)
        out_batch.append(out_sentens)
        out_target_batch.append(out_target_sentens)

    inp_batch = np.array(inp_batch)
    out_batch = np.array(out_batch)
    out_target_batch = np.array(out_batch)
    print(inp_batch.shape,out_batch.shape,out_target_batch.shape)
    for i in range(10):
        seq2seq.train(inp_batch, out_batch, out_target_batch)

    # seq2seq.waitController("save","tmp")


    """ test  """
    encoder_model, decoder_model = seq2seq.make_decode_net(encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense)

    inp_batch = []
    for value in range(seq2seq.batch_size):
        sentens = []
        for j in range(sentens_len):
            one_word = []
            num = random.randint(0,10)/10
            for i in range(word_len):
                one_word.append(num+i/10)
            sentens.append(one_word)
        inp_batch.append(sentens)

    states_value = encoder_model.predict(inp_batch)
    for seq_index in range(1):
        decord_sentens,states = seq2seq.make_sentens_vec(decoder_model, states_value)


    # decoded_sentence = decode_sequence(sentens)
    # 
    #     # Take one sequence (part of the training test)
    #     # for trying out decoding.
    #     input_seq = encoder_input_data[seq_index: seq_index + 1]
    #     decoded_sentence = decode_sequence(input_seq)
    #     print('-')
    #     print('Input selfentence:', input_texts[seq_index])
    #     print('Decoded sentence:', decoded_sentence)



    """ test  """
    # for value in range(10):
    #     inp_batch = []
    #     word_len = 20
    #     sentens_len = 5
    #     sentens = []
    #     for j in range(sentens_len):
    #         one_word = []
    #         num = random.randint(0,10)/10
    #         for i in range(word_len):
    #             one_word.append(num+i/10)
    #         sentens.append(one_word)
    #     inp_batch.append(sentens)

    #     inp_batch = np.array(inp_batch)

    #     predict = seq2seq.predict(inp_batch)
    #     print("inp :",inp_batch)
    #     print("test:",predict)


    # seq2seq.waitController("load","tmp")
    # inp_batch = [[[ 0.7,  0.8,  0.9,  1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0,
    #                 2.1,  2.2,  2.3,  2.4,  2.5,  2.6],
    #               [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,
    #                 1.5,  1.6,  1.7,  1.8,  1.9,  2.0],
    #               [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,
    #                 1.5,  1.6,  1.7,  1.8,  1.9,  2.0],
    #               [ 0.6,  0.7,  0.8,  0.9,  1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,
    #                 2.,   2.1,  2.2,  2.3,  2.4,  2.5],
    #               [ 0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,
    #                 1.8,  1.9,  2. ,  2.1,  2.2 , 2.3]]]
    # inp_batch = np.array(inp_batch)
    # predict = seq2seq.predict(inp_batch)
    # print("inp :",inp_batch)
    # print("test:",predict)


if __name__ == "__main__":
    main()
