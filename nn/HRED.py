import numpy as np
import matplotlib.pylab as plt
from keras.models import Model

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
# from keras.layers.wrappers import TD

from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector, concatenate, Dropout, Bidirectional
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.layers          import merge, multiply
from keras.optimizers import Adam,SGD,RMSprop

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization as BN

from keras import regularizers


import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# mylib
import lib


class HRED(lib.Const.Const):
    """This is a test program."""

    def __init__(self):
        super().__init__()
        self.input_word_num = 1
        self.latent_dim = 256
        self.latent_dim2 = 512
        tb_cb = TensorBoard(log_dir="~/tflog/", histogram_freq=1)
        self.cbks = [tb_cb]

    def build_context_net(self):
        input_dim = self.latent_dim
        output_dim = self.latent_dim
        inputs = Input(shape=(None, input_dim))
        outputs = LSTM(self.latent_dim, return_state=True)(inputs)
        return Model(inputs, outputs)


    def build_net(self, context_model_h, context_model_c):
        """ make net by reference to Keras official doc """
        # # テスト用ぱらめた
        # input_dim = 5
        # output_dim = 5

        input_dim = self.word_feat_len
        output_dim = self.word_feat_len

        encoder_inputs = Input(shape=(None, input_dim))
        encoder_dense_outputs = Dense(self.latent_dim, activation='sigmoid')(encoder_inputs)
        _, state_h, state_c = LSTM(self.latent_dim2, return_state=True, dropout=0.2, recurrent_dropout=0.2)(encoder_dense_outputs)

        h_input, h_lstm = context_model_h.layers
        c_input, c_lstm = context_model_c.layers

        state_h = h_lstm(state_h)
        state_c = c_lstm(state_c)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, input_dim))
        decoder_dense_outputs = Dense(input_dim, activation='sigmoid')(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(decoder_dense_outputs, initial_state=encoder_states)
        decoder_outputs = Dense(output_dim, activation='relu')(decoder_outputs)
        decoder_outputs = Dense(output_dim, activation='linear')(decoder_outputs)

        self.sequence_autoencoder = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    def build_decode_net(self):
        """ for decoding net """

        input_dim = self.word_feat_len
        output_dim = self.word_feat_len

        ei, di, ed, dd, eb, db, el, dl, dd2, dd3 = self.sequence_autoencoder.layers

        encoder_inputs = Input(shape=(None, input_dim))
        encoder_dense_output = Dense(input_dim, activation='sigmoid', weights=ed.get_weights())(encoder_inputs)
        encoder_bi_output = eb(encoder_dense_output)
        _, state_h, state_c = LSTM(self.latent_dim, return_state=True, weights=el.get_weights())(encoder_bi_output)
        encoder_states = [state_h, state_c]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_inputs = Input(shape=(None, input_dim))
        decoder_dense_outputs = Dense(input_dim, activation='sigmoid', weights=dd.get_weights())(decoder_inputs)
        decoder_lstm_outputs = db(decoder_dense_outputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, weights=dl.get_weights())
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_lstm_outputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = Dense(output_dim, activation='relu', weights=dd2.get_weights())(decoder_outputs)
        decoder_outputs = Dense(output_dim, activation='linear', weights=dd3.get_weights())(decoder_outputs)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)


    def model_complie(self):
        """ complie """
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        loss = 'mean_squared_error'
        # loss = 'kullback_leibler_divergence'
        self.sequence_autoencoder.compile(optimizer=optimizer,
                                          loss=loss,
                                          metrics=['accuracy'])

        self.sequence_autoencoder.summary()


    def train(self, encoder_input_data, decoder_input_data, decoder_target_data):
        """ Run training """
        loss = self.sequence_autoencoder.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                             batch_size=self.batch_size,
                                             epochs=1,
                                             validation_split=0.2)
        return loss



    def make_sentens_vec(self, decoder_model, states_value, start_token):
        sentens_vec = []
        end_len = 20
        word_vec = start_token

        stop_condition = False
        while not stop_condition:
            word_vec, h, c = decoder_model.predict([word_vec] + states_value)
            sentens_vec.append(word_vec)
            states_value = [h, c]
            if (sentens_vec == 0 or len(sentens_vec) == 5 ):
                stop_condition = True

        return sentens_vec


    def waitController(self,flag, fname, model):
        if flag == "save":
            print("save"+self.seq2seq_wait_save_dir+fname)
            model.save(self.seq2seq_wait_save_dir+fname)
        if flag == "load":
            print("load"+self.seq2seq_wait_save_dir+fname)
            from keras.models import load_model
            return load_model(self.seq2seq_wait_save_dir+fname)


def main():
    pass

if __name__ == "__main__":
    main()
