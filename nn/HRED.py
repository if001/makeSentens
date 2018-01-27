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
from keras.layers.core     import Flatten, Reshape
from keras.layers          import merge, multiply
from keras.optimizers import Adam,SGD,RMSprop

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization as BN

from keras import regularizers
from keras import backend as K


import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# mylib
import lib


class HRED(lib.Const.Const):
    """This is a test program."""

    def __init__(self):
        super().__init__()
        self.input_word_num = 1
        self.input_dim = self.word_feat_len
        self.output_dim = self.word_feat_len
        self.latent_dim = 512
        self.latent_dim2 = 256

        tb_cb = TensorBoard(log_dir="~/tflog/", histogram_freq=1)
        self.cbks = [tb_cb]


    def build_encoder(self, model=None):
        K.set_learning_phase(1) # set learning phase

        encoder_inputs = Input(shape=(None, self.input_dim))
        encoder_dense_outputs = Dense(self.input_dim, activation='sigmoid')(encoder_inputs)
        encoder_bi_lstm = LSTM(self.latent_dim, return_sequences=True , dropout=0.6, recurrent_dropout=0.6)
        encoder_bi_outputs = Bi(encoder_bi_lstm)(encoder_dense_outputs)
        _, state_h, state_c = LSTM(self.latent_dim, return_state=True, dropout=0.2, recurrent_dropout=0.2)(encoder_bi_outputs)

        return Model(encoder_inputs, [state_h, state_c])


    def build_decoder(self, model=None):
        K.set_learning_phase(1) # set learning phase

        encoder_h = Input(shape=(self.latent_dim,))
        encoder_c = Input(shape=(self.latent_dim,))
        encoder_states = [encoder_h, encoder_c]

        decoder_inputs = Input(shape=(None, self.input_dim))
        decoder_dense_outputs = Dense(self.input_dim, activation='sigmoid')(decoder_inputs)
        decoder_bi_lstm = LSTM(self.latent_dim, return_sequences=True, dropout=0.6, recurrent_dropout=0.6)
        decoder_bi_outputs = Bi(decoder_bi_lstm)(decoder_dense_outputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        # decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, output_h, output_c = decoder_lstm(decoder_bi_outputs, initial_state=encoder_states)

        # パターン1
        decoder_outputs = Dense(self.latent_dim2, activation='tanh')(decoder_outputs)
        decoder_outputs = Dropout(0.2)(decoder_outputs)
        decoder_outputs = Dense(self.output_dim, activation='linear')(decoder_outputs)

        # パターン2
        # decoder_outputs = Dense(self.output_dim, activation='tanh')(decoder_outputs)
        # decoder_outputs = Dropout(0.2)(decoder_outputs)
        # decoder_outputs = Dense(self.output_dim, activation='linear')(decoder_outputs)

        return Model([decoder_inputs, encoder_h, encoder_c], [decoder_outputs, output_h, output_c])


    def build_context_model(self):
        K.set_learning_phase(1) #set learning phase
        inputs = Input(shape=(None, self.latent_dim))
        state_h_input = Input(shape=(self.latent_dim,))
        state_c_input = Input(shape=(self.latent_dim,))
        state_value = [state_h_input, state_c_input]
        outputs, state_h, state_c = LSTM(self.latent_dim, return_state=True)(inputs, initial_state=state_value)
        return Model([inputs, state_h_input, state_c_input], [outputs, state_h, state_c])


    def build_autoencoder(self, encoder, decoder, context_h, context_c):
        # encoder
        encoder_inputs = Input(shape=(None, self.input_dim))
        ei, ed, eb, el = encoder.layers
        dense_outputs = ed(encoder_inputs)
        bi_outputs = eb(dense_outputs)
        encoder_output, state_h, state_c = el(bi_outputs)

        # context
        cih, _, _, clh = context_h.layers
        meta_hh = Input(shape=(self.latent_dim,))
        meta_hc = Input(shape=(self.latent_dim,))
        meta_h_state = [meta_hh, meta_hc]
        state_h = Reshape((1 , self.latent_dim))(state_h)
        state_h_output, _, _ = clh(state_h, initial_state=meta_h_state)

        cic, _, _, clc = context_c.layers
        meta_ch = Input(shape=(self.latent_dim,))
        meta_cc = Input(shape=(self.latent_dim,))
        meta_c_state = [meta_ch, meta_cc]
        state_c = Reshape((1 , self.latent_dim))(state_c)
        state_c_output, _, _ = clc(state_c, initial_state=meta_c_state)
        encoder_states = [state_h_output, state_c_output]

        # decoder
        decoder_inputs = Input(shape=(None, self.input_dim))
        di, dd1, db, di2, di3, dl, dd2, do, dd3 = decoder.layers
        decoder_dense_outputs = dd1(decoder_inputs)
        decoder_bi_outputs = db(decoder_dense_outputs)
        decoder_lstm_outputs, _ , _ =  dl(decoder_bi_outputs, initial_state=encoder_states)
        decoder_outputs = dd2(decoder_lstm_outputs)
        outputs = do(decoder_outputs)
        outputs = dd3(outputs)

        return Model([encoder_inputs, decoder_inputs, meta_hh, meta_hc, meta_ch, meta_cc], outputs)


    def model_compile(self, model):
        """ complie """
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        loss = 'mean_squared_error'
        # loss = 'kullback_leibler_divergence'
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])
        model.summary()
        return model


    def train_autoencoder(self, model, encoder_input_data, decoder_input_data, decoder_target_data, meta_hh, meta_hc, meta_ch, meta_cc):
        """ Run training """
        # loss = model.train_on_batch([encoder_input_data, decoder_input_data, meta_hh, meta_hc, meta_ch, meta_cc], decoder_target_data)
        loss = model.fit([encoder_input_data, decoder_input_data, meta_hh, meta_hc, meta_ch, meta_cc], decoder_target_data,
                         batch_size=self.batch_size,
                         epochs=1)
        #                  # validation_split=0.2)
        return loss

    def test_autoencoder(self,  model, encoder_input_data, decoder_input_data, decoder_target_data, meta_hh, meta_hc, meta_ch, meta_cc):
        loss = model.test_on_batch([encoder_input_data, decoder_input_data, meta_hh, meta_hc, meta_ch, meta_cc], decoder_target_data)
        return loss


    def train_context(self, model, train_data, teach_data):
        """ Run training """
        loss = model.fit(train_data, teach_data,
                         batch_size=self.batch_size,
                         epochs=1,
                         validation_split=0.2)

        return loss



    # def make_sentens_vec(self, decoder_model, states_value, start_token):
    #     sentens_vec = []
    #     end_len = 20
    #     word_vec = start_token

    #     stop_condition = False
    #     while not stop_condition:
    #         word_vec, h, c = decoder_model.predict([word_vec] + states_value)
    #         sentens_vec.append(word_vec)
    #         states_value = [h, c]
    #         if (sentens_vec == 0 or len(sentens_vec) == 5 ):
    #             stop_condition = True

    #     return sentens_vec


    def save_models(self, fname, model):
        print("save"+self.seq2seq_wait_save_dir+fname)
        model.save(self.seq2seq_wait_save_dir+fname)

    def load_models(self, fname):
        print("load"+self.seq2seq_wait_save_dir+fname)
        from keras.models import load_model
        return load_model(self.seq2seq_wait_save_dir+fname)



def main():
    pass

if __name__ == "__main__":
    main()
