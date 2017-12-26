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

        self.word_feat_len = 5
        self.latent_dim = 5
        self.latent_dim2 = 10

        tb_cb = TensorBoard(log_dir="~/tflog/", histogram_freq=1)
        self.cbks = [tb_cb]


    def build_encoder(self, model=None):
        input_dim = self.latent_dim
        output_dim = self.latent_dim

        encoder_inputs = Input(shape=(None, input_dim))
        if model ==  None :
            encoder_dense_outputs = Dense(input_dim, activation='sigmoid')(encoder_inputs)
            _, state_h, state_c = LSTM(self.latent_dim, return_state=True, dropout=0.2, recurrent_dropout=0.2)(encoder_dense_outputs)
        else :
            _, ed, el = model.layers
            encoder_dense_outputs = ed(encoder_inputs)
            _, state_h, state_c = el(encoder_dense_outputs)
        return Model(encoder_inputs, [state_h, state_c])


    def build_decoder(self, model=None):
        input_dim = self.latent_dim
        output_dim = self.latent_dim

        encoder_h = Input(shape=(self.latent_dim,))
        encoder_c = Input(shape=(self.latent_dim,))
        encoder_states =  [encoder_h, encoder_c]

        decoder_inputs = Input(shape=(None, input_dim))
        decoder_dense_outputs = Dense(input_dim, activation='sigmoid')(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2)
        decoder_outputs, decoder_h, decoder_c = decoder_lstm(decoder_dense_outputs, initial_state=encoder_states)
        decoder_states = [decoder_h, decoder_c]
        decoder_outputs = Dense(output_dim, activation='relu')(decoder_outputs)
        decoder_outputs = Dense(output_dim, activation='linear')(decoder_outputs)

        return Model([decoder_inputs] + encoder_states, [decoder_outputs] + decoder_states)


    def build_context_model(self):
        input_dim = self.latent_dim
        output_dim = self.latent_dim
        inputs = Input(shape=(None, input_dim))
        outputs = LSTM(self.latent_dim)(inputs)
        l = Model(inputs, outputs)
        return Model(inputs, outputs)


    def build_autoencoder(self, encoder, decoder):
        input_dim = self.latent_dim
        output_dim = self.latent_dim

        encoder_inputs = Input(shape=(None, input_dim))
        ei, ed, el = encoder.layers
        dense_outputs = ed(encoder_inputs)
        encoder_output, state_h, state_c = el(dense_outputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, input_dim))
        di, dd1, di2, di3, dl, dd2,dd3 = decoder.layers
        decoder_dense_outputs = dd1(decoder_inputs)
        decoder_lstm_outputs, _ , _ =  dl(decoder_dense_outputs, initial_state=encoder_states)
        decoder_dense2_outputs = dd2(decoder_lstm_outputs)
        outputs = dd3(decoder_dense2_outputs)

        return Model([encoder_inputs, decoder_inputs], outputs)


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


    def train_def_autoencoder(self, model, encoder_input_data, decoder_input_data, decoder_target_data):
        """ Run training """
        loss = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                             batch_size=self.batch_size,
                                             epochs=1,
                                             validation_split=0.2)
        return loss


    def train_autoencoder(self, model, encoder_input_data, decoder_input_data, decoder_target_data):
        """ Run training """
        loss = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                             batch_size=self.batch_size,
                                             epochs=1,
                                             validation_split=0.2)
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
