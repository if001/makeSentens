"""
定数用
"""
import pylab as plt

class Const():
    def __init__(self):

        self.word_feat_len = 500
        # self.batch_size = 3000
        self.batch_size = 2500
        self.learning_num = 8
        # self.seq_num = 10
        self.seq_num = 40
        # self.seq2seq_wait_save_dir = './wait/param_make_sentens_seq2seq_rnp.hdf5'
        # self.seq2seq_wait_save_dir = './wait/param_make_sentens_seq2seq_ymn.hdf5'
        self.seq2seq_wait_save_dir = './wait/param_make_sentens_seq2seq_rnp_ymn.hdf5'

        self.dict_dir = './dictionaly/dict.txt'

        # self.dict_train_file = './aozora_text/files_all.txt'
        self.dict_train_file = './aozora_text2/files_all.txt'

        #self.word2vec_wait = './model/text8_rnp.model'
        # self.word2vec_wait = './model/text8_ymn.model'
        self.word2vec_wait = './model/text8_rnp_ymn.model'

    def glaph_plot(self,data):
        t = range(len(data))
        plt.plot(t,data)
        plt.show()
