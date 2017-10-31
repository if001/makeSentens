"""
定数用
"""

#import pylab as plt
import lib


class Const():
    def __init__(self):
        """ valiable setting"""
        self.word_feat_len = 50
        self.batch_size = 20000
        # self.batch_size = 25
        self.learning_num = 10
        # self.seq_num = 40
        self.buckets = [(5, 10)]
        #self.buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

        """ directory setting"""
        self.project_dir = lib.set_project.get_path()
        # self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/param_make_sentens_seq2seq_rnp.hdf5'
        # self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'
        self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'

        self.dict_dir = self.project_dir+'/dictionaly/dict.txt'

        self.dict_train_file = self.project_dir+'/aozora_text/files_all.txt'
        self.dict_train_file = self.project_dir+'/aozora_text/files_all_conv.txt'
        # self.dict_train_file = self.project_dir+'/aozora_text2/files_all.txt'

        self.dict_load_file = self.project_dir+'/aozora_text/files_all.txt'

        #self.word2vec_wait = self.project_dir+'/nn/model/text8_rnp.model'
        # self.word2vec_wait = self.project_dir+'/nn/model/text8_ymn.model'
        self.word2vec_wait = self.project_dir+'/nn/model/text8_rnp_ymn.model'


    # def glaph_plot(self,data):
    #     t = range(len(data))
    #     plt.plot(t,data)
    #     plt.show()

