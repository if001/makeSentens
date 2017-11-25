"""
定数用
"""

#import pylab as plt
import lib

class Const():
    def __init__(self):
        """ valiable setting"""
        self.check_point = 200
        self.word_feat_len = 128
        self.batch_size = 32 # (5,10)のときちょうど良い,64でもわりと頑張る
        self.batch_size = 50
        self.learning_num = 10
        self.learning_num = 20000
        # self.seq_num = 40
        # self.buckets = [(5, 10),(10,15)]
        # self.buckets = [(40,50)]
        self.buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

        """ directory setting"""
        self.project_dir = lib.SetProject.get_path()
        # self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/param_make_sentens_seq2seq_rnp.hdf5'
        # self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'
        self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'

        self.dict_dir = self.project_dir+'/dictionaly/dict.txt'


        # self.dict_train_file = self.project_dir+'/aozora_text3/files/files_all_umn.txt'
        self.dict_train_file = self.project_dir+"/aozora_text3/files/files_all_rnp.txt"

        # self.dict_train_file = self.project_dir+"/aozora_text3/files/files_all_all.txt"

        self.dict_load_file = self.project_dir+"/aozora_text3/files/files_all_rnp.txt"
        # self.dict_load_file = self.project_dir+"/aozora_text3/files/files_all_all.txt"

        self.word2vec_wait = self.project_dir+'/lib/model/text8_rnp.model'
        #self.word2vec_wait = self.project_dir+'/nn/model/text8_umn.model'

    # def glaph_plot(self,data):
    #     t = range(len(data))
    #     plt.plot(t,data)
    #     plt.show()

