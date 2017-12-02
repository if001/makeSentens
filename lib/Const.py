"""
定数用
"""

#import pylab as plt
import lib

class Const():
    def __init__(self):
        """ valiable setting"""
        self.word_feat_len = 128
        # self.word_feat_len = 10

        self.batch_size = 32 # (5,10)のときちょうど良い,64でもわりと頑張る
        self.batch_size = 64
        self.batch_size = 10
        self.learning_num = 600000
        self.learning_num = 100
        self.check_point = 30

        self.buckets = [(5, 10), (10, 15), (20, 25), (40, 40)]
        self.buckets = [(5, 10)]

        """ directory setting"""
        self.project_dir = lib.SetProject.get_path()

        """ word2vec """
        self.word2vec_train_file = self.project_dir+"/aozora_text/files/files_all_rnp2.txt"
        # self.word2vec_train_file = self.project_dir+"/aozora_text/files/files_all.txt"
        self.word2vec_wait = self.project_dir+'/lib/model/text8.model'


        """ seq2seq """
        # self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'
        self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'
        self.seq2seq_train_file = self.project_dir+"/aozora_text/files/files_all_rnp.txt"
        self.seq2seq_train_file = self.project_dir+"/aozora_text/files/files_all_rnp2.txt"


    # def glaph_plot(self,data):
    #     t = range(len(data))
    #     plt.plot(t,data)
    #     plt.show()

