"""
定数用
"""

#import pylab as plt
import lib

class Const():
    def __init__(self):
        """ valiable setting"""
        self.word_feat_len = 128
        self.context_size = 3

        self.batch_size = 64
        self.batch_size = 1

        self.learning_num = 600000
        self.check_point = 200

        self.buckets = [(5, 10), (10, 15), (20, 25), (40, 40)]

        self.seq_len = 3
        """ directory setting"""
        self.project_dir = lib.SetProject.get_path()

        """ word2vec """
        self.word2vec_train_file = self.project_dir+"/aozora_text/files/files_all_rnp.txt"
        # self.word2vec_train_file = self.project_dir+"/aozora_text/files/files_all.txt"
        self.word2vec_wait = self.project_dir+'/lib/model/text8.model'

        """ seq2seq """
        # self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'
        self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'
        self.seq2seq_train_file = self.project_dir+"/aozora_text/files/files_all_rnp.txt"
        # self.seq2seq_train_file = self.project_dir+"/aozora_text/files/tmp.txt"


