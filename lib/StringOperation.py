import lib
import random as rand
import numpy as np

from lib import WordVec as wv


class StringOperation():
    def __init__(self, load_flag="load"):
        const = lib.Const.Const()
        self.word_feat_len = const.word_feat_len

        self.myW2V = wv.MyWord2Vec()
        fname = const.word2vec_train_file
        if load_flag == "train":
            self.myW2V.train(fname)
        self.word2vec_model = self.myW2V.load_model()


    def sentens_array_to_str(self, sentens_array):
        __sentens = ""
        for value in sentens_array:
            __sentens += value
            if (value == "。"): break
        return __sentens


    def sentens_array_to_vec(self,sentens_arr):
        __sentens_vec = []
        for value in sentens_arr:
            __vec = self.myW2V.str_to_vector(self.word2vec_model, value)
            __sentens_vec.append(__vec)
        return __sentens_vec


    def sentens_vec_to_sentens_arr(self,sentens_vec):
        __arr = []
        for value in sentens_vec:
            __word = self.myW2V.vec_to_word(self.word2vec_model, value)
            __arr.append(__word)
        return __arr


    def sentens_vec_to_sentens_arr_prob(self,sentens_vec):
        __arr = []
        for value in sentens_vec:
            __prob_word = self.myW2V.vec_to_some_word(self.word2vec_model, value, 5)
            print(__prob_word)
            __word_list = []
            __prob = []
            for p in __prob_word:
                __word_list.append(p[0])
                __prob.append(p[1])
            __prob = np.array(__prob)/sum(__prob)
            __word = np.random.choice(__word_list, p=__prob)
            __arr.append(__word)
        return __arr


    def EOF_padding(self,sentens,seq_num):
        if seq_num > len(sentens):
            __diff_len = seq_num - len(sentens)
            for i in range(__diff_len):
                sentens.append("。")
        return sentens


    def zero_padding(self,sentens_vec,seq_num):
        if seq_num > len(sentens_vec):
            __diff_len = seq_num - len(sentens_vec)
            for i in range(__diff_len):
                sentens_vec.append([0 for i in range(self.word_feat_len)])
        return sentens_vec


    def reshape_sentens(self, sentens):
        __sentens = sentens[::]
        if ('「' in __sentens): __sentens.remove('「')
        if ('」' in __sentens): __sentens.remove('」')
        while '' in __sentens: __sentens.remove('')
        return __sentens


    def add_BOS(self, sentens):
        __sentens = sentens
        if ('BOS' not in __sentens): __sentens.insert(0, 'BOS')
        return __sentens


    def rm_BOS(self, sentens):
        __sentens = sentens
        if ('BOS' in __sentens): __sentens.remove('BOS')
        return __sentens
