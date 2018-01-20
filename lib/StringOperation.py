import lib
import random as rand
import numpy as np


class StringOperation(lib.Const.Const):
    def __init__(self,flag="load"):
        super().__init__()
        self.word2vec = lib.WordVec.MyWord2Vec()
        self.word2vec.load_model()

        # if flag == "learn":
        #     self.word2vec.train(self.dict_train_file)
        # elif flag == "load":
        #     self.word2vec.load_model()
        # else:
        #     print("not set word2vec model")
        #     exit(0)


    def sentens_array_to_str(self,sentens_array):
        __sentens = ""
        for value in sentens_array:
            __sentens += value
            if (value == "。"): break
        return __sentens


    def sentens_array_to_vec(self,sentens_arr):
        __sentens_vec = []
        for value in sentens_arr:
            __vec = self.word2vec.get_vector2(value)
            __sentens_vec.append(__vec)
        return __sentens_vec


    def sentens_vec_to_sentens_arr(self,sentens_vec):
        """ if word does not exist in bocablaly , Substitute with an alternative word """
        __arr = []
        for value in sentens_vec:
            __word = self.word2vec.get_word(value)
            __arr.append(__word)
        return __arr


    def sentens_vec_to_sentens_arr_prob(self,sentens_vec):
        __arr = []
        for value in sentens_vec:
            __prob_word = self.word2vec.get_some_word(value, 5)
            __word_list = []
            __prob = []
            for p in __prob_word:
                __word_list.append(p[0])
                __prob.append(p[1])
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
