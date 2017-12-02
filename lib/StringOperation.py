import lib

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
            # try:
            #     __vec = self.word2vec.get_vector(value)
            # except ValueError:
            #     print("value error",value)
            #     __vec = self.word2vec.get_similar_vector(value)
            # except KeyError:
            #     print("key error",value)
            #     __vec = self.word2vec.get_similar_vector(value)
            __sentens_vec.append(__vec)
        return __sentens_vec


    def sentens_vec_to_sentens_arr(self,sentens_vec):
        """ if word does not exist in bocablaly , Substitute with an alternative word """
        __arr = []
        for value in sentens_vec:
            __word = self.word2vec.get_word(value)
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


    def reshape_sentens(self,sentens):
        if ('「' in sentens): sentens.remove('「')
        if ('」' in sentens): sentens.remove('」')
        while '' in sentens: sentens.remove('')
        return sentens
