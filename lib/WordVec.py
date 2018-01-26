#単語をベクトル化

import gensim
# from gensim.models import word2vec
#from gensim import models as mod
#import pylab as plt

import numpy as np

# mylib
import lib

word_feat_len = lib.Const.Const().word_feat_len
word2vec_wait = lib.Const.Const().word2vec_wait

class MyWord2Vec():

    @staticmethod
    def train(fname, saveflag="save"):
        print("train word2vec")
        sentences = gensim.models.word2vec.Text8Corpus(fname)
        #model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=5)
        model = gensim.models.word2vec.Word2Vec(sentences, size=word_feat_len, window=5, workers=4, min_count=1, hs=1)
        if saveflag == "save":
            print("save " + word2vec_wait)
            model.save(word2vec_wait)

    @staticmethod
    def load_model():
        # 読み込み
        print("load " + word2vec_wait)
        model = gensim.models.word2vec.Word2Vec.load(word2vec_wait)
        return model

    @staticmethod
    def vec_to_word(model, vec):
        return model.most_similar( [ vec ], [], 1)[0][0]


    @staticmethod
    def vec_to_some_word(model, vec, num):
        return model.most_similar( [ vec ], [], num)


    # def similar_words(self,st,top):
    #     # 類似ワード出力
    #     results = self.model.most_similar(positive=st, topn=top)
    #     for result in results:
    #         print(result[0], '\t', result[1])

    @staticmethod
    def str_to_vector(model, st):
        return model.wv[st]

    # def get_similar_vector(self,st):
    #     __st = self.model.most_similar(positive=st, topn=1)[0][0]
    #     return self.model.wv[__st]


def plot(vec):
    t = range(len(vec))
    plt.plot(t,vec)
    plt.show()


def main():
    #net.train(const.dict_train_file,"not save")
    net.train("/aozora_text3/files/files_all_rnp.txt","not save")

    #net.load_model()
    # vec = net.get_vector("博士")
    # vec = net.get_vector("明智")
    vec = MyWord2Vec().get_vector("怪盗")
    print(vec)
    # plot(vec)

    # vec = np.array(vec,dtype='f')
    # word = net.get_word(vec)
    # print("word",word)

    #net.get_word()

if __name__ == "__main__":
    main()
