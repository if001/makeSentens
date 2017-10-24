#単語をベクトル化

import gensim
from gensim.models import word2vec
#from gensim import models as mod

import pylab as plt

import numpy as np

# mylib
from Const import Const


class MyWord2Vec(Const):
    def __init__(self):
        super().__init__()

    def train(self,fname,saveflag="save"):
        sentences = gensim.models.word2vec.Text8Corpus(fname)
        #model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=5)
        self.model = gensim.models.word2vec.Word2Vec(sentences, size=self.word_feat_len, window=5, workers=4, min_count=1)
        if saveflag == "save":
            self.model.save(self.word2vec_wait)

    def load_model(self):
        # 読み込み
        self.model = word2vec.Word2Vec.load(self.word2vec_wait)


    def get_vector(self,st):
        return self.model.wv[st]

    def get_similar(self,st,top):
        # 類似ワード出力
        results = self.model.most_similar(positive=st, topn=top)
        for result in results:
            print(result[0], '\t', result[1])

    def get_word(self,vec):
        return self.model.most_similar( [ vec ], [], 1)[0][0]

def plot(vec):
    t = range(len(vec))
    plt.plot(t,vec)
    plt.show()

def main():
    net = MyWord2Vec()
    const = Const()

    net.train(const.dict_train_file,"not save")

    #net.load_model()
    # vec = net.get_vector("博士")
    # vec = net.get_vector("明智")
    vec = net.get_vector("怪盗")
    # print(vec)
    plot(vec)

    # vec = np.array(vec,dtype='f')
    # word = net.get_word(vec)
    # print("word",word)

    #net.get_word()

if __name__ == "__main__":
    main()
