#単語をベクトル化

import gensim
from gensim.models import word2vec
#from gensim import models as mod


import numpy as np

class myword2vec():
    def __init__(self):
        self.fname="./aozora_text/files_all.txt"

    def train(self):
        sentences = gensim.models.word2vec.Text8Corpus(self.fname)
        #model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=5)

        self.model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=1)
        self.model.save("./model/text8.model")

    def load_model(self):
        # 読み込み
        self.load_model = word2vec.Word2Vec.load("./model/text8.model")


    def get_vector(self,st):
        # ベクトル表示
        # print(self.load_model[st])
        # print(self.load_model[st].shape)
        return self.model.wv[st]


    def get_similar(self,st,top):
        # 類似ワード出力
        results = self.load_model.most_similar(positive=st, topn=top)
        for result in results:
            print(result[0], '\t', result[1])

    def get_word(self):pass


def main():
    net = myword2vec()
    net.load_model()
    net.train()
    vec = net.get_vector("彼")
    vec = np.array(vec, dtype='f')

    print("vec",vec)
    word = net.model.most_similar( [ vec ], [], 1)
    print("word",word)
    
    #net.get_word()

if __name__ == "__main__":
    main()
