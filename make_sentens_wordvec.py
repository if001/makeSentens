'''
lstmを使って文章生成
word2vecを自作して、vector化
今度こそ

word2vecはSkip-gramではなく
CBoWを採用

batch size 1なので過学習してそう

batch_input_shape=(None,\
                   LSTMの中間層に入力するデータの数（※文書データなら単語の数）,\
                   LSTM中間層に投入するデータの次元数（※文書データなら１次元配列なので1)
                  )
'''


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras import optimizers
from keras.models import Model
from keras import backend as K

import numpy as np
import random
import sys

from mecab_test import get_words
#import matplotlib.pyplot as plt
import pylab as plt
from itertools import chain #配列ふらっと化

#import myword2vec
from Myword2vec import myWord2vec 
from MyVec2Word import myVec2Word
from Lstm import Lstm



word_f = 3000



class Trainer():
    def __init__(self):
        self.window_size = 1

    def getWordLists(self):
        print("make wordlists!")
        #self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/files_all.txt")
        #self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/re_re_test.txt")
        #self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/re_re_akumano_monsho.txt")
        #self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/re_re_kohantei_jiken.txt")
        print("")

    def getDict(self):
        print("make dict!")
        print("word_lists len : ",len(self.word_lists))
        self.word_dict = list(set(chain.from_iterable(self.word_lists)))
        print("word dict length : ",len(self.word_dict))

    def dictController(self,flag):
        if flag == "save" :
            print("save dict")
            file = open('./dictionaly/dict.txt', 'w')
            for value in self.word_dict:
                file.write(value+",")
            file.close()

        if flag == "load" :
            print("load dict")
            file = open('./dictionaly/dict.txt', 'r')
            string = file.read()
            self.word_dict = string.split(",")
            file.close()

    def make_sentens(self,lstm,word2vec,vec2word):
        sentens = ""
        wbin = np.zeros(word_f)
        r = random.randint(0,word_f-1)
        wbin[r] = 1

        while(True):
            # 次の特徴量を予測
            predict_list = lstm.predict(wbin)
            predict_index = np.random.choice(range(len(predict_list[0])), 1, p=predict_list[0])[0]
            tmp_bin = np.zeros(word_f)

            tmp_bin[predict_index] = 1

            # 特徴量を単語に変換
            tmp_bin = np.array(tmp_bin)
            tmp_bin = tmp_bin.reshape(1,word_f)
            predict_wordvec = vec2word.vec_to_word(tmp_bin)
            #predict_wordvec = word2vec.vec_to_word(tmp_bin)

            word_index = np.where(predict_wordvec[0] == predict_wordvec[0].max())[0][0]

            # #出力にwindowサイズを設けているので、特徴量から単語を生成する場合は、
            # #出力個数の中間から単語を作成する
            # #1.きりだし
            # window = len(self.word_dict) * word2vec.window_size + 1
            # window_next = len(self.word_dict) * (word2vec.window_size + 1) + 1
            # predict_wordvec = predict_wordvec[0][window:window_next]
            # predict_wordvec = predict_wordvec.reshape(1,len(self.word_dict))
            # # print("predict:",predict_wordvec)
            # # print("predict max:",predict_wordvec[0].max())
            # #2.最大値のインデックスを求める
            # word_index = np.where(predict_wordvec[0] == predict_wordvec[0].max())[0][0]
            # #predict_dict_index = np.random.choice(range(len(predict_wordvec[0])), 1, p=predict_wordvec[0])[0]
            # # print("index:",word_index)

            sentens += self.word_dict[word_index]
            # print(self.word_dict[word_index])
            wbin = tmp_bin

            if((sentens[-1] == "e") or (sentens[-1] == ".") or (sentens[-1] == "。")) :  break
            if len(sentens) > 100: break
        print(sentens)


def main():
    # flag = "learn"
    flag = "make"

    rnp_model = Trainer()
    rnp_model.getWordLists()
    rnp_model.getDict()
    print(rnp_model.word_dict)

    exit(0)
    rnp_model.dictController("save")
    rnp_model.dictController("load")

    lstm = Lstm()
    lstm.input_len_set(len(rnp_model.word_dict),1)
    lstm.make_net()

    word2vec = myWord2Vec()
    word2vec.neuron_num_set(rnp_model.word_dict)
    word2vec.make_net()
    word2vec.hidden_layer()

    vec2word = myVec2Word()
    vec2word.neuron_num_set(rnp_model.word_dict)
    vec2word.make_net()


   if flag == "learn":
        for i in range(10):
            print("learn word2vec:",i,", ",end="")
            # vec2word,word2vec
            for j in range(10):
                # window_selfizeぶん単語をとってくる
                words = word2vec.get_words_W2V(rnp_model.word_lists)
                train_x = []
                train_y = []

                train_x.append(word2vec.make_one_hot(rnp_model.word_dict,words[word2vec.window_size]))
                for j in range(len(words)) :
                    if j != word2vec.window_size :
                        train_y.extend(word2vec.make_one_hot(rnp_model.word_dict,words[j]))

                train_x = np.array(train_x)
                # train_x = train_x.reshape(1,len(train_x))
                train_y = np.array(train_y)
                train_y = train_y.reshape(1,len(train_y))

                word2vec.train_net(train_x,train_y)

                # vec2word train
                vec = word2vec.word_to_vec(train_x[0])
                vec2word.train_net(vec,train_x)
                print("")

            # lstm
            for j in range(10):
                word2vec.waitController("save")
                vec2word.waitController("save")

                # 学習ループ
                print("learn lstm")
                # 文章からランダムに1行とってくる
                input_word_lists = random.sample(rnp_model.word_lists,1)
                for j in range(len(input_word_lists[0])-1-1):
                    x_train = word2vec.word_to_vec(word2vec.make_one_hot(rnp_model.word_dict,input_word_lists[0][j]))
                    y_train = word2vec.word_to_vec(word2vec.make_one_hot(rnp_model.word_dict,input_word_lists[0][j+1]))
                    x_train = x_train.reshape(len(x_train), 1, len(x_train[0]))
                    lstm.train(x_train,y_train)

                rnp_model.make_sentens(lstm,word2vec,vec2word)
                lstm.waitController("save")



    # if flag == "learn":
    #     for i in range(2000):
    #         print("learn word2vec:",i,", ",end="")

    #         # window_sizeぶん単語をとってくる
    #         words = word2vec.get_words_W2V(rnp_model.word_lists)
    #         train_x = []
    #         train_y = []

    #         train_x.append(word2vec.make_one_hot(rnp_model.word_dict,words[word2vec.window_size]))
    #         for j in range(len(words)) :
    #             if j != word2vec.window_size :
    #                 train_y.extend(word2vec.make_one_hot(rnp_model.word_dict,words[j]))

    #         train_x = np.array(train_x)
    #         # train_x = train_x.reshape(1,len(train_x))
    #         train_y = np.array(train_y)
    #         train_y = train_y.reshape(1,len(train_y))

    #         word2vec.train_net(train_x,train_y)

    #         # vec2word train
    #         vec = word2vec.word_to_vec(train_x[0])
    #         vec2word.train_net(vec,train_x)
    #         print("")

    #         # 100回に1回lsmt学習
    #         if i % 100 == 0 :
    #             word2vec.waitController("save")
    #             vec2word.waitController("save")

    #             # 学習ループ
    #             print("learn lstm")
    #             # 文章からランダムに1行とってくる
    #             input_word_lists = random.sample(rnp_model.word_lists,1)
    #             for j in range(len(input_word_lists[0])-1-1):
    #                 x_train = word2vec.word_to_vec(word2vec.make_one_hot(rnp_model.word_dict,input_word_lists[0][j]))
    #                 y_train = word2vec.word_to_vec(word2vec.make_one_hot(rnp_model.word_dict,input_word_lists[0][j+1]))
    #                 x_train = x_train.reshape(len(x_train), 1, len(x_train[0]))
    #                 lstm.train(x_train,y_train)

    #             rnp_model.make_sentens(lstm,word2vec,vec2word)
    #             lstm.waitController("save")



    if flag == "make":
        rnp_model.dictController("load")
        word2vec.waitController("load")
        vec2word.waitController("load")
        lstm.waitController("load")
        for i in range(10):
            rnp_model.make_sentens(lstm,word2vec,vec2word)




if __name__ == "__main__" :
    main()
