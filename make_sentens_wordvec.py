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

#import myword2vec
#from myword2vec import 

#import matplotlib.pyplot as plt
import pylab as plt

# cython
import pyximport; pyximport.install()
import cythonFunc

word_f = 3000

from itertools import chain #配列ふらっと化

class NN():
    def __init__(self):
        self.window_size = 1

    def input_len_set(self,worddict_len,window_size):
        self.input_len = word_f * window_size
        self.output_len = 3000
        self.hidden_neurons = 5000
        # self.input_len = worddict_len * window_size
        # self.output_len = worddict_len



    def make_net(self):
        print("make network")
        self.model = Sequential()

        self.model.add(LSTM(self.hidden_neurons, input_shape=(1, self.input_len)))

        self.model.add(Dense(self.output_len))
        self.model.add(Activation("softmax"))

        loss = 'categorical_crossentropy'
        loss = "mean_squared_error"
        loss = "binary_crossentropy"
        optimizer = "adam"
        optimizer = RMSprop(lr=0.01)

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()

    def train(self,X_train,Y_train):
        self.history = self.model.train_on_batch(X_train,Y_train, class_weight=None, sample_weight=None)
        print("lstm :",self.history)

    def predict(self,inp):
        inp = np.array(inp)
        inp = inp.reshape(1,1,self.input_len)
        predict_list = self.model.predict_on_batch(inp)

        return predict_list


    def netScore(self,X_train,Y_train):
        self.score = self.model.evaluate(X_train, Y_train, verbose=0)
        print("lstm : ",self.score)
        # print('test loss:', self.score[0])
        # print('test acc:', self.score[1])

    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.model.save_weights('./wait/param_make_sentens_wordvec_lstm.hdf5')
            if flag == "load":
                print("load")
                self.model.load_weights('./wait/param_make_sentens_wordvec_lstm.hdf5')
        except :
            print("no such file")
            sys.exit(0)



class myVec2Word():
    def __init__(self):pass

    def neuron_num_set(self,word_dict):
        self.input_len = word_f
        self.hidden_len = 3000
        self.output_len = len(word_dict)

    def make_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_len , input_shape=(self.input_len,)))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_len))
        self.model.add(Activation('softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loss = 'categorical_crossentropy'
        loss = "binary_crossentropy"

        self.model.compile(loss=loss, optimizer=sgd)
        self.model.summary()


    def train_net(self,train_x,train_y):
        # モデルの訓練
        self.history = self.model.train_on_batch(train_x,train_y,
                                                 class_weight=None, sample_weight=None)
        print("vec to word:",self.history,", ",end="")

    def vec_to_word(self,vec):
        # inp = np.array(vec)
        # inp = inp.reshape(1,1,self.input_len)
        word_vec = self.model.predict_on_batch(vec)

        # hidden_layer_output = K.function([self.model.layers[2].input],
        #                                   [self.model.layers[3].output])

        # word_vec = hidden_layer_output([vec])[0]
        return word_vec

    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.model.save_weights('./wait/param_make_sentens_wordvec_vec2word.hdf5')
            if flag == "load":
                print("load")
                self.model.load_weights('./wait/param_make_sentens_wordvec_vec2word.hdf5')
        except :
            print("no such file")
            sys.exit(0)


class myWord2Vec():
    def __init__(self):
        # 0はだめ
        self.window_size = 3

    def neuron_num_set(self,word_dict):
        self.input_len = len(word_dict)
        self.hidden_len = word_f
        self.output_len = len(word_dict)*(self.window_size*2)

    def get_words_W2V(self,word_lists):
        sentens_num = 3
        # ランダムに文章選択
        r1 = random.randint(0,len(word_lists)-1-sentens_num)
        r2 = r1 + 1
        r3 = r1 + 2
        tmp_word_lists = word_lists[r1] + word_lists[r2] + word_lists[r3]

        r4 = random.randint(0,len(tmp_word_lists)-(self.window_size*2+1)-1)

        return tmp_word_lists[r4:r4+(self.window_size)*2+1 ]

        # # 文章内から1語選択
        # r2 = random.randint(0,len(word_lists[r])-(self.window_size*2+1)-1)
        # # 選択した文章からwindowsize文単語を取ってくる
        # return word_lists[r][r2:r2+(self.window_size)*2+1 ]


    def make_one_hot(self,word_dict,word):
        # print("make one hot vector")
        self.wordvec_one_hot = [0 for i in range(len(word_dict))]
        self.wordvec_one_hot[word_dict.index(word)] = 1
        return self.wordvec_one_hot


    def make_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_len , input_shape=(self.input_len,)))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_len))
        self.model.add(Activation('sigmoid'))
        #self.model.add(Activation('softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loss = 'categorical_crossentropy'
        loss = "binary_crossentropy"

        self.model.compile(loss=loss, optimizer=sgd)
        self.model.summary()


    def hidden_layer(self):
        # 中間層を出力するためにモデルを形成
        # with a Sequential model
        self.get_hidden_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[1].output])

        # layer_name = 'my_layer'
        # self.intermediate_layer_model = Model(inputs = self.model.input,outputs = self.model.get_layer(layer_name).output)


    def word_to_vec(self,input_word):
        input_word = np.array(input_word)
        input_word = input_word.reshape(1,len(input_word))
        layer_output = self.get_hidden_layer_output([input_word])[0]
        return layer_output


    def train_net(self,train_x,train_y):
        # モデルの訓練
        self.history = self.model.train_on_batch(train_x,train_y,
                                                 class_weight=None, sample_weight=None)
        print("word to vec",self.history,", ",end="")



    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.model.save_weights('./wait/param_make_sentens_wordvec.hdf5')
            if flag == "load":
                print("load")
                self.model.load_weights('./wait/param_make_sentens_wordvec.hdf5')
        except :
            print("no such file")
            sys.exit(0)



class Trainer():
    def __init__(self):
        self.window_size = 1

    def getWordLists(self):
        print("make wordlists!")
        self.word_lists = cythonFunc.readfile_for_word2vec("./aozora_text/files_all.txt")
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
    rnp_model.dictController("save")
    rnp_model.dictController("load")

    lstm = NN()
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
        for i in range(2000):
            print("learn word2vec:",i,", ",end="")

            # window_sizeぶん単語をとってくる
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

            # 100回に1回lsmt学習
            if i % 100 == 0 :
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



    if flag == "make":
        rnp_model.dictController("load")
        word2vec.waitController("load")
        vec2word.waitController("load")
        lstm.waitController("load")
        for i in range(10):
            rnp_model.make_sentens(lstm,word2vec,vec2word)




if __name__ == "__main__" :
    main()
