from seq2seq.models import SimpleSeq2Seq
import numpy as np
import matplotlib.pylab as plt

# mylib
from Const import Const

class Seq2Seq(Const):
    def __init__(self):
        super().__init__()
        self.word_dim = 3000
        self.input_word_num = 1
        self.input_length = 100
        # self.hidden_dim = 6000
        self.hidden_dim = 100

    def make_net(self):
        # テスト用
        # self.model = SimpleSeq2Seq(input_dim=self.word_dim,
        #                            input_length=1,
        #                            hidden_dim=self.hidden_dim,
        #                            output_length=1,
        #                            output_dim=self.word_dim,
        #                            depth=self.seq_num)

        self.model = SimpleSeq2Seq(input_dim=self.word_dim,
                                   input_length=self.input_length,
                                   hidden_dim=self.hidden_dim,
                                   output_length=self.input_length,
                                   output_dim=self.word_dim,
                                   depth=1)


        loss = 'mse'
        optimizer = 'rmsprop'
        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.summary()


    def train(self,X_train,Y_train):
        self.model.fit(X_train, Y_train,
                       nb_epoch=1,
                       batch_size=self.batch_size,
                       validation_split=0.3,
                       verbose=1)

    def predict(self,inp):
        #inp = np.array(inp)
        #inp = inp.reshape(1,1,self.input_len)
        #word_vec = self.model.predict(vec, batch_size=self.batch_size, verbose=0)
        #predict_list = self.model.predict(inp,batch_size=self.batch_size, verbose=0)
        predict_list = self.model.predict_on_batch(inp)
        return predict_list

    def waitController(self,flag):
        try:
            if flag == "save":
                print("save")
                self.model.save_weights('./wait/param_make_sentens_seq2seq.hdf5')
            if flag == "load":
                print("load")
                self.model.load_weights('./wait/param_make_sentens_seq2seq.hdf5')
        except :
            print("no such file")
            sys.exit(0)


    

def main():
    seq2seq = Seq2Seq()
    seq2seq.make_net()

    word = np.array([1,2,3])
    word2 = np.array([2,3,4])

    # one of train
    input_vec = []
    input_vec.append(word)
    input_vec.append(word2)
    input_vec = np.array(input_vec)
    
    # one of train
    input_vec2 = []
    input_vec2.append(word)
    input_vec2.append(word2)
    input_vec2 = np.array(input_vec)

    # batch
    batch_input_vec = []
    batch_input_vec.append(input_vec)
    batch_input_vec.append(input_vec2)
    batch_input_vec = np.array(batch_input_vec)
    print(batch_input_vec.shape)

    # train
    seq2seq.train(batch_input_vec,batch_input_vec)

    # test1
    test_input = np.array([input_vec])
    print(test_input)
    predict = seq2seq.predict(test_input)
    print(predict)


if __name__ == "__main__":
   main()
