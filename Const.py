"""
定数用
"""
import pylab as plt

class Const():
    def __init__(self):
        self.word_feat_len = 3000
        self.batch_size = 2
        self.learning_num = 100

    def glaph_plot(self,data):
        t = range(len(data))
        plt.plot(t,data)
        plt.show()
