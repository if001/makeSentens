from lib import Const
from lib import WordVec as wv

fname = Const.Const().word2vec_train_file

#w2v = wv.MyWord2Vec().train(fname, "save")
model = wv.MyWord2Vec().load_model()

print("corpus: ", model.corpus_count)
voc = model.wv.vocab.keys()
print("vocab: ", len(voc))
print(voc)

# vec = wv.MyWord2Vec().str_to_vector(model, "引か")
# print(vec)
# import pylab as plt
# plt.plot(vec)
# plt.show()
