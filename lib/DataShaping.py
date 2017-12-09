

import numpy as np
import lib
import random
TMP_BUCKET = (10,15)

class DataShaping():
    def __init__(self):
        self.str_op = lib.StringOperation.StringOperation("load")

    def select_random_sentens(self,word_lists):
        index = random.randint(0,len(word_lists)-2)
        __sentens = word_lists[index]
        return __sentens


    def select_random_sentens2(self,word_lists):
        index = random.randint(0,len(word_lists)-2)
        __sentens1 = word_lists[index]
        __sentens2 = word_lists[index+1]
        return __sentens1,__sentens2


    def select_bucket(self,sentens_arr,flag=0):
        """ flag=0 is train, flag=1 is teach """
        __buckets = self.str_op.buckets[:]
        __buckets.append((100, 100))
        index = 0
        for i in range(len(__buckets)-1):
            if (len(sentens_arr) > __buckets[i][flag]):
                index = __buckets.index(__buckets[i+1])
        return index


    def train_data_shaping(self, train_sentens_vec_batch, train_sentens, seq_num):
        if "BOS" in train_sentens: train_sentens.remove("BOS")
        train_sentens = self.str_op.EOF_padding(train_sentens, seq_num)
        train_sentens = train_sentens[::-1]
        train_sentens_vec = self.str_op.sentens_array_to_vec(train_sentens)
        train_sentens_vec_batch.append(train_sentens_vec)
        return train_sentens_vec_batch


    def teach_data_shaping(self, teach_sentens_vec_batch, teach_sentens, seq_num):
        teach_sentens = self.str_op.EOF_padding(teach_sentens, seq_num)
        teach_sentens_vec = self.str_op.sentens_array_to_vec(teach_sentens)
        teach_sentens_vec_batch.append(teach_sentens_vec)
        return teach_sentens_vec_batch


    def teach_target_data_shaping(self, teach_target_sentens_vec_batch, teach_sentens, seq_num):
        if "BOS" in teach_sentens: teach_sentens.remove("BOS")
        teach_target_sentens = self.str_op.EOF_padding(teach_sentens, seq_num)
        teach_target_sentens_vec = self.str_op.sentens_array_to_vec(teach_target_sentens)
        teach_target_sentens_vec_batch.append(teach_target_sentens_vec)
        return teach_target_sentens_vec_batch


    def make_data(self, word_lists, batch_size, chose_bucket):
        train_sentens_vec_batch = []
        teach_sentens_vec_batch = []
        teach_target_sentens_vec_batch = []

        for j in range(batch_size):
            __word_lists = word_lists[::]
            while(True):
                train_sentens,teach_sentens = self.select_random_sentens2(__word_lists)
                train_sentens = self.str_op.reshape_sentens(train_sentens)
                train_sentens = self.str_op.rm_BOS(train_sentens)
                teach_sentens = self.str_op.reshape_sentens(teach_sentens)
                bucket_index = lib.Const.Const().buckets.index(chose_bucket)
                if((self.select_bucket(train_sentens,0) <= bucket_index) and (self.select_bucket(teach_sentens,1) == bucket_index) ): break

            train_sentens_vec_batch = self.train_data_shaping(train_sentens_vec_batch, train_sentens, chose_bucket[0])
            teach_sentens_vec_batch = self.teach_data_shaping(teach_sentens_vec_batch, teach_sentens, chose_bucket[1])
            teach_target_sentens_vec_batch = self.teach_target_data_shaping(teach_target_sentens_vec_batch, teach_sentens, chose_bucket[1])

        train_sentens_vec_batch = np.array(train_sentens_vec_batch)
        teach_sentens_vec_batch = np.array(teach_sentens_vec_batch)
        teach_target_sentens_vec_batch = np.array(teach_target_sentens_vec_batch)
        print("train shape:",train_sentens_vec_batch.shape)
        print("teach shape:",teach_sentens_vec_batch.shape)
        print("target shape:",teach_target_sentens_vec_batch.shape)

        return train_sentens_vec_batch, teach_sentens_vec_batch, teach_target_sentens_vec_batch

