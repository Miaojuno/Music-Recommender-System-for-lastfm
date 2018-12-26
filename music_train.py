
from gensim.models import Word2Vec
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from music_w2v import w2v_train
import matrix
from numpy import argmax
from keras.utils import to_categorical
import heapq



#向量输出------------------------------------------------------------------------------------------------------------
class lstm_train:


    def __init__(self,id,seq_length,w2c_length,window,nb_epoch,rawpath):#id：userid   seq_length：每seq_length个推测1   w2c_length：词向量长度  window：词向量训练窗口  nb_epoch:lstm训练次数
        self.seq_length = seq_length
        self.w2c_length = w2c_length
        #词向量模型，源数据
        self.w2v_model,data= w2v_train(id,w2c_length,window,rawpath)
        print(str(id)+'词向量训练完成')
        #构建emb字典
        row_data=open(rawpath,"r", encoding='UTF-8')
        self.dict = matrix.createdict(row_data,id)
        # 构建emb矩阵
        embedding_matrix = matrix.create_embedding_matrix(self.dict,self.w2v_model)
        embedding_matrix = np.array(embedding_matrix)
        row_data.close()

        x_train, y_train, x_test, y_test = self.loadtraindata(data, seq_length, w2c_length)

        model = Sequential()
        model.add(Embedding(len(self.dict), w2c_length, weights=[embedding_matrix], trainable=False))
        model.add(LSTM(units=w2c_length, input_shape=(seq_length, w2c_length)))
        model.add(Dropout(0.2))
        model.add(Dense(w2c_length))
        adam = Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
        model.compile(loss="mse", optimizer=adam, metrics=["accuracy"])
        model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=100, verbose=0)
        self.count, self.all_count=self.testacc(model,x_test,y_test,seq_length,x_train,y_train)
        print(str(id) + ":" + str(self.count) + " " + str(self.all_count))




    #训练集测试集构建 return x_train,y_train,x_test,y_test
    def loadtraindata(self,data,seq_length,w2c_length):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for text_stream in data:
            for i in range(0, len(text_stream) - seq_length):  # 最后10%做测试
                if i < (len(text_stream) - seq_length) * 0.9:
                    given = text_stream[i:i + seq_length]
                    predict = text_stream[i + seq_length]
                    xl=[]
                    for gi in given:
                        xl.append(self.dict[gi])
                    x_train.append(xl)
                    y_train.append(self.w2v_model[predict])
                else:
                    given = text_stream[i:i + seq_length]
                    predict = text_stream[i + seq_length]
                    xl = []
                    for gi in given:
                        xl.append(self.dict[gi])
                    x_test.append(xl)
                    y_test.append(predict)
        x_train = np.reshape(x_train, (-1, seq_length))
        y_train = np.reshape(y_train, (-1, w2c_length))

        return x_train,y_train,x_test,y_test

    #精确率测试
    def testacc(self, model, x_test, y_test, seq_length,x_train,y_train):
        count = 0
        all_count = 0
        for i in range(len(x_train)):
            given = x_train[i]
            predict = y_train[i]
            x = np.reshape(given, (-1, seq_length))
            y = model.predict(x)
            dic = self.w2v_model.wv.similar_by_vector(np.array(y[0]), topn=10)
            all_count += 1
            for x in self.w2v_model.wv.similar_by_vector(predict, topn=1):
                predict=x
            for ww in dic:
                print(predict[0] ,end=' ')
                print( ww[0])
                if  predict[0]== ww[0]:
                    count += 1
        print('训练集' + ":" + str(count) + " " + str(all_count))
        count = 0
        all_count = 0
        for i in range(len(x_test)):
            given = x_test[i]
            predict = y_test[i]
            x = np.reshape(given, (-1, seq_length))
            y = model.predict(x)
            dic = self.w2v_model.wv.similar_by_vector(np.array(y[0]), topn=10)
            all_count += 1
            for ww in dic:
                if predict == ww[0]:
                    count += 1
        return count, all_count


