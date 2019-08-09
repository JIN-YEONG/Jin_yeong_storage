from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Input
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

batch_size=7
cnn1= 358
cnn2=291
cnn3=361
keep_prob= 0.1
optimizer='adadelta'


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)   #  50000, 32, 32, 3
# print(y_train.shape)   #  50000, 1
# print(x_test.shape)   # 10000,32,32,3
# print(y_test.shape)   # 10000,1


# minmaxscaler
x_train = x_train.reshape(x_train.shape[0], (32*32*3)).astype('float32') 
x_test = x_test.reshape(x_test.shape[0], (32*32*3)).astype('float32') 
# print(x_train.shape)   # 300,3072
# print(x_test.shape)   # 10000, 3072

mm = MinMaxScaler()
mm.fit(x_train)
x_train = mm.transform(x_train)
x_test = mm.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 32,32,3)
x_test = x_test.reshape(x_test.shape[0], 32,32,3)


# # # onehotencoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# # print(y_train.shape)   # (300,10)
# print(y_test.shape)   # (10000,10)

inputs = Input(shape=(32,32,3), name = 'input')
x = Conv2D(cnn1, (2,2),activation='relu', name = 'hidden1')(inputs)
x = Dropout(keep_prob)(x)
x = Conv2D(cnn2, (2,2),activation='relu', name = 'hidden2')(x)
# x = Dropout(keep_prob)(x)
x = Conv2D(cnn3, (2,2),activation='relu', name = 'hidden3')(x)
# x = Dropout(keep_prob)(x)

x = Flatten()(x)
    
prediction = Dense(10, activation='softmax', name = 'output')(x)
model = Model(inputs=inputs, output=prediction)

model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics=['accuracy'])

earlystop = EarlyStopping(monitor='loss', patience=5,mode='auto')
model.fit(x_train,y_train, batch_size = batch_size, epochs=100,callbacks=[earlystop])

print('\n Test Accuracy: %.4f' % (model.evaluate(x_test, y_test)[1]))   # 분류모델에서는 Accuracy를 사용한다.

 
# Test Accuracy: 0.6553
