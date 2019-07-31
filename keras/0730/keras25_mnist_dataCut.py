
# 회귀모델 -> 예측
# 분류모델 -> 분류(지정된 값으로만 출력)


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping


import numpy
import os
import tensorflow as tf


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# # 데이터시각화
# import matplotlib.pyplot as plt
# digit = X_train[59000]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# 300개의 데이터 만들 사용하여 정확도 올리기(95%)


# 데이터 분할
from sklearn.model_selection import train_test_split

X_train, _, Y_train, _ = train_test_split(
    X_train, Y_train, random_state=66, train_size = 0.005  
)


# print(X_train.shape)   # (300,28,28)
# print(Y_train.shape)   # (300,)



X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32') / 255   # shape(60000, 28,28,1)   데이터의 개수가 6만개   ???????????????reshape 함수 확인
                                                                               # 0~1 으로 만들기 위해 255로 나눔(데이터 전처리)
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32') / 255



# Y값들은 0~9사이의 값을 가져야 한다.
Y_train = np_utils.to_categorical(Y_train)  # OneHotEnCoding
                                            # 표현하고 싶은 값의 인덱스의 위치에 1을 부여 나머지는 0부여 ex)3: 0001000000, 5: 0000010000
                                            # y는 결과값이기 때문에 0~9값을 갖는다.
Y_test = np_utils.to_categorical(Y_test)


# OneHotEnCoding
# 컴퓨터에서의 단어 표현방법
# 표현하고 싶은 단어의 인덱스에 1을 부여하고 다른 인덱스에는 0을 부여하여 표현하는 방법


# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu', padding='same'))
model.add(Conv2D(60, (3,3), activation='relu',padding='same'))
model.add(Conv2D(30, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=5)) # (2,2) 커널
model.add(Dropout(0.5))
model.add(Flatten())   # 데이터 피기
model.add(Dense(30, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(10,activation='softmax'))  # 지정된 분류모델을 사용하기 위해 activation='softmax'사용
                                           # to_categorical()을 사용했기 때문에 10개의 값이 나온다.

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # 분류모델이기 때문에(activation='softmax') loss='categorical_crossentropy' 사용


# activatioin = 'softmax' 와 loss = 'categorical_crossentropy' 는 한세트로 분류 모델에 사용


early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=50, batch_size=30, verbose=1, callbacks=[early_stopping_callback])

print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))   # 분류모델에서는 Accuracy를 사용한다.
