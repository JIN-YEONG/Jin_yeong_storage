# CNN 모델로 변경


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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


# 하이퍼 파리미터 최적화


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv2D, Flatten
import numpy as np


def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1), name = 'input')
    x = Conv2D(512, kernel_size=(3,3), activation='relu', name = 'hidden1')(inputs)
    x = MaxPooling2D(pool_size = 2)(x)
    x = Conv2D(256, kernel_size=(3,3), activation='relu', name = 'hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Conv2D(128, kernel_size=(3,3), activation='relu', name = 'hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', name = 'hidden4')(x)
    x = Dropout(keep_prob)(x)

    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(keep_prob)(x)

    prediction = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, output=prediction)
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    
    return {'batch_size': batches, 'optimizer': optimizers, 'keep_prob': dropout}

from keras.wrappers.scikit_learn import KerasClassifier   # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)   # 교차검증을 하기위해 이런한 형태로 사용

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=model, # 모델을 사용해 cv가능
    param_distributions=hyperparameters,
    n_iter=10,   # epochs=10
    n_jobs=1,   # ???????????
    cv=3,   # 3조각으로 나눠서 3번 작업
    verbose=1
)

search.fit(X_train, Y_train)

print(search.best_params_)   # 제일 좋은 파리미터 출력 (hyperparameter 값 출력)