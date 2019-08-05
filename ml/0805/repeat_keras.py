from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

'''keras의 분류 모델을 이용한 iris예제'''

name = ['a', 'b', 'c', 'd', 'e']
dataset = pd.read_csv('./data/iris.csv', names=name, encoding='utf-8')   # 열이름에 name값을 주면서 csv파일 읽기
# print(dataset.shape)

# calumn 명을 이용한 데이터 분할
x = dataset.loc[:, ['a','b','c','d']]
y = dataset.loc[:, 'e']

# y가 숫자가 아니기 때문에 숫자로 변경
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

y = np_utils.to_categorical(y,3)   # OnHotEncoding

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, test_size=0.2)   # 데이터 분할
# print(x_train.shape)   # 120,4
# print(y_train.shape)   # 120,3

# 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(5))

model.add(Dense(3, activation='softmax'))   # 분류모델이기 때문에 softmax사용

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
# 분류모델이기 때문에 categorical_crossentropy사용

# 실행
earlystopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.fit(x_train, y_train, epochs=200, batch_size=1, callbacks=[earlystopping])

# 검증 
loss , acc = model.evaluate(x_test, y_test, batch_size=1)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)   # reverse OneHotEncoding
# print(y_predict)
y_predict = le.inverse_transform(y_predict)   # 숫자데이터 문자데이터 변경

print('acc:', acc)
print(y_predict)
