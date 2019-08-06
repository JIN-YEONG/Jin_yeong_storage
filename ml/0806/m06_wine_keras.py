# 정확도 70% 이상 만들기

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv', sep=';',encoding='utf-8')

# 데이터를 레이블과 데이터로 분리
y= wine['quality']
x= wine.drop('quality', axis=1)

# print(y.value_counts())   # 7개의 데이터

y= np_utils.to_categorical(y)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# print(x_train.shape)   # 3918,11
# print(y_train.shape)   # 3918,10

model = Sequential()
model.add(Dense(150, input_dim=11, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(75))
model.add(Dropout(0.5))
model.add(Dense(90))
model.add(Dense(34))
model.add(Dropout(0.2))
model.add(Dense(55))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(80))
# model.add(BatchNormalization())
# model.add(Dense(75))
# model.add(BatchNormalization())
# model.add(Dense(175))
# model.add(Dropout(0.4))

model.add(Dense(10, activation = 'softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

early_stop = EarlyStopping(monitor='acc', patience=10, mode='auto')
model.fit(x_train, y_train,epochs=10000,batch_size=32, callbacks=[early_stop])

loss, acc = model.evaluate(x_test, y_test)
print('acc:', acc)   # 0.514

# y_pred = model.predict(x_test)
# y_pred = np.argmax(y_pred,axis=1)
# print(y_pred)
