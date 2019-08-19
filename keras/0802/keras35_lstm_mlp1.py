# input_shape(4,1)의 lstm

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))

size = 8


def split_10(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    
    return np.array(aaa)


dataset = split_10(x,size)
# print(dataset.shape)   # (93,8)

x_train = dataset[:,0:4]
y_train = dataset[:,4:]
# print(x_train.shape)   # (93,4)
# print(y_train.shape)   # (93,4)


x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, random_state=66, train_size = 0.6
)


x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)   # (55,4,1)
print(x_test.shape)   # (38,4,1)


print(y_train.shape)   # (55,4)
print(y_test.shape)   # (38,4) 


model = Sequential()

# LSTM 여러개 쌓기
model.add(LSTM(32, input_shape=(4,1)))
# Dense 결합
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
            callbacks=[early_stopping])


loss,acc = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)

print('loss:', loss)
print('acc:', acc)
# print(y_predict)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):   # 평균 제곱근 오차
    return np.sqrt(mean_squared_error(y_test, y_predict))   # root(mean((y_test - y_predict)^2))
# 루트를 씨우는 이유 
# 값을 작게 만들기 위해

print('RMSE: ', RMSE(y_test, y_predict))   # 작을 수록 좋다.

# R2 구하기
from sklearn.metrics import r2_score


# 0.95 만들기
r2_y_predict = r2_score(y_test, y_predict)   # 1에 가까울수록 좋음
print('R2:', r2_y_predict)