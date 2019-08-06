from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
import pandas as pd
import numpy as np

# 기온 데이터 읽어 들이기
df = pd.read_csv('./data/tem10y.csv', encoding='utf-8')

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year= (df['연'] <= 2015)
test_year = (df['연'] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []   # 학습데이터
    y = []   # 결과
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval: continue

        y.append(temps[i])
        xa = []

        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)

    return(np.array(x), np.array(y))

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

# print(train_x.shape)   # 3646, 6
# print(train_y.shape)   # 3646,

train_x = train_x.reshape( (train_x.shape[0], train_x.shape[1], 1) )
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

# print(train_x.shape)   # 3646, 6, 1
# print(train_y.shape)   # 3646,


model = Sequential()

# R2:0.96
# model.add(LSTM(64, input_shape=(6,1), return_sequences=True))
# model.add(LSTM(16))

# model.add(Dense(6, activation='relu'))
# model.add(Dense(5, activation='relu'))

# model.add(Dense(1))

# R2: 0.97
model.add(LSTM(128, input_shape=(6,1), return_sequences=True))
model.add(LSTM(32))

model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))

model.add(Dense(1))

'''
# R2 :0.97 (고병주)
model.add(LSTM(100, input_shape=(6,1), return_sequences=True))
model.add(LSTM(50))

model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1))
'''
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(test_x, test_y, epochs=300)

loss, mse = model.evaluate(test_x, test_y)

y_predict = model.predict(test_x)   # 모델의 예측값

# R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(test_y, y_predict)   # 1에 가까울수록 좋음
print('R2:', r2_y_predict)   # 0.964