import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1.데이터
a = np.array(range(1,11))

size = 5
def split_5(seq, size):   # seq를 size 단위로 나눠 행을 구분
    aaa=[]
    for i in range(len(seq) - size +1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))

    return np.array(aaa)

dataset = split_5(a, size)
print("=============================")
# print(dataset)

# 데이터 분할
x_train = dataset[:,0:4]
y_train = dataset[:,4]
print(x_train)
print(y_train)

print(x_train.shape)   # (6,4)
print(y_train.shape)   # (6,)


# x_train = np.reshape(x_train, (6,4,1))
x_train = np.reshape(x_train, (len(a)-size+1,4,1))
# print(x_train.shape)   # (6,4,1)


x_test = np.array([[[11],[12],[13],[14]], 
                   [[12],[13],[14],[15]], 
                   [[13],[14],[15],[16]], 
                   [[14],[15],[16],[17]]])

y_test = np.array([15,16,17,18])

print(x_test.shape)   # (4,4,1)
print(y_test.shape)   # (4,)

# 2. 모델구성
model = Sequential()

# LSTM 여러개 쌓기
model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10))
# Dense 결합
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# model.summary()

# 3.훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
            callbacks=[early_stopping])


loss,acc = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)

print('loss:', loss)
print('acc:', acc)
print(y_predict)
