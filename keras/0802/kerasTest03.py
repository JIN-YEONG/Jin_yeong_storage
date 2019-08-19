#-*- coding: utf-8 -*-

# 수정 필요 (데이터의 개수)



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

batchSize = 1
file_path = './0802/data/kospi200test.csv'
repeat_num = 10
# num_epochs = 100

file_data = pd.read_csv(file_path, encoding='euc-kr')
# print(file_data.head())


file_data.drop('Unnamed: 7' ,axis=1, inplace=True)
file_data=file_data.sort_values(by=['일자']) # 시간순으로 정렬
file_data.drop('일자', axis=1, inplace=True)
file_data.drop('거래량', axis=1, inplace = True)
file_data.drop('환율(원/달러)', axis=1, inplace = True)


price = file_data.values[:,:-1]
scaler = MinMaxScaler()
scaler.fit(price)
mm_price = scaler.transform(price)

volume = file_data.values[:,-1:]
scaler.fit(volume)
mm_volume = scaler.transform(volume)


x = np.concatenate((mm_price, mm_volume), axis=1)
# y = x[:,3]   # MinMaxScaler()를 적용한 y값
# y =np.array(file_data['종가'].values)
y = file_data['종가'].values
# print(x.shape)   # (599,4)
# print(y.shape)   # (599,)



size = 2
def split_5(seq, size):   # seq를 size 단위로 나눠 행을 구분
    aaa=[]
    for i in range(len(seq) - size):
        subset = seq[i+1:(i+size)+1]
        aaa.append([item for item in subset])
    print(type(aaa))

    return np.array(aaa)

y = split_5(y, size)
x = x[:-2]   # 데이터의 크기를 맞추기 위해 마지막 행 제외
             # y에서 데이터를 2개씩 묶었기 때문에 598개의 데이터가 나온다.


print(x.shape)   # (598, 4)
print(y.shape)   # (598,2)


# 랜덤한 데이터 분할
train_x, test_x , train_y, test_y = train_test_split(
    x, y, random_state = 82, test_size=0.4
)

val_x, test_x ,val_y, test_y = train_test_split(
    test_x, test_y, random_state = 82, test_size = 0.5
)

train_x = train_x.reshape((train_x.shape[0], train_x.shape[1],1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1],1))
val_x = val_x.reshape((val_x.shape[0], val_x.shape[1],1))

# # DNN용 데이터 shape변환
# train_x= train_x.reshape(356, 5*6)
# test_x= test_x.reshape(119, 5*6)
# val_x= val_x.reshape(119,5*6)

print(train_x.shape)
print(train_y.shape)
print('-'*10)
print(test_x.shape)
print(test_y.shape)
print('-'*10)
print(val_x.shape)
print(val_y.shape)
print('-'*10)

# x_predict = file_data.values[:5,:]
# print(x_predict)
# print(x_predict.shape)


model = Sequential()
model.add(LSTM(80, batch_input_shape=(batchSize,4,1), stateful=True))
# model.add(LSTM(119, batch_input_shape=(batchSize,4,1)))


model.add(Dense(100,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(356))
# model.add(Dropout(0.3))
model.add(Dense(30))
# model.add(BatchNormalization())
# model.add(Dense(119))
# model.add(Dropout(0.2))
# model.add(Dense(475))
# model.add(BatchNormalization())
# model.add(Dense(231))
# model.add(Dropout(0.2))
model.add(Dense(58))
# model.add(BatchNormalization())
model.add(Dense(81))
# # model.add(Dropout(0.2))
# model.add(Dense(122))
# model.add(Dropout(0.2))

model.add(Dense(2))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=3, mode='auto')

# 상태유지 LSTM용
for rp_id in range(repeat_num):
    print('num:' + str(rp_id))
    model.fit(train_x, train_y, 
        epochs=10, batch_size=batchSize, verbose=2, 
        shuffle=False, validation_data=(val_x, val_y), callbacks=[early_stopping]
    )
    model.reset_states()

# DNN용 fit
# model.fit(train_x, train_y, epochs=repeat_num, batch_size=batchSize, verbose=2, validation_data=(val_x, val_y), callbacks=[early_stopping])


loss, mse = model.evaluate(test_x, test_y, batch_size=batchSize)
print('mse:', mse)
model.reset_states()


x_predict = file_data.values[-1:,:]
x_predict = x_predict.reshape((x_predict.shape[0], x_predict.shape[1],1))
# print(x_predict)
# print(x_predict.shape)
# print(x_predict[:,-1:])
y_predict = model.predict(x_predict)

# y_predict = scaler.inverse_transform(y_predict)
# y_predict = np.mean(y_predict)
print(y_predict)
