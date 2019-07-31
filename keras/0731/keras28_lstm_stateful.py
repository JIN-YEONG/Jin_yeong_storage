# 상태유지 lstm


# 문제
# 1.mse값을 1이하로 만들것 -> 3개 이상의 히든 레이어 추가, 드랍아웃/batchnormalization 적용
# 2.RMSE함 적용
# 3.r2 함수 적용
# 4.earlystoping기능 적용
# 5.tensorboard 적용
# 6.matplotlib 이미지 적용 mse/epochs


# 값이 범위 밖으로 나가면 잘 맞지 않는다.


import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

a = np.array(range(1,101))   # 100개의 데이터
batch_size = 1

size = 5
def split_5(seq, size):   # 한 행에 seq의 각 원소부터 size개를 원소로 갖게 분할
    aaa=[]
    for i in range(len(seq) - size +1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))

    return np.array(aaa)

dataset = split_5(a, size)

print("===============================")
print(dataset)
print(dataset.shape)

x_train = dataset[:,0:4]
y_train = dataset[:,4]

x_train = np.reshape(x_train, (len(x_train),size-1, 1))

x_test = x_train + 100
y_test = y_train + 100


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(100, batch_input_shape=(batch_size,4,1), stateful=True))
                    # batch_input_shape=(batch_size, 컬럼수, 데이터 자르는 수)
                    # 데이터를 자르는 수는 한번에 비교할 데이터의 수
                    # batch_size는 데이터 묶음을 만들기 위해 자르는 수
                    # stateful=True 상태를 유지
# model.add(Dropout(0.3))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

model.add(Dense(100, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# model.summary()

model.compile(loss='mse', optimizer='adam',metrics=['mse'])

early_stopping = EarlyStopping(monitor='val_mse', patience=10, mode='auto')
tb_hist = TensorBoard(log_dir='./graph', histogram_freq = 0, write_graph=True, write_images=True)


his = []
num_epochs = 100
# 상태유지의 가장 큰 특징(여러번의 fit)
for epoch_idx in range(num_epochs):
    print("epochs: " +str(epoch_idx))
    his.append(model.fit(x_train,y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False, validation_data=(x_test,y_test),callbacks=[early_stopping]))#,tb_hist]))
                                                                # shuffle = False  데이터를 석지 않는다 -> 훈련된 상태를 그대로 유지
                                                                # fit을 했을때 초기화 시키지 않는다.
    model.reset_states()   # 상태유지를 위해 꼭 필요한 구문
                            # 훈련한 데이터가 사라지지는 않는다

mse, _ =model.evaluate(x_train, y_train, batch_size=batch_size)
print('mse: ', mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=batch_size)
print(y_predict[0:10])


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):   # 평균 제곱근 오차
    return np.sqrt(mean_squared_error(y_test, y_predict))   # root(mean((y_test - y_predict)^2))

print('RMSE: ', RMSE(y_test, y_predict)) 

# R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)   # 1에 가까울수록 좋음
print('R2:', r2_y_predict)

# print(his[0].history['mean_squared_error'])

mse_data=[]
for data in his:
    mse_data.append( data.history['mean_squared_error'])


plt.plot(mse_data)   # 그래프에 들어갈 값1
plt.title('model mse')   # 그래프의 제목
plt.ylabel('mse')   # y축이름
plt.xlabel('epoch')   # x축 이름
plt.legend(['mse'], loc='upper left')   # 그래프를 설명하는 작은 박스, 외쪽 위에 위치
                            # loc 를 안쓸경우 알아서 빈찬에 출력
plt.show()
 