# 0722 과제

##############과제##############
# keras04_homework.py
# x_train = 1~100
# y_trian = 501 ~ 600
# x_test = 1001 ~ 1100
# y_test = 1101 ~ 1200
# 위 데이터를 이용하여 모델 만들기
################################

# 데이터
import numpy as np

x_train = np.arange(1,101)
y_train = np.arange(501,601)
x_test = np.arange(1001,1101)
y_test = np.arange(1101,1201)

# x_train_mean = int(np.mean(x_train))

# y_train_mean = int(np.mean(y_train))


# x_test_mean = int(np.mean(x_test))

# y_test_mean = int(np.mean(y_test))


# input_data_x = np.arange(x_train_mean, x_test_mean+1)
# input_data_y = np.arange(y_train_mean, y_test_mean+1)

# print(input_data_x)
# print(input_data_y)
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)


# 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(50,input_dim=1, activation='relu'))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(80))
model.add(Dense(1))


# 훈련
model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,epochs=1000)

# 평가
lose, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc:', acc)

y_predict = model.predict(x_test)
print(y_predict)
