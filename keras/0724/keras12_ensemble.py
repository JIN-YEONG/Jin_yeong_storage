
# 1. 데이터
import numpy as np

# x = np.array(range(1,101))
# y = np.array(range(1,101))

# 여러개의 데이터가 들어가서 1개의 데이터가 나온다
x1 = np.array([range(100), range(311,411),range(100)])
y1 = np.array([range(501,601),range(711, 811), range(100)])
x2 = np.array([range(100,200), range(311,411),range(100,200)])
y2 = np.array([range(501,601),range(711, 811), range(100)])

# print(x.shape)
# print(y.shape)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)


from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=66, test_size = 0.4   # test_size = 40%
)

x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state = 66, test_size = 0.5
)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=66, test_size = 0.4   # test_size = 40%
)

x2_val, x2_test, y2_val, y2_test = train_test_split(
    x2_test, y2_test, random_state = 66, test_size = 0.5
)


print(x2_test.shape)

# 2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input

# 앙상블
# 여러모델의 합칠수 있다.

# 모델1
input1 = Input(shape=(3,))
dense1 = Dense(100, activation='relu')(input1)  
dense1_2 = Dense(30)(dense1)    # Dense(output)(input)
dense1_3 = Dense(7)(dense1_2)

# 모델2
input2 = Input(shape=(3,))
dense2 = Dense(50,activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

# 모델 병합
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])   # 각 모델의 최종 output을 리스트로 대입

# 병합된 모델을 사용하는 모델3
middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)

#############################################

# output모델
output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(middle3)
output2_1 = Dense(70)(output2)
output2_2 = Dense(3)(output2_1)

model = Model(input=[input1, input2],output=[output1_3, output2_2])

# model.summary()

# 과적합 -> 너무 많은 노드와 레이어에 의해 결과가 떨어지짐


# 3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # mse = mean squared error 평균 제곱 에러
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # metrics=['mse'] 결과 값이 mse값으로 나온다
# model.fit(x,y,epochs=100, batch_size = 3)   
model.fit([x1_train, x2_train],[y1_train,y2_train],
            epochs=100, batch_size=1,
            validation_data=([x1_val, x2_val],[y1_val,y2_val]))   # validation_data = 검증을 위한 데이터 셋

# 4. 평가 예측
_,_,_, acc1, acc2 = (model.evaluate([x1_test,x2_test],[y1_test,y2_test],batch_size=1))   
print('acc1: ',acc1)   # acc는 회귀모델에서만 사용할 수 있다.
print('acc2:', acc2)
print('------------------------------------')
y1_predict, y2_predict = model.predict([x1_test,x2_test])   # 모델의 예측값
print(y1_predict)
print('------------------------------')
print(y2_predict)


# RMSE 구하기
# 
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):   # 평균 제곱근 오차
    return np.sqrt(mean_squared_error(y_test, y_predict))   # root(mean((y_test - y_predict)^2))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)

print('RMSE1: ', RMSE1)  # 작을 수록 좋다.
print('RMSE2: ', RMSE2)
print('RMSE: ', (RMSE1 + RMSE2)/2)

# R2 구하기
from sklearn.metrics import r2_score

r2_y1_predict = r2_score(y1_test, y1_predict)   # 1에 가까울수록 좋음
r2_y2_predict = r2_score(y2_test, y2_predict)   # 1에 가까울수록 좋음

print('R2_1:', r2_y1_predict)
print('R2_2:', r2_y2_predict)
print('R2: ', (r2_y1_predict + r2_y2_predict)/2)

