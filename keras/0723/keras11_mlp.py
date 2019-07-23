
# 1. 데이터
import numpy as np

# x = np.array(range(1,101))
# y = np.array(range(1,101))
x = np.array([range(100), range(311,411),range(100)])
y = np.array([range(501,601),range(711,811),range(100)])

x = np.transpose(x)
y = np.transpose(y)
# print(x.shape)
# print(y.shape)





from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size = 0.4   # test_size = 40%
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test,y_test, random_state = 66, test_size = 0.5
)



# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()   # 순서대로 내려가는 모델

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델

model.add(Dense(5, input_dim=3, activation='relu'))   # input_dim = 입력 데이터의 컬넘의 개수
                                                      # 데이터의 행과 상관없이 열의 개수만 맞아도 데이터를 넣을 수 있다.
# model.add(Dense(10, input_shape=(2,), activation='relu'))   # input_shape = 데이터의 shape를 기준으로 입력
model.add(Dense(17))
model.add(Dense(11))
model.add(Dense(5))
model.add(Dense(3))   # 출력 값임 y도 컬럼이 2개

# model.summary()

# 과적합 -> 너무 많은 노드와 레이어에 의해 결과가 떨어지짐


# 3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # mse = mean squared error 평균 제곱 에러
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # metrics=['mse'] 결과 값이 mse값으로 나온다
# model.fit(x,y,epochs=100, batch_size = 3)   
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_data=(x_val, y_val))   # validation_data = 검증을 위한 데이터 셋

# 4. 평가 예측
lose,acc = model.evaluate(x_test,y_test,batch_size=1)   
print('acc: ',acc)   # acc는 회귀모델에서만 사용할 수 있다.

y_predict = model.predict(x_test)   # 모델의 예측값
print(y_predict)

# RMSE 구하기
# 
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):   # 평균 제곱근 오차
    return np.sqrt(mean_squared_error(y_test, y_predict))   # root(mean((y_test - y_predict)^2))
# 루트를 씨우는 이유 
# 값을 작게 만들기 위해

print('RMSE: ', RMSE(y_test, y_predict))   # 작을 수록 좋다.

# R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)   # 1에 가까울수록 좋음
print('R2:', r2_y_predict)
