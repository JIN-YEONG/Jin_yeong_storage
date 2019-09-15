# 1. 데이터 
import numpy as np

x =  np.array([range(100), range(311,411), range(100)])
y =  np.array([range(501, 601), range(711, 811), range(100)])

x = np.transpose(x)   # 데이터의 행, 열 변환
y = np.transpose(y)

# print(x.shape)   # 데이터의 크기를 확인
# print(y.shape)

# 데이터 분할
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4   # test size 40%
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5   # test size 50%
)


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()   # 순서대로 내려가는 모델

model.add(Dense(5, input_dim=3, activation='relu'))   # input_dim = 입력 데이터의 칼럼(열)수
# model.add(Dense(5, input_shape=(3,), activation='relu'))  # input_dim 대신 input_shape 사용
model.add(Dense(17))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))   # 출력 값 y의 컬럼의 수

# model.summary()   # 모델의 layer수, 노드 수, param의 수를 출력


# 3. 훈련
# 다:다 데이터 삽입
# model.compile(loss='mse', optimizer='adam', metrics=['accuarcy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val,y_val))
    # epochs = 반복횟수, validation_data = 검증 데이터 

# 4. 평가 예측
lose, acc = model.evaluate(x_test, y_test, batch_size=1)   # 테스트 데이터를 이용한 모델 평가
print('acc: ', acc)

y_predict = model.predict(x_test)   # x_test 데이터를 이용한 y값 예측
print(y_predict)

# RMSE   (Root Mean Square Error - 평균 제곱근 오차)
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # root(mean((y_test - y_predict)^2))

print('RMSE: ', RMSE(y_test, y_predict))   # 작을 수록 좋음

# R2
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)   # 1에 가까울수록 좋음
print('R2:', r2_y_predict)