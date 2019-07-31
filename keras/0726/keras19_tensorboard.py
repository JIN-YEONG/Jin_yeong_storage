
# overfit을 해결하는 방법
# 1. 데이터의 양 증가(비현실적)
# 2. feature(노드) 개수 수정
# 3. regularization(일반화)




# 1. 데이터
import numpy as np

# x = np.array(range(1,101))
# y = np.array(range(1,101))

# 여러개의 데이터가 들어가서 1개의 데이터가 나온다
x = np.array([range(1000), range(3110,4110),range(1000)])
y = np.array([range(5010,6010)])


print(x.shape)
print(y.shape)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size = 0.4   # test_size = 40%
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test,y_test, random_state = 66, test_size = 0.5
)



# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers

model = Sequential()   # 순서대로 내려가는 모델

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델

model.add(Dense(30, input_dim=3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))   
                                                    # kernel_regularizer=regularizers
                                                    # -> 일반화 과정,가중치 규제, 모델 복잡도에 대한 패널티로 가중치가 작은 값을 가지도록 강제
                                                    # -> 과적합(overfitting) 예방
                                                    #  regularizers.l1(0.01)  릿지
                                                    # -> 가중치 행렬의 모든 원소의 절대값에 0.01을 곱하여 전체손실에 더한다.
                                                    # regularizers.l2(0.01)  라스
                                                    # -> 가중치 행렬의 모든 원소를 제곱하고 0.01을 곱하여 전체 손실에 더한다.
                                                
# model.add(Dense(10, input_shape=(3,), activation='relu'))   # input_shape = 데이터의 shape를 기준으로 입력
model.add(Dense(17,kernel_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())   # 정규화 과정
model.add(Dense(6,kernel_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.1))
model.add(Dense(23))
model.add(Dense(1))   # 출력 값임 y도 컬럼이 2개

# model.summary()

# 과적합 -> 너무 많은 노드와 레이어에 의해 결과가 떨어지짐




# 텐서보드 노드의 모양을 그레픽으로 출력
import keras
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq = 0, write_graph=True, write_images=True)




# 3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # mse = mean squared error 평균 제곱 에러
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # metrics=['mse'] 결과 값이 mse값으로 나온다
# model.fit(x,y,epochs=100, batch_size = 3)   
model.fit(x_train,y_train,epochs=100, batch_size=1, validation_data=(x_val, y_val),callbacks=[early_stopping,tb_hist])   # validation_data = 검증을 위한 데이터 셋

# 4. 평가 예측
loss,acc = model.evaluate(x_test,y_test,batch_size=1)   
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
print('loss:', loss)