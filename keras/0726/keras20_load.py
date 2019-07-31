
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
from keras.models import load_model
model = load_model("savetest01.h5")


# 3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # mse = mean squared error 평균 제곱 에러
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # metrics=['mse'] 결과 값이 mse값으로 나온다
# model.fit(x,y,epochs=100, batch_size = 3)   
model.fit(x_train,y_train,epochs=1000, batch_size=1, validation_data=(x_val, y_val),callbacks=[early_stopping])   # validation_data = 검증을 위한 데이터 셋

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