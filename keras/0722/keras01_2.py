
# 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])
x2 = np.array([4,5,6])


# 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델
model.add(Dense(100, input_dim=1, activation='relu'))   # input_dim 입력 노드 개수
model.add(Dense(75))
model.add(Dense(59))
model.add(Dense(1))   # 출력노드 개수


# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x,y,epochs=2000, batch_size = 1)   # epochs (에포)모델 반복 횟수   4. 5. 6.0000005
model.fit(x,y,epochs=100, batch_size = 1)   # epochs (에포)모델 반복 횟수


# 평가 예측
lose,acc =model.evaluate(x,y,batch_size=1)   # evaluate모델 평가
print('acc: ',acc)

y_predict = model.predict(x2)   # 새로운 값을 모델에 적용
print(y_predict)
