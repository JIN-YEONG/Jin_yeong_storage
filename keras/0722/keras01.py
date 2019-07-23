
# 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델
model.add(Dense(5, input_dim=1, activation='relu'))   # input_dim 입력 노드 개수
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))   # 출력노드 개수

# 머신 러닝 -> 딥러닝에 비해 빠른 결과 산출
# 딥 러닝 -> 정확하지만 느린 결과 산출

# y= wx +b
# x와 y값이 주어진다.
# b값은 작은 값이기 때문에 w값이 중요하다.




# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x,y,epochs=100, batch_size = 1)   # epochs 모델 반복 횟수
                                            # fit 훈련 실행
                                            # batch_size 잘라서 작업할 데이터의 수



# 평가 예측
lose,acc =model.evaluate(x,y,batch_size=1)
print('acc: ',acc)

 