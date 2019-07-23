# 모델의 최적화를 위해 변경해야하는 값
# Deep, Node, epochs, batch_size


# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

# 트레이닝과 테스트 데이터를 나눈이유
# 다른 데이터를 사용해서 보다 정확한 평가를 하기위해

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()   # 순서대로 내려가는 모델

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델
# 노드와 레이어의 수가 적을 수록 최적화된 레이어다
model.add(Dense(8, input_dim=1, activation='relu'))   # input_dim -> 입력 노드 개수
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))   # 출력노드 개수
                      # Dense -> DNN을 사용

# model.summary()

# summary의 Param값이 예상과 다르게 나오는 이유
# y=wx+b의 b(바이오스) 값이 존재하기 때문에
# 레이어마다 1개의 바이오스 값이 존재한다.
# param의 개수 = (input의 개수 + 1(바이오스의 수)) * output의 개수


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])   # 모델 컴파일
# model.fit(x,y,epochs=100, batch_size = 3)   # epochs -> (에포)모델 반복 횟수
                                            # batch_size -> 몇개씩 잘라서 작업할지 결정
                                            # fit ->훈련 실행
                                            # batch_size -> 잘라서 작업할 데이터의 수
model.fit(x_train,y_train,epochs=500)

# 4. 평가 예측
lose,acc =model.evaluate(x_test,y_test,batch_size=1)   # evaluate -> 모델 평가
                                             # x,y에는 테스트용 데이터가 들어간다
print('acc: ',acc)   # 분류모델에 대한 결과이기 때문에
                     # predict 했을 때 정확한 값이 다르게 나온다.

y_predict = model.predict(x_test)   # 모델의 예측값
                                    
print(y_predict)
