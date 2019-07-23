
# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x2 = np.array([4,5,6])


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()   # 순서대로 내려가는 모델

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델
model.add(Dense(5, input_dim=1, activation='relu'))   # input_dim 입력 노드 개수
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))   # 출력노드 개수

model.summary()

# summary의 Param값이 예상과 다르게 나오는 이유
# y=wx+b의 b(바이오스) 값이 존재하기 때문에
# 레이어마다 1개의 바이오스 값이 존재한다.
# param의 개수 = (input의 개수 + 1(바이오스의 수)) * output의 개수

'''
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])   # 모델 컴파일
# model.fit(x,y,epochs=100, batch_size = 3)   # epochs (에포)모델 반복 횟수
                                            # batch_size 몇개씩 잘라서 작업할지 결정
                                            # fit 훈련 실행
                                            # batch_size 잘라서 작업할 데이터의 수
model.fit(x,y,epochs=100)

# 4. 평가 예측
lose,acc =model.evaluate(x,y,batch_size=3)   # evaluate 모델 평가
                                             # x,y에는 테스트용 데이터가 들어간다
print('acc: ',acc)   # 분류모델에 대한 결과이기 때문에
                     # predict 했을 때 정확한 값이 다르게 나온다.

y_predict = model.predict(x2)   # 새로운 값을 모델에 적용
print(y_predict)
'''