

# RNN (Recurrent Neural Network) 순환 신경망
# 시배열(연속된) 데이터의 분석의 유리
# 시배열(time distributed) -> 연속된 데이터



from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# print('x.shape: ', x.shape)
# print('y.shape: ', y.shape)   # 결과 값의 개수를 알수 있다.

x = x.reshape((x.shape[0], x.shape[1],1))   # x의 shape를 (4,3,1)로 변경(데이터의 개수는 변함 없음)
                                            # 행은 무시하고 (3,1)이 input_shape에 들어간다.
print('x.shape: ', x.shape)


# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))   # inpurt_shape=(칼럼의 수, 자를 데이터수)
                                                            # 10 -> 아웃풋 노드의 개수,Dense에서 사용
                                                            # 이미 많은 파라미터를 가지고 있어서 에포를 올리는 것이 효과적일 수 있다.
model.add(Dense(28))
model.add(Dense(30))
model.add(Dense(14))
model.add(Dense(1))

# model.summary()

#################과제###################
# LSTM의 Param의 개수가 왜 많은지 답하시오
# (인풋 3 + 바이어스 1) * ? * 아웃풋 10 = 480
# ? = 12 -> '?'가 어디서 나왔는가?
########################################

# 3.실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping   # 모델의 훈련을 조기에 종료 시킬 수 있다.
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')   # loss값을 기준으로 훈련을 멈춤
                                                                            # parience => 동일한loss가 반복되는 횟수, 반복되면 종료

model.fit(x,y,epochs=10000,batch_size=1, callbacks=[early_stopping])   # callbacks 속성으로 조기 종료 조건 전달

x_input = array([25,35,45])   # (1,3,?)
# x_input = array([70,80,90])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input, verbose=0)
print(yhat)
