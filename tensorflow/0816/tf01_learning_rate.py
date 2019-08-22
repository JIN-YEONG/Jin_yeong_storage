 
# 데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델
model.add(Dense(5, input_dim=1, activation='relu'))   # input_dim 입력 노드 개수
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))   # 출력노드 개수



# 훈련
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, Nadam, RMSprop, SGD

# optimizer= Adam(lr=0.036)   # lr= learning_rate
# optimizer = Adagrad(lr=0.04)
# optimizer = Adadelta(lr=0.77)
optimizer = Adamax(lr=0.06)   # mse:  1.4210854715202004e-14 [[1.5      ] [2.5000002] [3.4999995]]
# optimizer = Nadam(lr=0.057)   # mse:  2.8183677613924374e-11 [[1.499998 ] [2.4999955] [3.499993 ]]
# optimizer = RMSprop(lr=0.01)   # mse:  0.0019019381925318157 [[1.5030563] [2.5307326] [3.5584083]]
# optimizer = SGD(lr=0.01)   # mse:  1.6496670696142246e-09  [[1.5000147] [2.4999819] [3.4999495]]

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

model.fit(x,y,epochs=100, batch_size = 1)   

'''

최적의 w 는 최소의 loss(cost)을 가진다.
leaning_rate란 w와 loss의 그래프에서 얼마나 잘라서 기울기를 구할지
정하는 것 (기울기가 0인 경우가 최적의 w값)

경사 하강법: 최적의 w값을 찾기위해 모델의 모든 w를 표현하는 그래프에서 
            한 지점의 기울기를 계산하여 최소의 값(cost)을 찾는 방법

learning_rate: 경사하강법의 그래프에서 한번에 얼마나 이동할지를 결정하는 수치
                값이 너무 크면 그래프를 벗어나 이동하는 overshooting이 발생하거나
                값이 너무 작으면 최적의 cost를 찾는데 너무 많은 시간이 걸린다.
                적절한 값을 구하기 위해서것은 경험의 영역이기 때문에 여러번 실행하는 것이 최고이다.
                


optimizer
    SGD(lr=0.01)
    RMSprop(lr=0.001)
    Adagrad(lr=0.01)
    Adaelta(lr=1.0)
    Adam(lr=0.001)
    Adamax(lr=0.002)
    Nadam(lr=0.002)

'''


# 평가 예측
mse,_ =model.evaluate(x,y,batch_size=1)   # mse의 경우 loss값과 mse가 동일
print('mse: ',mse)

pred1 = model.predict([1.5,2.5,3.5])
print(pred1)
 