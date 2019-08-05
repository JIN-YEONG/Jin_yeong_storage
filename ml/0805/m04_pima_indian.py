

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import tensorflow as tf

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")   # 경로의 './' 은 현재 경로(study)
                                                                          # 경로의 '../'은 상위 폴더

X = dataset[:,0:8]
Y = dataset[:,8]

 
# 모델 설정 (이진 분류모델)
model = Sequential()
model.add(Dense(24, input_dim =8 , activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))

model.add(Dense(1, activation='sigmoid'))   # 이진 분류모델을 사용하기 위해 activation='sigmoid' 사용

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])   #  이진 분류모델 에서는 loss='binary_crossentropy를 사용

# 모델 실행
model.fit(X,Y, epochs = 200, batch_size=10)

# 결과 출력
print('\n Accuracy : %.4f' % (model.evaluate(X,Y)[1]))
