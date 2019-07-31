
# overfit을 해결하는 방법
# 1. 데이터의 양 증가(비현실적)
# 2. feature(노드) 개수 수정
# 3. regularization(일반화)

# loss 0.01이하

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers

model = Sequential()   # 순서대로 내려가는 모델

# 노드가 5개, 3개, 4개인 레이어 3개를 가진 모델

model.add(Dense(5, input_dim=3, activation='relu'))   
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(15))
model.add(Dense(1))   # 출력 값임 y도 컬럼이 2개

# model.summary()

# 과적합 -> 너무 많은 노드와 레이어에 의해 결과가 떨어지짐

model.save('savetest01.h5')
print('저장완료')