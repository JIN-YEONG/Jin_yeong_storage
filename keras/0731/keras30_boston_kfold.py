from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)

# 데이터 표준화(standardization)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# mean = train_data.mean(axis=0)
# train_data -= mean
# std = train_data.std(axis=0)
# train_data /= std

# test_data -= mean
# test_data /= std

from keras import models
from keras import layers

def build_model():
    # 동일한 모델을 여러번 생성할 것이므로 함수를 만들어 사용
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # mae -> mean absolute error  평균 절대값 오차

    return model

# 데이터를 분할하여 train과 test로 번갈아 가며 사용
# train  /  test
# 1 2 3  /   4
# 1 2    /   3 4
# 1      /   2 3 4


import numpy as np
k = 5
num_epochs = 1
all_score=[]

from sklearn.model_selection import KFold
# stratifiedKFold 구현


skf = KFold(n_splits=k)
# KFold 데이터를 n_split로 나눈다
# ex n_split =3 일때 데이터를 3등분을 3번한다.
# skf.split(x,y)가 실직적으로 데이터를 나눈다.

for partial_index, val_index in skf.split(train_data,train_targets):
    partial_train_data, val_data = train_data[partial_index], train_data[val_index]
    partial_train_targets, val_targets = train_targets[partial_index], train_targets[val_index]
    # print(val_data.shape)
    
    model = build_model()
    
    # 모델 훈련(verbose=0 이므로 훈련과정이 출력되지 않습니다.)
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1 ,verbose=0)
    
    # 검증세트로 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_score.append(val_mae)


'''
num_val_samples  = len(train_data) // k
for i in range(k):
    print("처리중인 폴드 #", i)
    
    # 검증 데이터 준비 
    val_data = train_data[i*num_val_samples : (i+1) * num_val_samples]
    val_targets = train_targets[i*num_val_samples : (i+1) * num_val_samples]
    print('val_data[',i*num_val_samples,',',(i+1) * num_val_samples,']')
    print('val_target[',i*num_val_samples,',',(i+1) * num_val_samples,']')

    # 훈련 데이터 준비: 검증데이터 외의 모든 데이터
    partial_train_data = np.concatenate([train_data[: i*num_val_samples], train_data[(i+1) * num_val_samples :]], axis=0)
                            # concatenate 사슬처럼 연결하다
                            # concatenate((a,b), axis=0)   a값 다음에 b값을 삽입 (행 기준)
                            # concatenate((a,b), axis=1)   a값 안에 b값을 삽입 (열 기준)
                            # concatenate((a,b), axis=None) a값과 b값을 하나로 합친다.

    partial_train_targets = np.concatenate([train_targets[: i*num_val_samples], train_targets[(i+1)*num_val_samples :]], axis=0)
    print('partial_train_data[ [:', i*num_val_samples, '], [',(i+1) * num_val_samples,':] ]')
    print('partial_train_targets[ [:', i*num_val_samples, '], [',(i+1) * num_val_samples,':] ]')
    
    # 케라스 모델 구성(컴파일 포함)
    model = build_model()
    
    # 모델 훈련(verbose=0 이므로 훈련과정이 출력되지 않습니다.)
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1 ,verbose=0)
    
    # 검증세트로 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_score.append(val_mae)
'''
print(all_score)
print(np.mean(all_score)) # 1밑으로 내리기
