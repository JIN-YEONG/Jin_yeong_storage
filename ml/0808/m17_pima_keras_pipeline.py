# pima-indians-diabetes.csv를 파이프 라인처리하시오
# 최적의 파라미터를 구한뒤 모델리해서 acc 확인

import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 데이터 로드
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")   # 경로의 './' 은 현재 경로(study)
                                                                          # 경로의 '../'은 상위 폴더

x = dataset[:,0:8]
y = dataset[:,8]
# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

# 시퀀스형 모델도 잘 작동한다.
def mk_model(node_nums = 5, optimizers= 'adam'):
    model = Sequential()
    model.add(Dense(node_nums, input_dim =8 , activation='relu'))
    model.add(Dense(node_nums, activation='relu'))
    model.add(Dense(node_nums, activation='relu'))
    model.add(Dense(node_nums, activation='relu'))
    model.add(Dense(node_nums, activation='relu'))
    model.add(Dense(node_nums, activation='relu'))
    model.add(Dense(node_nums, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))   # 이진 분류모델을 사용하기 위해 activation='sigmoid' 사용

    model.compile(loss='binary_crossentropy', optimizer=optimizers, metrics = ['accuracy'])   #  이진 분류모델 에서는 loss='binary_crossentropy를 사용

    return model

def create_hyperparameter():
    node_nums = [24,34,56,200,100,150,230]
    optimizers = ['rmsprop', 'adam', 'adadelta']

    return {'model__node_nums' : node_nums, 'model__optimizers': optimizers}


model = KerasClassifier(build_fn=mk_model)

parameter = create_hyperparameter()

kfold_cv = KFold(n_splits=5, shuffle=True)

pipe = Pipeline( [ ('scaler', StandardScaler()), ('model', model)])   # 파이프 라인사용

search = RandomizedSearchCV(pipe, parameter, n_iter=20, cv=kfold_cv, n_jobs=-1)
search.fit(x_train, y_train)

print('최적의 매개변수', search.best_params_)

y_pred = search.predict(x_test)

print('최종 정답률', search.score(x_test,y_test))

'''
최적의 매개변수 {'model__optimizers': 'adam', 'model__node_nums': 200}
최종 정답률 0.7662337639115073
'''