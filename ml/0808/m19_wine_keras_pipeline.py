from keras.models import Sequential, Model
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV , KFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import np_utils
import pandas as pd
import numpy as np

# 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv', sep=';',encoding='utf-8')

# 데이터를 레이블과 데이터로 분리
y= wine['quality']
x= wine.drop('quality', axis=1)

# y레이블 변경하기
# y레이블의 값에 따라 세가지의 값으로 변경
newlist = []
for v in list(y):
    if v <=4:
        newlist += [0]   # newlist = newlist.append(0)
    elif v <= 7:
        newlist += [1]   # newlist = newlist.append(1)
    else:
        newlist += [2]   # newlist = newlist.append(2)

y = newlist
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# print(x_train.shape)
# print(y_train.shape)


def create_model(node_num1=1, node_num2=1,optimizer = 'adam'):
    inputs = Input(shape=(11,), name='inputs')
    
    x = Dense(node_num1, activation='relu', name='hidden1')(inputs)
    x = Dense(node_num1, activation='relu', name='hidden2')(x)
    x = BatchNormalization()(x)
    x = Dense(node_num1, activation='relu', name='hidden3')(x)
    x = Dense(node_num2, activation='relu', name='hidden4')(x)
    x = Dense(node_num2, activation='relu', name='hidden5')(x)
    x = BatchNormalization()(x)
    x = Dense(node_num2, activation='relu', name='hidden6')(x)


    predict = Dense(3, activation ='softmax', name='output')(x)

    model = Model(inputs = inputs, outputs=predict)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def create_hyperparameter():
    node_nums1 = [10,34,56,33,200,100,400]
    node_nums2 = [5,13,24,36,130,180,270]
    optimizers = ['rmsprop', 'adam', 'adadelta']


    return {'model__node_num1' : node_nums1, 'model__node_num2' : node_nums2,
            'model__optimizer': optimizers}

model = KerasClassifier(build_fn=create_model)

parameter = create_hyperparameter()

kfold_cv = KFold(n_splits=5, shuffle=True)
pipe = Pipeline([ ('scaler', MinMaxScaler()), ('model',model)])
clf = RandomizedSearchCV(pipe, parameter, n_iter=10, cv=kfold_cv, n_jobs=-1)   # n_jobs 병렬 CPU사용 값
clf.fit(x_train, y_train)

print("최적의 매개변수= ", clf.best_estimator_)
# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)

'''
최적의 매개변수=  Pipeline(memory=None,
     steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('model', <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A93D475550>)])
최종 정답률 =  0.9224489795918367
최종 정답률 =  0.9224489793485525
'''