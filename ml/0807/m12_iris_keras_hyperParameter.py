# m05_iris_keras.py를 RandomSearch 적용

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV , KFold
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
import pandas as pd
import numpy as np

iris_data = pd.read_csv('./data/iris2.csv', encoding = 'utf-8')

# iloc 열의 순서를 이용한 나누기
x= iris_data.iloc[:, 0:4]
y= iris_data.iloc[:, 4]   # y.type = str


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, test_size =0.2
)

# # OneHotEncoding   자동으로 실행되는 듯하다.
# y_train = np_utils.to_categorical(y_train, 3)
# y_test = np_utils.to_categorical(y_test, 3)

# print(x_train.shape)   # 120,4
# print(y_train.shape)   # 120

# keras 모델 생성
def create_model(node_num=20, optimizer= 'adam'):
    inputs= Input(shape=(4,), name='input')
    l = Dense(node_num, activation = 'relu', name='hidden1')(inputs)
    l = Dense(node_num, activation = 'relu', name='hidden2')(l)
    l = Dense(node_num, activation = 'relu', name='hidden3')(l)
    l = Dense(node_num, activation = 'relu', name='hidden4')(l)
    l = Dense(node_num, activation = 'relu', name='hidden5')(l)
    l = Dense(node_num, activation = 'relu', name='hidden6')(l)
    
    prediction = Dense(3, activation='softmax',name='output')(l)   # y_train.shape는 (120,)이지만 출력shape는 3이어야 한다.
    
    model = Model(inputs=inputs, output=prediction)


    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameters():
    node_nums=[10,20,30,32,40,50,60,70]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    

    return {'node_num': node_nums, 'optimizer': optimizers}

model = KerasClassifier(build_fn=create_model)

parameter = create_hyperparameters()
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(model, parameter, n_iter=20, cv=kfold_cv)

clf.fit(x_train, y_train)
print("최적의 매개변수= ", clf.best_estimator_)

# 최적의 매개 변수로 평가하기
y_pred = clf.predict(x_test)
print(y_pred)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
# accuracy_score 와 score 의 결과가 같다.
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)
