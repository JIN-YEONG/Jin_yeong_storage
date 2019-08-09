

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , KFold
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np
import pandas as pd

iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 구분
y= iris_data.loc[:,'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# print(x_train.shape)   # 120,4
# print(y_train.shape)   # 120

def create_model(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(4,), name='input')
    l = Dense(50, activation = 'relu', name='hidden0')(inputs)
    l = Dense(50, activation='relu',name='hidden1')(l)
    l = Dense(25, activation='relu', name='hidden2')(l)

    prediction = Dense(3, activation='softmax', name ='output')(l)

    model = Model(inputs=inputs, output= prediction)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    
    return {'batch_size': batches, 'optimizer': optimizers, 'keep_prob': dropout}

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model)   # ML에서 사용할수 있는 keras분류 모델

parameters = create_hyperparameters()
kfold_cv = KFold(n_splits=5, shuffle=True)

clf = GridSearchCV(model, parameters, cv=kfold_cv)

clf.fit(x_train, y_train)

# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)

