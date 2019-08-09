
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
sy = pd.Series(cancer.target, dtype="category")
sy = sy.cat.rename_categories(cancer.target_names)
df['class'] = sy
# print(df.tail())

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(x.shape)   # 569, 30
print(y.shape)   # 569

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


def create_model(node_num = 20, optimizer = 'adam'):
    inputs = Input(shape=(30,), name='inputs')
    
    x = Dense(node_num, activation='relu', name='hidden1')(inputs)
    x = Dense(node_num, activation='relu', name='hidden2')(x)
    x = Dense(node_num, activation='relu', name='hidden3')(x)
    x = Dense(node_num, activation='relu', name='hidden4')(x)
    x = Dense(node_num, activation='relu', name='hidden5')(x)


    predict = Dense(1, activation ='sigmoid', name='output')(x)

    model = Model(inputs = inputs, outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def create_hyperparameter():
    node_nums = [10,34,56,33,200,100,400]
    optimizers = ['rmsprop', 'adam', 'adadelta']


    return {'model__node_num' : node_nums, 'model__optimizer': optimizers}

model = KerasClassifier(build_fn=create_model)

parameter = create_hyperparameter()

kfold_cv = KFold(n_splits=5, shuffle=True)
pipe = Pipeline([ ('scaler', MinMaxScaler()), ('model',model)])
clf = RandomizedSearchCV(pipe, parameter, n_iter=10, cv=kfold_cv, n_jobs=4)   # n_jobs 병렬 CPU사용 값
clf.fit(x_train, y_train)

'''
검증 값의 과적합을 피하기 위해 pipeline을 사용하고
먼저 x_train에서 train_data와 val_data를 분리하고 나서 전처리를 진행 해야 하지 때문에
RandomizedSearchCV를 실행하여 데이터를 분리하고 pipeline에 넣어 전처리 후 모델을 실행한다.


clf = RandomizedSearchCV(model, parameter, n_iter=10, cv=kfold_cv, n_jobs=4)
pipe = Pipeline([ ('scaler', MinMaxScaler()), ('model',clf)])
pipe.fit(x_train, y_train)

위과 같이 Pipeline안에 clf를 넣어서도 실행가능한데 
이경우 x_train 전체를 전처리 하고나서 train_data와 val_data를 분리하기 때문에 
다시 과적합 문제가 발생한다.
'''


print("최적의 매개변수= ", clf.best_estimator_)
# 최적의 매개 변수로 평가하기
# accuracy_score 와 score 의 결과가 같다.
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)
