# m05_iris_keras.py를 RandomSearch 적용

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV , KFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# # OneHotEncoding   자동으로 실행됨
# y_train = np_utils.to_categorical(y_train, 3)
# y_test = np_utils.to_categorical(y_test, 3)

# print(x_train.shape)   # 120,4
# print(y_train.shape)   # 120

# keras 모델 생성
def create_model(node_num1=1, node_num2=1,optimizer= 'adam'):
    inputs= Input(shape=(4,), name='input')
    l = Dense(node_num1, activation = 'relu', name='hidden1')(inputs)
    l = Dense(node_num1, activation = 'relu', name='hidden2')(l)
    l = Dense(node_num1, activation = 'relu', name='hidden3')(l)
    l = BatchNormalization()(l)
    l = Dense(node_num2, activation = 'relu', name='hidden4')(l)
    l = Dense(node_num2, activation = 'relu', name='hidden5')(l)
    l = Dense(node_num2, activation = 'relu', name='hidden6')(l)
    l = BatchNormalization()(l)
    
    prediction = Dense(3, activation='softmax',name='output')(l)   # y_train.shape는 (120,)이지만 OneHotEncoding이 자동으로
                                                                    # 수행되기 때문에 출력shape는 3이어야 한다.
    
    model = Model(inputs=inputs, output=prediction)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameters():
    node_nums1=[10,34,56,33,200,100,400]
    node_nums2 = [5,13,24,36,130,180,270]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    

    return {'model__node_num1': node_nums1, 'model__node_num2': node_nums2, 'model__optimizer': optimizers}

model = KerasClassifier(build_fn=create_model)
parameter = create_hyperparameters()
pipe = Pipeline( [ ('scaler', MinMaxScaler()), ('model', model)])   # 파이프라인
kfold_cv = KFold(n_splits=5, shuffle=True)

clf = RandomizedSearchCV(pipe, parameter, n_iter=10, cv=kfold_cv, n_jobs=-1)
clf.fit(x_train, y_train)

print("최적의 매개변수= ", clf.best_estimator_)
print('최적의 파라미터=', clf.best_params_)

# 최적의 매개 변수로 평가하기
y_pred = clf.predict(x_test)
# print(y_pred)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
# accuracy_score 와 score 의 결과가 같다.
# last_score = clf.score(x_test, y_test)
# print("최종 정답률 = ", last_score)



'''
최적의 매개변수=  Pipeline(memory=None,
     steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('model', <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F3DA7A0EF0>)])
최적의 파라미터= {'model__optimizer': 'adam', 'model__node_num2': 180, 'model__node_num1': 33}
최종 정답률 =  0.8666666666666667
'''