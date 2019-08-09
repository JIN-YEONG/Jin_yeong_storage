'''
1. 데이터를 300개로 자른다.
2. generator 사용 증폭
3. hyperparameter튜닝   소스1  증포된 데이터
4. 최종소스                 소스2   원데이터 6만개
5. acc 제출
4. 총 5회까지 제출가능

kingkeras@naver.com
메일제목 홍길동 99.9[1차]
메일 내용 없음
메일 첨부 이름_tset01.py    이름_test02.py   acc 스크린샷(전체화면)
'''

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Input
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)   #  50000, 32, 32, 3
# print(y_train.shape)   #  50000, 1
# print(x_test.shape)   # 10000,32,32,3
# print(y_test.shape)   # 10000,1

x_train, _, y_train, _ = train_test_split(x_train,y_train, train_size = 300)
# print(x_train.shape)   # 300,32,32,3
# print(y_train.shape)   # 300,1

data_generator = ImageDataGenerator(
    rotation_range=20,   # 회전정도
    width_shift_range= 0.02,   # 넓이 길이 조정
    height_shift_range= 0.02,   # 높이 길이 조정
    horizontal_flip= True
)

i=0
for x, y in data_generator.flow(x_train, y_train ,batch_size=200):
    
    x_train = np.append(x_train, x, axis=0)
    y_train = np.append(y_train, y, axis=0)

    i += 1
    # print(i)
    if i>300:
        break

print(x_train.shape)
print(y_train.shape)

# minmaxscaler
x_train = x_train.reshape(x_train.shape[0], (32*32*3)).astype('float32') 
x_test = x_test.reshape(x_test.shape[0], (32*32*3)).astype('float32') 
# print(x_train.shape)   # 300,3072
# print(x_test.shape)   # 10000, 3072

mm = MinMaxScaler()
mm.fit(x_train)
x_train = mm.transform(x_train)
x_test = mm.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 32,32,3)
x_test = x_test.reshape(x_test.shape[0], 32,32,3)


# # # onehotencoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# # print(y_train.shape)   # (300,10)
# print(y_test.shape)   # (10000,10)

####################################################################

def build_network(cnn1=1,cnn2=1,cnn3=1,keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(32,32,3), name = 'input')
    x = Conv2D(cnn1, (2,2),activation='relu', name = 'hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Conv2D(cnn2, (2,2),activation='relu', name = 'hidden2')(x)
#     x = Dropout(keep_prob)(x)
    x = Conv2D(cnn3, (2,2),activation='relu', name = 'hidden3')(x)
#     x = Dropout(keep_prob)(x)

    x = Flatten()(x)
    
    prediction = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, output=prediction)
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics=['accuracy'])

    return model

def create_hyperparameters():
    batches = np.arange(2,100)
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    cnn1 = np.arange(1,500)
    cnn2 = np.arange(1,501)
    cnn3 = np.arange(1,501)
    
    return {'batch_size': batches, 'optimizer': optimizers, 'keep_prob': dropout, 'cnn1':cnn1, 'cnn2': cnn2, 'cnn3':cnn3}

from keras.wrappers.scikit_learn import KerasClassifier   # 사이킷런과 호환하기 위해 사용
model = KerasClassifier(build_fn=build_network, verbose=1)   # 교차검증을 하기위해 이런한 형태로 사용

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=model, # 모델을 사용해 cv가능
    param_distributions=hyperparameters,
    n_iter=10,   # epochs=10
    n_jobs=1,   # ???????????
    cv=3,   # 3조각으로 나눠서 3번 작업
    verbose=1
)

search.fit(x_train, y_train)

# print(search.best_params_)   # 제일 좋은 파리미터 출력 (hyperparameter 값 출력)

'''
def create_model(cnn1=1, dnn1=1, keep_drop=0.1):
    
    inputs = Input(shape=(32,32,3))
    x = Conv2D(cnn1, (2,2),activation='relu')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(cnn1, (2,2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)

    x= Dense(dnn1,activation='relu')(x)
    
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='categorical_crossentorpy', metrics=['accuracy'])

    return model

# def create_model(cnn1=1, dnn1=1, drop=0.1):
#     model = Sequential()

#     model.add(Conv2D(cnn1, (2,2), padding='same', input_shape=(32,32,3), activation='relu')) 
#     model.add(Dropout(drop))
#     model.add(Conv2D(cnn1, (2,2), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))

#     model.add(Flatten())

#     model.add(Dense(dnn1, activation='relu'))
#     model.add(Dense(10, activation='softmax'))

#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return model


parameters ={
    'cnn1':[np.arange(1,201)],
    'dnn1':[np.arange(1,200)], 
    'keep_drop':[np.arange(0.1,0.9,0.1)]#, 
    # 'pad':['valid','same'],
    # 'optimizer':['rmsprop', 'adam', 'adadelta']
}

model  = KerasClassifier(build_fn=create_model)
kfold_cv = KFold(n_splits=5, shuffle= True)
search = RandomizedSearchCV(model,
                            parameters,
                            n_iter=10,
                            cv=kfold_cv                         
)

search.fit(x_train, y_train)
'''
y_predict = search.predict(x_test)
print(y_predict)
print('최적의 매개변수 = ',search.best_params_)
print('훈련 정확도 =', search.score(x_test,y_test))

# # R2 구하기
# from sklearn.metrics import r2_score

# r2_y_predict = r2_score(y_test, y_predict)   # 1에 가까울수록 좋음
# print('R2:', r2_y_predict)




# 최적의 매개변수 =  {'optimizer': 'adadelta', 'keep_prob': 0.1, 'cnn3': 361, 'cnn2': 291, 'cnn1': 358, 'batch_size': 7}
# 훈련 정확도 = 0.9978681319952011

