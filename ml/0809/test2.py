from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV,KFold
import numpy as np

(x_train, y_train) , (x_test, y_test) = cifar10.load_data()

x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=300)

data_generator = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.02,
    height_shift_range= 0.02,
    horizontal_flip=True
)

i=0
for x, y in data_generator.flow(x_train, y_train ,batch_size=200):
    
    x_train = np.append(x_train, x, axis=0)
    y_train = np.append(y_train, y, axis=0)

    i += 1
    # print(i)
    if i>300:
        break

# print(x_train.shape)   # 45500,32,32,3
# print(y_train.shape)   # 45500,1


# minmaxscaler
x_train = x_train.astype('float32') / 255
x_test  = x_test.astype('float32')/ 255

# x_train = x_train.reshape(x_train.shape[0], 32*32*3)
# x_test = x_test.reshape(x_test.shape[0], 32*32*3)

# mm= MinMaxScaler()
# mm.fit(x_train)
# x_train = mm.transform(x_train)
# x_test = mm.transform(x_test)

# x_train = x_train.reshape(x_train.shape[0], 32,32,3)
# x_test = x_test.reshape(x_test.shape[0], 32,32,3)


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def create_model(cnn1=1,cnn2=1,cnn3=1,drop=0.1,dnn1=1,dnn2=2, padding='same', optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(cnn1, (2,2),input_shape=(32,32,3), activation='relu', padding=padding))
    model.add(Dropout(drop))
    model.add(Conv2D(cnn2, (2,2), padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(cnn3, (2,2), padding=padding))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())

    model.add(Dense(dnn1))
    model.add(Dense(dnn2))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

parameters = {
    'cnn1':np.arange(1,200,13),
    'cnn2' :np.arange(1,200,13),
    'cnn3' :np.arange(1,200,13),
    'drop':np.arange(0.1,0.8,0.1),
    'dnn1':np.arange(1,200,13),
    'dnn2':np.arange(1,200,13),
    'padding':['same','valid'],
    'optimizer': ['rmsprop', 'adam', 'adadelta'],
    'batch_size' :np.arange(1,100,3)
}

model = KerasClassifier(build_fn=create_model)
kfold_cv = KFold(n_splits=5,shuffle=True)
search = RandomizedSearchCV(model,parameters,
                            n_iter=10, n_jobs= -1, cv=kfold_cv
)

search.fit(x_train,y_train)

print('최적 파라미터', search.best_params_)
print("최적 정확도:", search.score(x_test,y_test))

# # 값 랜덤하게 섞기
# s = np.arrange(x_train.shape[0])
# np.random.shuffle(s)
# x_train= x_train[s]
# y_train = y_train[s]
