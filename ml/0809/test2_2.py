from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

cnn1=1
cnn2=1
cnn3=1
drop=0.1
padding='same'
dnn1=1
dnn2=1
batch_size=32
optimizer='adam'


(x_train,y_train), (x_test, y_test) = cifar10.load_data()

# minmaxsclar
x_train = x_train.astype('float32') / 255
x_test = x_train.astype('float32') / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

inputs = Input(shape=(32,32,3))
x = Conv2D(cnn1, (2,2), activation='relu', padding=padding)(inputs)
x = Dropout(drop)(x)
x = Conv2D(cnn2, (2,2), activation='relu', padding=padding)(x)
x = BatchNormalization()(x)
x = Conv2D(cnn3, (2,2), activation='relu', padding=padding)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)

x = Dense(dnn1)(x)
x = Dense(dnn2)(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs = outputs)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=100)

loss ,acc = model.evaluate(x_test, y_test)

print('Test Accuracy', acc)
