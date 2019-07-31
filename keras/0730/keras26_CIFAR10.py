# 정확도 80%로 올리기
# 정규화 MinMaxScalar해보고
# standardScalar해서 비교
# 텐서보드와 얼리스탑필 사용
# 사진 한장을 출력(시각화) 확인후 주석처리할것

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt

# 상수정의
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

BATCH_SIZE=128
NB_EPOCH = 50
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2   # x_train, y_train에서 20%를 validation_data로 사용
OPTIM = RMSprop()

# 데이터셋 불러오기
(x_train , y_train), (x_test, y_test) = cifar10.load_data()

print("x_train shape:", x_train.shape)   # (50000,32,32,3)
print(x_train.shape[0] , "train samples")
print(x_test.shape[0], 'test sample')

# 병주형으로 변환
y_train = np_utils.to_categorical(y_train, NB_CLASSES)   # OneHotEnCoding 기계가 빨리 인식하기 때문에 사용
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 실수형으로 지정하고 정규화
x_train = x_train.astype('float32')   # 255로 나눴을때 0~1사이의 값이 되어야 하기 때문에
x_test = x_test.astype('float32')


# 정규화를 위한 데이터 reshape
x_train = x_train.reshape((x_train.shape[0],IMG_ROWS*IMG_COLS*IMG_CHANNELS))
print(x_train.shape)   # (50000,3072)

x_test = x_test.reshape((x_test.shape[0],IMG_ROWS*IMG_COLS*IMG_CHANNELS))
print(x_test.shape)   # (10000,3072)

# 정규화 작업(MinMaxScalar)
mm_scalar = MinMaxScaler()
mm_scalar.fit(x_train)
x_train = mm_scalar.transform(x_train)
x_test = mm_scalar.transform(x_test)
# x_train /= 255   # 정규화 작업
# x_test /= 255

# # StandarScalar()
# std_scalar = StandardScaler()
# std_scalar.fit(x_train)
# x_train = std_scalar.transform(x_train)
# x_test = std_scalar.transform(x_test)

# 데이터 shape 복구
x_train = x_train.reshape((x_train.shape[0], IMG_COLS, IMG_ROWS, IMG_CHANNELS))
x_test = x_test.reshape((x_test.shape[0], IMG_COLS, IMG_ROWS, IMG_CHANNELS))
# print(x_train.shape)
# print(x_test.shape)

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

# 얼리스탑핑과 텐서보드
early_stopping = EarlyStopping(monitor='acc',patience=30, mode='auto')
tb_his = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

history = model.fit(x_train, y_train , batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,verbose=VERBOSE,callbacks=[early_stopping,tb_his])

print("Testing ----")
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

print('\nTest score:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())

plt.plot(history.history['acc'])   # 그래프에 들어갈 값1
plt.plot(history.history['val_acc'])   # 그래프에 들어갈 값2
plt.title('model accuracy')   # 그래프의 제목
plt.ylabel('accuracy')   # y축이름
plt.xlabel('epoch')   # x축 이름
plt.legend(['train', 'test'], loc='upper left')   # 그래프를 설명하는 작은 박스, 외쪽 위에 위치
                            # loc 를 안쓸경우 알아서 빈찬에 출력
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 데이터시각화

digit = x_train[30]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
