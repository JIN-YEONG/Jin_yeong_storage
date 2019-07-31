# dnn으로 만들기 70%

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import RMSprop
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

x_train = x_train.reshape((x_train.shape[0],32*32*3))
x_test = x_test.reshape((x_test.shape[0],32*32*3))


mm_scaler  = MinMaxScaler()
mm_scaler.fit(x_train)
x_train = mm_scaler.transform(x_train)
x_test = mm_scaler.transform(x_test)

# # StandarScalar()
# std_scalar = StandardScaler()
# std_scalar.fit(x_train)
# x_train = std_scalar.transform(x_train)
# x_test = std_scalar.transform(x_test)

# 병주형으로 변환
y_train = np_utils.to_categorical(y_train, NB_CLASSES)   # OneHotEnCoding 기계가 빨리 인식하기 때문에 사용
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(512, input_shape=(IMG_ROWS*IMG_COLS*IMG_CHANNELS,), activation='relu'))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='acc',patience=30, mode='auto')
# tb_his = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

history = model.fit(x_train, y_train , batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,verbose=VERBOSE,callbacks=[early_stopping]) # , tb_his])

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

# # 데이터시각화
# digit = x_train[30]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
