from keras.datasets import mnist
(train_image, train_labels), (test_image, test_labels) = mnist.load_data()

from keras import models
from keras import layers

# 모델
network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
              # optimizer='adam'대신 사용

# 데이터
# 이미지가 컬러면 dnn은 어떻게 하나?
train_image = train_image.reshape((60000,28*28)) 
train_image = train_image.astype('float32') / 255

test_image = test_image.reshape((10000,28*28))
test_image = test_image.astype('float32') / 255

print(train_image.shape)   # (60000,784)
print(test_image.shape)   # (10000,784)


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.summary()

# 훈련
network.fit(train_image, train_labels, epochs=5 , batch_size=128)
# 평가
test_loss , test_acc = network.evaluate(test_image, test_labels)

print('test_acc:', test_acc)
