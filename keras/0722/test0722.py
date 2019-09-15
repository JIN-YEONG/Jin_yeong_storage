

import numpy as np
x_train = np.arange(1,11)
y_train = np.arange(1,11)
x_test = np.arange(11, 21)
y_test = np.arange(11, 21)
# print(x_train)   # [ 1  2  3  4  5  6  7  8  9 10]
# print(y_train)   # [ 1  2  3  4  5  6  7  8  9 10]
# print(x_test)   # [11 12 13 14 15 16 17 18 19 20]
# print(y_test)   # [11 12 13 14 15 16 17 18 19 20]


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.fit(x_train,y_train, epochs=100,batch_size=1)

loss, acc= model.evaluate(x_test, y_test, batch_size=1)

print('acc', acc)

y_pred = model.predict(x_test)
print('Predict\n', y_pred)

'''
acc 1.0
Predict
 [[10.986959]
 [11.98387 ]
 [12.980781]
 [13.977692]
 [14.974604]
 [15.971513]
 [16.968426]
 [17.96534 ]
 [18.96225 ]
 [19.959158]]
'''


