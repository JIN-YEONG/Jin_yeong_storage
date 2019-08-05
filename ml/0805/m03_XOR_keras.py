from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1. 데이터
# AND 값
x_data = [[0,0], [1,0], [0,1], [1,1]]   
y_data = [0,1,1,0]                    

x_data = np.array(x_data)
y_data = np.array(y_data)

# 2. 모델
# model = LinearSVC()   # 머신러닝 회귀모델

model = Sequential()

model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(64))
model.add(Dense(10))
model.add(Dense(10))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])



# 3. 실행
model.fit(x_data, y_data,batch_size=1,epochs=100)         

# 4. 평가 예측
loss ,acc = model.evaluate(x_data, y_data, batch_size=1)

x_test = [[0,0], [1,0], [0,1], [1,1]]
x_test = np.array(x_test)

y_predict = model.predict(x_test)

print('예측 결과: ', y_predict)
print('acc= ', acc)  # y_data와 y_predict가 얼마나 일치하는지 비교


