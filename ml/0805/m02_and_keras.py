from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]   
y_data = [0,0,0,1]                    

# list데이터는 넣을 수 없기 때문에 array로 변경
x_data = np.array(x_data)
y_data = np.array(y_data)

# print(x_data.shape)   # (4,2)

# 모델
model = Sequential()

model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])


'''
model.add(Dense(1, activation='sigmoid))
이진 분류모델을 사용하기 위해 activation='sigmoid'사용 

sigmoid는 0~1의 값이 출력되며 이는 확률로 읽을 수 있는데 
50%가 넘으면 1(True) 50%보다 작으면 0(False)로 구분 할 수 있다.(round() 이용하여 반올림)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
이진 분류모델 에서는 loss='binary_crossentropy를 사용
'''


# 3. 실행
model.fit(x_data, y_data,batch_size=1,epochs=100)         

# 4. 평가 예측
loss ,acc = model.evaluate(x_data, y_data, batch_size=1)

x_test = [[0,0], [1,0], [0,1], [1,1]]
x_test = np.array(x_test)
y_predict = model.predict(x_test)

print('예측 결과: ', y_predict.round())
print('acc= ', acc)  # y_data와 y_predict가 얼마나 일치하는지 비교


